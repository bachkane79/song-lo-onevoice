"""Test tầng điều phối pipeline — không cần micro, không cần model thật.

Dùng máy dịch giả để kiểm tra đúng phần logic dễ sai: chọn chiều dịch, hàng đợi
quá tải, xử lý lỗi. Test có model thật nằm ở `test_translate.py`.
"""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from onevoice.config import PipelineConfig  # noqa: E402
from onevoice.display import Display, DisplayEvent  # noqa: E402
from onevoice.pipeline import OneVoicePipeline, _pick_lang  # noqa: E402
from onevoice.stt import MockSttEngine, SttResult  # noqa: E402
from onevoice.translate import TranslationResult  # noqa: E402


class FakeTranslator:
    """Máy dịch giả — trả về chuỗi có thể đoán trước, đếm số lần được gọi."""

    def __init__(self, delay_s=0.0):
        self.delay_s = delay_s
        self.calls = []

    def warmup(self):
        pass

    def translate(self, text, src_lang, tgt_lang=None, beam_size=None):
        if self.delay_s:
            time.sleep(self.delay_s)
        self.calls.append((text, src_lang))
        tgt = "vi" if src_lang == "zh" else "zh"
        return TranslationResult(source=text, target="[%s→%s] %s" % (src_lang, tgt, text),
                                 src_lang=src_lang, tgt_lang=tgt, latency_ms=1.0)

    def stats(self):
        return {"compute_type": "fake", "disk_mb": 0, "device": "cpu",
                "beam_size": 1, "zh_unk_rules": 0, "glossary_entries": 0}


class CollectDisplay(Display):
    def __init__(self):
        self.events = []
        self.statuses = []

    def show(self, event):
        self.events.append(event)

    def show_status(self, message):
        self.statuses.append(message)

    def close(self):
        pass


def build(stt, cfg=None, translator=None):
    disp = CollectDisplay()
    pipe = OneVoicePipeline(config=cfg or PipelineConfig(), stt_engine=stt,
                            display=disp, translator=translator or FakeTranslator())
    return pipe, disp


# ── Chọn chiều dịch ─────────────────────────────────────────────────────────
class TestPickLang:
    def test_forced_wins(self):
        assert _pick_lang("zh", "vi", "Xin chào") == "zh"

    def test_stt_lang_used_when_no_force(self):
        assert _pick_lang(None, "vi", "你好") == "vi"

    def test_falls_back_to_script_detection(self):
        assert _pick_lang(None, None, "你好") == "zh"
        assert _pick_lang(None, None, "Xin chào") == "vi"

    def test_unknown_stt_lang_falls_back(self):
        # Whisper có thể trả "en", "ja"... — không dùng được thì đoán theo chữ
        assert _pick_lang(None, "ja", "你好") == "zh"

    def test_returns_none_when_undecidable(self):
        assert _pick_lang(None, None, "12345") is None


# ── Luồng end-to-end (không mic, không model) ───────────────────────────────
class TestPipelineFlow:
    def test_translates_each_utterance(self):
        stt = MockSttEngine(script=[("zh", "你好"), ("vi", "Xin chào")])
        pipe, disp = build(stt)
        pipe.run_offline([(None, 16000, 1.0)] * 2)

        assert len(disp.events) == 2
        assert disp.events[0].src_lang == "zh" and disp.events[0].tgt_lang == "vi"
        assert disp.events[1].src_lang == "vi" and disp.events[1].tgt_lang == "zh"
        assert pipe.stats["translated"] == 2
        assert pipe.stats["dropped"] == 0

    def test_offline_never_drops(self):
        """Chế độ offline phải chờ hàng đợi chứ không bỏ câu như luồng mic."""
        cfg = PipelineConfig()
        cfg.queue_maxsize = 1
        stt = MockSttEngine(script=[("zh", "你好%d" % i) for i in range(10)])
        pipe, disp = build(stt, cfg, FakeTranslator(delay_s=0.005))
        pipe.run_offline([(None, 16000, 1.0)] * 10)

        assert len(disp.events) == 10
        assert pipe.stats["dropped"] == 0

    def test_empty_stt_result_is_skipped(self):
        class SilentStt(MockSttEngine):
            def transcribe(self, audio, sample_rate):
                return SttResult(text="   ")

        pipe, disp = build(SilentStt())
        pipe.run_offline([(None, 16000, 1.0)] * 3)
        assert disp.events == []
        assert pipe.stats["stt_empty"] == 3

    def test_undecidable_language_is_skipped(self):
        class NumberStt(MockSttEngine):
            def transcribe(self, audio, sample_rate):
                return SttResult(text="12345")

        pipe, disp = build(NumberStt())
        pipe.run_offline([(None, 16000, 1.0)])
        assert disp.events == []
        assert pipe.stats["unknown_lang"] == 1

    def test_skipped_utterance_is_never_silent(self):
        """Câu bị bỏ phải để lại dấu vết — nếu không nó biến mất khỏi màn hình.

        Đo thật trên mic DGM20: số thứ tự nhảy 045 → 047 và 081 → 083, người
        dùng không có cách nào biết máy nghe mà không hiểu, hay máy đã treo.
        """
        class SilentStt(MockSttEngine):
            def transcribe(self, audio, sample_rate):
                return SttResult(text="   ")

        pipe, disp = build(SilentStt())
        pipe.run_offline([(None, 16000, 1.0)])
        assert any("#001" in s for s in disp.statuses)
        # ...và phải hiện lại trong dòng thống kê lúc dừng
        assert any("không ra chữ" in s for s in disp.statuses)

    def test_undecidable_language_is_reported(self):
        class NumberStt(MockSttEngine):
            def transcribe(self, audio, sample_rate):
                return SttResult(text="12345")

        pipe, disp = build(NumberStt())
        pipe.run_offline([(None, 16000, 1.0)])
        assert any("#001" in s and "không xác định" in s for s in disp.statuses)
        assert any("không rõ tiếng nào" in s for s in disp.statuses)

    def test_degenerate_translation_falls_back_to_source(self):
        """Máy dịch chỉ nhả dấu câu → hiện nguyên văn STT, không in rác.

        Ca thật: "Dạ đúng rồi đúng rồi nghĩa là cái đấy là là nó sẽ chạy song
        song với nhau cái chuyện gì" → ",,,,,,,,,,,,." (câu #033).
        """
        class CommaTranslator(FakeTranslator):
            def translate(self, text, src_lang, tgt_lang=None, beam_size=None):
                return TranslationResult(
                    source=text, target=",,,,,,,,,,,,.", src_lang=src_lang,
                    tgt_lang="zh", latency_ms=1.0, degenerate=True)

        stt = MockSttEngine(script=[("vi", "Dạ đúng rồi đúng rồi")])
        pipe, disp = build(stt, translator=CommaTranslator())
        pipe.run_offline([(None, 16000, 1.0)])

        assert len(disp.events) == 1
        assert disp.events[0].target_text == "Dạ đúng rồi đúng rồi"
        assert any("dấu câu" in w for w in disp.events[0].warnings)
        assert pipe.stats["mt_degenerate"] == 1

    def test_stt_exception_does_not_kill_pipeline(self):
        class FlakyStt(MockSttEngine):
            def __init__(self):
                MockSttEngine.__init__(self)
                self.n = 0

            def transcribe(self, audio, sample_rate):
                self.n += 1
                if self.n == 1:
                    raise RuntimeError("model nổ")
                return SttResult(text="你好", lang="zh")

        pipe, disp = build(FlakyStt())
        pipe.run_offline([(None, 16000, 1.0)] * 3)
        assert pipe.stats["errors"] == 1
        assert len(disp.events) == 2       # hai câu sau vẫn chạy bình thường

    def test_force_lang_applies(self):
        cfg = PipelineConfig()
        cfg.force_src_lang = "zh"
        # STT khai "vi" nhưng cấu hình ép "zh" → phải theo cấu hình
        stt = MockSttEngine(script=[("vi", "你好")])
        pipe, disp = build(stt, cfg)
        pipe.run_offline([(None, 16000, 1.0)])
        assert disp.events[0].src_lang == "zh"


class TestDropPolicy:
    def test_drops_oldest_when_full(self):
        """Luồng mic: hàng đợi đầy thì bỏ câu CŨ nhất, giữ câu mới."""
        cfg = PipelineConfig()
        cfg.queue_maxsize = 2
        stt = MockSttEngine()
        pipe, _ = build(stt, cfg)
        # Không khởi động worker → hàng đợi chỉ đầy lên
        for _ in range(5):
            pipe.on_segment_ready(None, 16000, 1.0)

        assert pipe._queue.qsize() == 2
        assert pipe.stats["dropped"] == 3
        assert pipe.stats["segments"] == 5
        # Câu còn lại phải là hai câu MỚI nhất
        seqs = [pipe._queue.get_nowait().seq for _ in range(2)]
        assert seqs == [4, 5]

    def test_can_disable_dropping(self):
        cfg = PipelineConfig()
        cfg.queue_maxsize = 2
        cfg.drop_oldest_when_full = False
        pipe, _ = build(MockSttEngine(), cfg)
        for _ in range(5):
            pipe.on_segment_ready(None, 16000, 1.0)
        seqs = [pipe._queue.get_nowait().seq for _ in range(2)]
        assert seqs == [1, 2]        # giữ câu cũ, bỏ câu mới


class TestDisplayEvent:
    def test_lag_ratio(self):
        e = DisplayEvent(seq=1, source_text="a", target_text="b",
                         audio_duration_s=2.0, total_ms=1000.0)
        assert e.lag_ratio == pytest.approx(0.5)

    def test_lag_ratio_zero_duration(self):
        e = DisplayEvent(seq=1, source_text="a", target_text="b")
        assert e.lag_ratio == 0.0


class TestVadSegmenterContract:
    """Chốt lại hợp đồng với Module 1/2 — nếu đổi chữ ký là test này gãy."""

    def test_modules_importable(self):
        pytest.importorskip("onnxruntime")
        pytest.importorskip("sounddevice")
        from onevoice.pipeline import load_vad_segmenter
        SileroVADDetector, AudioSegmenter = load_vad_segmenter()
        assert hasattr(AudioSegmenter, "process_vad_frame")
        assert hasattr(SileroVADDetector, "start")

    def test_on_segment_ready_signature(self):
        """Segmenter gọi on_segment_ready(audio, sample_rate, duration_sec)."""
        pipe, _ = build(MockSttEngine())
        pipe.on_segment_ready(None, 16000, 2.5)
        item = pipe._queue.get_nowait()
        assert item.sample_rate == 16000
        assert item.duration_s == 2.5
