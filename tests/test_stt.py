"""Test tầng STT.

Luật định tuyến ngôn ngữ được test bằng engine giả (không cần model 257 MB) —
đó là phần logic dễ sai nhất. Test cần model thật được đánh dấu `needs_stt_model`
và tự bỏ qua nếu chưa chạy `scripts/download_stt.py`.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from onevoice.stt import (  # noqa: E402
    MockSttEngine,
    ScriptedSttEngine,
    SttResult,
    create_stt_engine,
)

_HAS_SHERPA = True
try:
    import sherpa_onnx  # noqa: F401
except ImportError:
    _HAS_SHERPA = False

if _HAS_SHERPA:
    from onevoice.stt_sherpa import (  # noqa: E402
        SENSEVOICE_DIR,
        ZIPFORMER_VI_DIR,
        BilingualSttEngine,
        _cjk_count,
        _normalize_lang_tag,
        _tidy_vi,
        _word_count,
    )

HAS_STT_MODEL = _HAS_SHERPA and (
    os.path.isfile(os.path.join(SENSEVOICE_DIR, "model.int8.onnx"))
    and os.path.isfile(os.path.join(ZIPFORMER_VI_DIR, "config.json")))


def _needs(cond, reason):
    def deco(obj):
        return pytest.mark.skipif(not cond, reason=reason)(obj)
    return deco


needs_sherpa = _needs(_HAS_SHERPA, "chưa cài sherpa-onnx")


def needs_stt_model(obj):
    return pytest.mark.needs_model(
        pytest.mark.skipif(
            not HAS_STT_MODEL,
            reason="chưa có model STT — chạy scripts/download_stt.py")(obj))


# ── Engine giả lập ──────────────────────────────────────────────────────────
class TestMockEngines:
    def test_cycles_script(self):
        eng = MockSttEngine(script=[("zh", "你好"), ("vi", "Xin chào")])
        assert eng.transcribe(None, 16000).text == "你好"
        assert eng.transcribe(None, 16000).text == "Xin chào"
        assert eng.transcribe(None, 16000).text == "你好"   # quay vòng

    def test_scripted_reads_lang_prefix(self, tmp_path):
        f = tmp_path / "p.txt"
        f.write_text("# ghi chú\nzh: 你好\nvi: Xin chào\nkhông có tiền tố\n",
                     encoding="utf-8")
        eng = ScriptedSttEngine(str(f))
        assert eng.transcribe(None, 16000).lang == "zh"
        assert eng.transcribe(None, 16000).lang == "vi"
        r = eng.transcribe(None, 16000)
        assert r.lang is None and r.text == "không có tiền tố"

    def test_scripted_rejects_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("# chỉ có ghi chú\n", encoding="utf-8")
        with pytest.raises(ValueError):
            ScriptedSttEngine(str(f))

    def test_empty_result(self):
        assert SttResult(text="   ").is_empty
        assert not SttResult(text="a").is_empty


class TestRegistry:
    def test_unknown_engine_raises(self):
        with pytest.raises(ValueError):
            create_stt_engine("khong-ton-tai")

    def test_filters_kwargs_per_engine(self):
        # `mode`/`force_lang` là tham số của engine song ngữ; MockSttEngine không
        # nhận chúng — factory phải tự lọc thay vì nổ TypeError.
        eng = create_stt_engine("mock", mode="dual", force_lang="zh",
                                num_threads=4, fake_latency_ms=0)
        assert isinstance(eng, MockSttEngine)


# ── Hàm phụ trợ của luật định tuyến ─────────────────────────────────────────
@needs_sherpa
class TestHelpers:
    def test_word_count(self):
        assert _word_count("Xin chào các bạn") == 4
        assert _word_count("") == 0
        assert _word_count(None) == 0

    def test_cjk_count(self):
        assert _cjk_count("你好世界") == 4
        assert _cjk_count("GPU 有 8GB") == 1
        assert _cjk_count("Xin chào") == 0

    def test_normalize_lang_tag(self):
        assert _normalize_lang_tag("<|zh|>") == "zh"
        assert _normalize_lang_tag("<|yue|>") == "zh"   # tiếng Quảng vẫn ra chữ Hán
        assert _normalize_lang_tag("<|ja|>") == "ja"
        assert _normalize_lang_tag("") is None
        assert _normalize_lang_tag(None) is None

    def test_tidy_vi(self):
        assert _tidy_vi("XIN CHÀO CÁC BẠN") == "Xin chào các bạn"
        assert _tidy_vi("Đã đúng rồi") == "Đã đúng rồi"
        assert _tidy_vi("") == ""


# ── Luật định tuyến (dùng engine giả, không cần model) ──────────────────────
class FakeEngine:
    """Engine trả về đúng chuỗi cho trước — để test riêng phần quyết định."""

    def __init__(self, text, lang=None):
        self.text = text
        self.lang = lang
        self.calls = 0

    def transcribe(self, audio, sample_rate):
        self.calls += 1
        return SttResult(text=self.text, lang=self.lang)

    def warmup(self):
        pass


@needs_sherpa
class TestRouting:
    """Các con số dưới đây lấy từ đo thật, xem docstring BilingualSttEngine."""

    @staticmethod
    def router(zh_text, vi_text, **kw):
        r = BilingualSttEngine.__new__(BilingualSttEngine)
        r.mode = kw.get("mode", "dual")
        r.force_lang = kw.get("force_lang")
        r.stats = {"zh": 0, "vi": 0, "skipped": 0}
        r.zh = FakeEngine(zh_text)
        r.vi = FakeEngine(vi_text)
        return r

    @staticmethod
    def audio(seconds):
        return np.zeros(int(seconds * 16000), dtype=np.float32)

    def test_chinese_audio_routes_zh(self):
        # Đo thật: audio tiếng Hoa → Zipformer ra rỗng, SenseVoice ra câu đầy đủ
        r = self.router("这并不是告别，这是一个篇章的结束。", "")
        res = r.transcribe(self.audio(10.0), 16000)
        assert res.lang == "zh"
        assert res.text.startswith("这并不是")

    def test_chinese_audio_with_zipformer_noise_still_zh(self):
        # Zipformer "nghe nhầm" 1-2 từ tiếng Việt trên audio tiếng Hoa
        r = self.router("桥下垂直净空15米，该项目于2011年8月完工。", "Chào")
        assert r.transcribe(self.audio(13.9), 16000).lang == "zh"

    def test_vietnamese_audio_routes_vi(self):
        r = self.router("问下话无力搞笑但不能远养光滑腻。",
                        "Văn hóa và bộ lạc cổ xưa đã bắt đầu giữ những con vật này")
        res = r.transcribe(self.audio(11.3), 16000)
        assert res.lang == "vi"
        assert res.text.startswith("Văn hóa")

    def test_skips_when_both_weak(self):
        # Cả hai model gần như không nghe ra gì → tiếng ồn, phải bỏ chứ không bịa
        r = self.router("。", "")
        res = r.transcribe(self.audio(2.0), 16000)
        assert res.is_empty and res.lang is None
        assert res.detail["route"] == "skip"
        assert r.stats["skipped"] == 1

    def test_force_zh_skips_vi_model(self):
        r = self.router("你好。", "Xin chào", force_lang="zh")
        res = r.transcribe(self.audio(2.0), 16000)
        assert res.lang == "zh" and res.text == "你好。"
        assert r.vi.calls == 0, "ép zh mà vẫn chạy model tiếng Việt"

    def test_force_vi_skips_zh_model(self):
        r = self.router("你好。", "Xin chào", force_lang="vi")
        res = r.transcribe(self.audio(2.0), 16000)
        assert res.lang == "vi" and res.text == "Xin chào"
        assert r.zh.calls == 0

    def test_vi_first_mode_skips_sensevoice(self):
        text = " ".join(["từ"] * 12)          # 12 từ / 3s = 4.0 từ/s > FAST_ACCEPT
        r = self.router("你好世界你好世界", text, mode="vi-first")
        res = r.transcribe(self.audio(3.0), 16000)
        assert res.lang == "vi"
        assert r.zh.calls == 0, "vi-first phải bỏ qua SenseVoice khi đã chắc chắn"

    def test_dual_mode_always_runs_both(self):
        text = " ".join(["từ"] * 12)
        r = self.router("你好世界", text, mode="dual")
        r.transcribe(self.audio(3.0), 16000)
        assert r.zh.calls == 1 and r.vi.calls == 1

    def test_rejects_bad_mode(self):
        with pytest.raises(ValueError):
            BilingualSttEngine(mode="linh-tinh", zh_engine=FakeEngine(""),
                               vi_engine=FakeEngine(""))

    def test_detail_is_populated(self):
        r = self.router("你好世界你好", "")
        d = r.transcribe(self.audio(2.0), 16000).detail
        assert d["route"] == "zh"
        assert d["vi_rate"] == 0.0 and d["zh_rate"] > 0

    def test_summary_counts(self):
        r = self.router("这并不是告别，这是一个篇章。", "")
        r.transcribe(self.audio(10.0), 16000)
        assert "1 câu zh" in r.summary()


# ── Cần model thật ──────────────────────────────────────────────────────────
@needs_stt_model
class TestRealModels:
    @pytest.fixture(scope="class")
    def router(self):
        r = BilingualSttEngine()
        r.warmup()
        return r

    def test_silence_gives_empty(self, router):
        res = router.transcribe(np.zeros(2 * 16000, dtype=np.float32), 16000)
        assert res.is_empty

    def test_chinese_sample_wav(self, router):
        wav = os.path.join(SENSEVOICE_DIR, "test_wavs", "zh.wav")
        if not os.path.isfile(wav):
            pytest.skip("thiếu file mẫu zh.wav")
        import wave
        with wave.open(wav, "rb") as w:
            audio = (np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
                     .astype(np.float32) / 32768.0)
            sr = w.getframerate()
        res = router.transcribe(audio, sr)
        assert res.lang == "zh"
        assert _cjk_count(res.text) > 3

    def test_engines_load_from_registry(self):
        eng = create_stt_engine("sherpa", mode="dual", num_threads=1)
        assert eng.name == "bilingual"
        assert eng.zh is not None and eng.vi is not None
