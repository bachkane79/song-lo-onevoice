"""Nối toàn bộ pipeline: Mic → VAD → Segmenter → STT → Translate → Display.

## Mô hình luồng (quan trọng nhất trong file này)

    luồng audio (sounddevice)  ──►  hàng đợi  ──►  luồng VAD/Segmenter
                                                          │
                                                          │ on_segment_ready()
                                                          ▼
                                                    hàng đợi câu
                                                          │
                                                          ▼
                                              worker STT + Translate + Display

Vì sao phải tách worker riêng: `Segmenter.on_segment_ready()` được gọi **ngay
trong luồng xử lý VAD**. Nếu chạy STT + dịch tại đó (0.5-2s) thì luồng VAD dừng
đọc hàng đợi audio suốt thời gian đó → hàng đợi micro đầy → **mất frame và mất
luôn câu nói tiếp theo**. Đẩy sang worker là bắt buộc, không phải tối ưu.

## Chính sách khi xử lý không kịp

Người nói nhanh hơn máy dịch là chuyện sẽ xảy ra trên Jetson. Khi hàng đợi đầy,
pipeline **bỏ câu CŨ nhất** chứ không bỏ câu mới và cũng không chặn luồng VAD:
phụ đề trễ 5 câu thì vô dụng, thà mất một câu giữa chừng còn hơn mất bám hội
thoại. Số câu bị bỏ được đếm và in ra khi dừng — nếu con số này lớn thì phải hạ
`beam_size` hoặc đổi model nhẹ hơn, chứ không phải tăng `queue_maxsize`.
"""

import os
import queue
import sys
import threading
import time
from dataclasses import dataclass

from onevoice.config import PipelineConfig, repo_root
from onevoice.display import ConsoleDisplay, DisplayEvent
from onevoice.langid import detect_lang
from onevoice.mic import pick_input_device
from onevoice.stt import MockSttEngine
from onevoice.translate import NllbTranslator
from onevoice.vadlog import VadTurnLogger, muted_stdout

__all__ = ["OneVoicePipeline", "Utterance", "load_vad_segmenter"]

# Thư mục module VAD/Segmenter có dấu gạch ngang nên không import kiểu package
# thường được — phải thêm vào sys.path rồi import theo tên file.
VAD_DIR = os.path.join(repo_root(), "VAD-Segmenter")


def load_vad_segmenter():
    """Nạp `SileroVADDetector` và `AudioSegmenter` từ thư mục `VAD-Segmenter/`.

    Hai module đó giữ nguyên không sửa — đây chỉ là lớp nạp.
    """
    if not os.path.isdir(VAD_DIR):
        raise ImportError("Không tìm thấy thư mục %s" % VAD_DIR)
    if VAD_DIR not in sys.path:
        sys.path.insert(0, VAD_DIR)
    from Segmenter_process import AudioSegmenter      # noqa: E402
    from VAD_process import SileroVADDetector         # noqa: E402
    return SileroVADDetector, AudioSegmenter


@dataclass
class Utterance:
    """Một câu nói do Segmenter chốt, đang chờ STT."""

    seq: int
    audio: object            # np.ndarray float32 16kHz
    sample_rate: int
    duration_s: float
    queued_at: float         # time.perf_counter() lúc Segmenter chốt câu


class OneVoicePipeline(object):
    """Điều phối toàn bộ thiết bị.

    Ví dụ::

        from onevoice.pipeline import OneVoicePipeline
        from onevoice.stt import MockSttEngine

        OneVoicePipeline(stt_engine=MockSttEngine()).run()   # Ctrl+C để dừng
    """

    def __init__(self, config=None, stt_engine=None, display=None, translator=None):
        self.config = config or PipelineConfig()
        self.stt = stt_engine or _build_default_stt(self.config)
        self.display = display or ConsoleDisplay()
        self.translator = translator or NllbTranslator(self.config.translate)

        self._queue = queue.Queue(maxsize=self.config.queue_maxsize)
        self._worker = None
        self._running = False
        self._seq = 0
        self.vad_turns = None       # chỉ có ở chế độ micro, gán trong run()

        # Thống kê để đánh giá thiết bị chạy có kịp không
        self.stats = {"segments": 0, "dropped": 0, "translated": 0,
                      "stt_empty": 0, "errors": 0, "unknown_lang": 0,
                      "too_short": 0}

    # ── Khởi động / dừng ────────────────────────────────────────────────────
    def run(self):
        """Chạy pipeline với micro thật. Chặn cho tới khi Ctrl+C.

        Micro được **chọn và mở thử trước khi nạp model** (xem `onevoice/mic.py`):
        thiết bị hỏng thì báo ngay trong 1 giây, thay vì để người dùng chờ nạp
        655 MB model rồi mới đổ lỗi luồng âm thanh.
        """
        SileroVADDetector, AudioSegmenter = load_vad_segmenter()
        cfg = self.config

        device_id, device_name = pick_input_device(cfg.mic_device_id)
        self.display.show_status("🎙️ Micro   : [%d] %s%s"
                                 % (device_id, device_name,
                                    "" if cfg.mic_device_id is not None
                                    else "   (tự nhận)"))

        self._start_worker()
        self.display.show_status("🔤 Máy dịch: %s" % _fmt_stats(self.translator.stats()))
        self.display.show_status("🎤 STT     : %s" % _describe_stt(self.stt))
        self.display.show_status(
            "\n🚀 Sẵn sàng — nói tiếng Việt hoặc tiếng Hoa vào micro (Ctrl+C để dừng)")

        segmenter = AudioSegmenter(
            sample_rate=16000,
            silence_timeout=cfg.silence_timeout,
            min_segment_duration=cfg.min_segment_duration,
            max_segment_duration=cfg.max_segment_duration,
            # ratio 0.0 làm điều kiện `energy < ratio * peak` không bao giờ đúng
            # → tắt sạch tầng Energy Valley mà không phải sửa Segmenter_process.
            valley_ratio=cfg.valley_ratio if cfg.use_energy_valley else 0.0,
            valley_min_frames=cfg.valley_min_frames,
            on_segment_ready=self.on_segment_ready,
            save_wav_debug=cfg.save_wav_debug,
        )
        # Gộp frame VAD thành "lượt nói" để log gọn; hangover khớp silence_timeout
        # nên mốc "hết lượt" trùng đúng lúc Segmenter chốt câu.
        self.vad_turns = VadTurnLogger(self.display,
                                       hangover_s=cfg.silence_timeout,
                                       enabled=cfg.log_vad_turns)
        try:
            # Bịt visualizer 31 dòng/giây của VAD-Segmenter (không sửa file của
            # họ). Lỗi và cảnh báo luồng âm thanh vẫn lọt qua.
            # Bọc cả lúc KHỞI TẠO: `SileroVADDetector.__init__` in 3 dòng banner
            # lặp lại đúng thông tin micro ta vừa in ở trên.
            with muted_stdout(enabled=cfg.quiet_vad_output):
                vad = SileroVADDetector(
                    threshold=cfg.vad_threshold,
                    device_id=device_id,
                    on_vad_frame=self.vad_turns.wrap(segmenter.process_vad_frame),
                )
                vad.start()      # chặn ở đây, tự bắt KeyboardInterrupt bên trong
        finally:
            self.stop()

    def run_offline(self, utterances):
        """Chạy pipeline KHÔNG cần micro — dùng để test/demo và đo trên thiết bị.

        :param utterances: danh sách ``(audio, sample_rate, duration_s)``.
            `audio` có thể là None nếu engine STT không dùng đến (vd MockStt).

        Khác với luồng micro thật: ở đây **chờ** khi hàng đợi đầy thay vì bỏ câu
        cũ. Chế độ này không có ràng buộc thời gian thực nên bỏ câu là mất dữ
        liệu test một cách vô nghĩa.
        """
        self._start_worker()
        for audio, sr, dur in utterances:
            self._seq += 1
            self.stats["segments"] += 1
            self._queue.put(Utterance(
                seq=self._seq, audio=audio, sample_rate=sr, duration_s=dur,
                queued_at=time.perf_counter()))
        self._queue.join()
        self.stop()

    def _start_worker(self):
        self.translator.warmup()
        if hasattr(self.stt, "warmup"):
            self.stt.warmup()
        self._running = True
        self._worker = threading.Thread(target=self._worker_loop,
                                        name="onevoice-stt-mt", daemon=True)
        self._worker.start()

    def stop(self):
        """Dừng worker và in thống kê."""
        if not self._running:
            return
        self._running = False
        if self._worker is not None:
            self._worker.join(timeout=5.0)
        turns = "%d lượt nói · " % self.vad_turns.turns if self.vad_turns else ""
        self.display.show_status(
            "\n📊 %s%d câu · %d đã dịch · %d bỏ do quá tải · %d lỗi"
            % (turns, self.stats["segments"], self.stats["translated"],
               self.stats["dropped"], self.stats["errors"]))
        if hasattr(self.stt, "summary"):
            self.display.show_status("   %s" % self.stt.summary())
        if self.stats["too_short"]:
            self.display.show_status(
                "   %d mẩu <%.1fs bị bỏ (Segmenter cắt vụn). Nếu mất cả câu trả "
                "lời ngắn thật thì hạ --min-utterance."
                % (self.stats["too_short"], self.config.min_utterance_s))
        if self.stats["dropped"]:
            self.display.show_status(
                "   ⚠️  Có câu bị bỏ → thiết bị xử lý không kịp tốc độ nói. "
                "Hạ ONEVOICE_MT_BEAM về 1 hoặc dùng model nhẹ hơn.")
        self.display.close()

    # ── Cầu nối từ Segmenter (chạy trong luồng VAD — phải trả về NGAY) ──────
    def on_segment_ready(self, audio, sample_rate, duration_sec):
        """Callback truyền cho `AudioSegmenter(on_segment_ready=...)`.

        Chỉ đẩy vào hàng đợi rồi trả về ngay lập tức. Tuyệt đối không xử lý gì
        nặng ở đây — xem docstring module.

        Chặn mẩu quá ngắn ngay tại đây: nhánh Silence Timeout của Segmenter
        không kiểm tra `min_segment_duration` (nó chỉ chặn tầng Energy Valley),
        nên phần đuôi ~150ms mà valley để lại có thể ra thành "câu" 0.2s. STT
        nghe ra một thán từ vô nghĩa, máy dịch vẫn tốn 100ms dịch nó, và nó
        chiếm một chỗ trong hàng đợi 4 slot của câu thật.
        """
        if duration_sec < self.config.min_utterance_s:
            self.stats["too_short"] += 1
            return

        self._seq += 1
        self.stats["segments"] += 1
        item = Utterance(seq=self._seq, audio=audio, sample_rate=sample_rate,
                         duration_s=duration_sec, queued_at=time.perf_counter())
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            if self.config.drop_oldest_when_full:
                try:
                    self._queue.get_nowait()        # bỏ câu cũ nhất
                    self._queue.task_done()
                    self.stats["dropped"] += 1
                    self._queue.put_nowait(item)
                except (queue.Empty, queue.Full):
                    self.stats["dropped"] += 1
            else:
                self.stats["dropped"] += 1

    # ── Worker: STT → Translate → Display ───────────────────────────────────
    def _worker_loop(self):
        while self._running:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._process(item)
            except Exception as exc:            # noqa: BLE001
                # Một câu lỗi không được phép giết cả thiết bị.
                self.stats["errors"] += 1
                self.display.show_status("❌ Lỗi khi xử lý câu #%d: %s: %s"
                                         % (item.seq, type(exc).__name__, exc))
            finally:
                self._queue.task_done()

    def _process(self, item):
        # ── STT ──────────────────────────────────────────────────────────
        stt_res = self.stt.transcribe(item.audio, item.sample_rate)
        if stt_res is None or stt_res.is_empty:
            self.stats["stt_empty"] += 1
            return

        # ── Chọn chiều dịch: ép cứng > lang của STT > đoán theo hệ chữ ────
        src = _pick_lang(self.config.force_src_lang, stt_res.lang, stt_res.text)
        if src is None:
            # Câu chỉ có số/dấu câu — dịch cũng vô nghĩa, bỏ còn hơn dịch bừa.
            self.stats["unknown_lang"] += 1
            return

        # ── Dịch ─────────────────────────────────────────────────────────
        mt = self.translator.translate(stt_res.text, src)
        self.stats["translated"] += 1

        warnings = []
        if mt.unk_chars:
            # Model KHÔNG nhìn thấy các ký tự này → bản dịch có thể mất nghĩa.
            warnings.append(
                "Model không đọc được ký tự: %s — thêm luật vào "
                "onevoice/data/zh_unk_map.json" % " ".join(mt.unk_chars))
        if mt.truncated:
            warnings.append("Câu quá dài, đã cắt bớt đầu vào")

        self.display.show(DisplayEvent(
            seq=item.seq,
            source_text=stt_res.text,
            target_text=mt.target,
            src_lang=mt.src_lang,
            tgt_lang=mt.tgt_lang,
            audio_duration_s=item.duration_s,
            stt_ms=stt_res.latency_ms,
            mt_ms=mt.latency_ms,
            total_ms=(time.perf_counter() - item.queued_at) * 1000.0,
            warnings=tuple(warnings),
            stt_detail=stt_res.detail,
        ))


def _pick_lang(forced, stt_lang, text):
    """Thứ tự ưu tiên xác định ngôn ngữ nguồn."""
    from onevoice.config import NLLB_TO_SHORT, SHORT_TO_NLLB

    for candidate in (forced, stt_lang):
        if candidate:
            nllb = SHORT_TO_NLLB.get(str(candidate).strip().lower())
            if nllb:
                return NLLB_TO_SHORT[nllb]
    return detect_lang(text)


def _build_default_stt(config):
    """Tạo engine STT theo `PipelineConfig`, kèm thông báo lỗi hướng dẫn được.

    Thiếu model STT là lỗi hay gặp nhất lúc mới cài, nên không để nó ném ra một
    `FileNotFoundError` trần trụi mà nói rõ phải chạy lệnh gì.
    """
    from onevoice.stt import create_stt_engine

    try:
        return create_stt_engine(
            config.stt_engine,
            mode=config.stt_router_mode,
            force_lang=config.force_src_lang,
            num_threads=config.stt_threads)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "%s\n\nHoặc chạy tạm không cần STT thật:\n"
            "    python run_pipeline.py --stt mock --offline" % exc)
    except ImportError as exc:
        raise ImportError(
            "%s\n→ pip install sherpa-onnx\n"
            "Hoặc chạy tạm: python run_pipeline.py --stt mock --offline" % exc)


def _describe_stt(stt):
    name = getattr(stt, "name", "?")
    if hasattr(stt, "mode"):
        extra = "mode=%s" % stt.mode
        if getattr(stt, "force_lang", None):
            extra = "ép %s (chỉ nạp 1 model)" % stt.force_lang
        return "%s (%s)" % (name, extra)
    return name


def _fmt_stats(d):
    return "%s, %.0f MB, %s, beam %d, %d luật <unk>, %d địa danh" % (
        d["compute_type"], d["disk_mb"], d["device"], d["beam_size"],
        d["zh_unk_rules"], d["glossary_entries"])
