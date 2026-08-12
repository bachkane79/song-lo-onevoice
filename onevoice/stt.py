"""Chỗ cắm cho Module 3 — STT (Speech-to-Text).

**Đây là file mà thành viên làm STT cần sửa.** Toàn bộ phần còn lại của pipeline
(VAD → Segmenter → **STT** → Translate → Display) đã chạy được; chỉ cần cung cấp
một class có đúng một phương thức `transcribe()`.

## Hợp đồng interface

    class MyStt(SttEngine):
        def transcribe(self, audio, sample_rate):
            # audio       : np.ndarray float32, shape (N,), giá trị trong [-1, 1]
            # sample_rate : luôn = 16000
            return SttResult(text="câu vừa nghe được", lang="zh")  # hoặc "vi"/None

`lang` nên trả về nếu model nhận diện được ngôn ngữ (Whisper có sẵn), vì nó
quyết định chiều dịch. Trả `None` cũng được — pipeline sẽ tự đoán theo hệ chữ
viết của `text` (xem `onevoice.langid`), độ chính xác gần như tuyệt đối vì tiếng
Việt dùng Latin còn tiếng Hoa dùng Hán tự.

## Ràng buộc quan trọng

- `transcribe()` chạy trong **worker thread**, không phải luồng audio. Chạy lâu
  cũng không làm rớt frame micro, nhưng làm phụ đề trễ hơn so với người nói.
- Ngân sách thời gian: Segmenter cắt câu tối đa 8s. Để phụ đề bám kịp hội thoại,
  tổng STT + dịch nên < ~1.5s/câu. Máy dịch đã chiếm ~0.2-0.4s trên CPU x86,
  nhiều hơn trên Jetson — đo bằng `scripts/bench_translate.py`.
- **Không** được ném exception ra ngoài: pipeline có bắt, nhưng câu đó sẽ mất.
"""

import time
from dataclasses import dataclass

__all__ = ["SttResult", "SttEngine", "MockSttEngine", "ScriptedSttEngine",
           "create_stt_engine"]


@dataclass
class SttResult:
    """Kết quả nhận dạng một câu."""

    text: str
    lang: str = None           # "zh" | "vi" | None (None = để pipeline tự đoán)
    confidence: float = None   # None nếu model không cung cấp
    latency_ms: float = 0.0
    # Thông tin phụ để log/gỡ lỗi (vd: engine nào chạy, điểm định tuyến).
    # Pipeline không dựa vào trường này để quyết định gì cả.
    detail: dict = None

    @property
    def is_empty(self):
        return not (self.text or "").strip()


class SttEngine(object):
    """Interface tối thiểu mà mọi engine STT phải thỏa mãn."""

    name = "base"

    def transcribe(self, audio, sample_rate):
        """Nhận dạng một câu. PHẢI trả về `SttResult`."""
        raise NotImplementedError

    def warmup(self):
        """Chạy thử một lần lúc khởi động để tránh câu đầu bị chậm."""

    def close(self):
        """Giải phóng tài nguyên."""


class MockSttEngine(SttEngine):
    """Engine giả lập — cho phép chạy và demo TOÀN BỘ pipeline khi chưa có STT.

    Nó bỏ qua audio và trả lần lượt các câu mẫu song ngữ, nên vẫn kiểm tra được:
    VAD ngắt câu đúng chưa, chiều dịch có tự đảo không, màn hình hiển thị ra sao,
    độ trễ dịch bao nhiêu. Thay bằng engine thật là xong pipeline.
    """

    name = "mock"

    DEFAULT_SCRIPT = [
        ("zh", "你好，请问厕所在哪里？"),
        ("vi", "Cho tôi hai vé đi Hà Nội vào sáng mai nhé."),
        ("zh", "这个多少钱？可以便宜一点吗？"),
        ("vi", "Món này cay quá, cho tôi xin thêm nước."),
        ("zh", "我对虾过敏，请不要放虾。"),
        ("vi", "Tàu đi Đà Nẵng khởi hành lúc mấy giờ?"),
        ("zh", "我们从胡志明市飞往北京，晚上七点起飞。"),
        ("vi", "Cảm ơn anh rất nhiều, hẹn gặp lại."),
    ]

    def __init__(self, script=None, fake_latency_ms=0.0):
        """
        :param script: danh sách ``(lang, text)``; None = dùng bộ mẫu sẵn
        :param fake_latency_ms: giả lập độ trễ của STT thật để đo hàng đợi
        """
        self.script = list(script or self.DEFAULT_SCRIPT)
        self.fake_latency_ms = fake_latency_ms
        self._i = 0

    def transcribe(self, audio, sample_rate):
        t0 = time.perf_counter()
        if self.fake_latency_ms > 0:
            time.sleep(self.fake_latency_ms / 1000.0)
        lang, text = self.script[self._i % len(self.script)]
        self._i += 1
        return SttResult(text=text, lang=lang, confidence=1.0,
                         latency_ms=(time.perf_counter() - t0) * 1000.0)


class ScriptedSttEngine(MockSttEngine):
    """Đọc câu từ file text (mỗi dòng một câu) thay vì bộ mẫu cứng.

    Dùng khi muốn thử pipeline với đúng bộ câu của mình::

        python run_pipeline.py --stt scripted --stt-script my_phrases.txt
    """

    name = "scripted"

    def __init__(self, path, fake_latency_ms=0.0):
        lines = []
        with open(path, encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw or raw.startswith("#"):
                    continue
                # Cho phép "zh: 你好" để ép ngôn ngữ; không có thì để None
                if len(raw) > 3 and raw[:3].lower() in ("zh:", "vi:"):
                    lines.append((raw[:2].lower(), raw[3:].strip()))
                else:
                    lines.append((None, raw))
        if not lines:
            raise ValueError("File %s không có câu nào." % path)
        MockSttEngine.__init__(self, script=lines,
                               fake_latency_ms=fake_latency_ms)


# ── Mẫu cho engine thật ─────────────────────────────────────────────────────
# Bỏ comment và hoàn thiện khi tích hợp. GIỮ NGUYÊN chữ ký transcribe().
#
# class FasterWhisperSttEngine(SttEngine):
#     """Faster-Whisper (CTranslate2) — cùng runtime với máy dịch, tiện cho Jetson."""
#
#     name = "faster-whisper"
#
#     def __init__(self, model_dir, compute_type="int8", num_workers=1):
#         from faster_whisper import WhisperModel
#         self.model = WhisperModel(model_dir, device="cpu",
#                                   compute_type=compute_type, num_workers=num_workers)
#
#     def transcribe(self, audio, sample_rate):
#         t0 = time.perf_counter()
#         segments, info = self.model.transcribe(
#             audio,                    # float32 16kHz — truyền thẳng, không cần ghi file
#             language=None,            # None = tự nhận diện; đặt "zh"/"vi" nếu biết trước
#             beam_size=1, vad_filter=False)   # VAD đã làm ở Module 1, đừng làm lại
#         text = "".join(s.text for s in segments).strip()
#         return SttResult(text=text,
#                          lang=info.language if info.language in ("zh", "vi") else None,
#                          confidence=getattr(info, "language_probability", None),
#                          latency_ms=(time.perf_counter() - t0) * 1000.0)


def _make_sherpa(**kwargs):
    """Nạp muộn để `mock`/`scripted` không bắt buộc phải cài sherpa-onnx."""
    from onevoice.stt_sherpa import BilingualSttEngine
    return BilingualSttEngine(**kwargs)


def _make_sensevoice(**kwargs):
    from onevoice.stt_sherpa import SenseVoiceSttEngine
    return SenseVoiceSttEngine(**kwargs)


def _make_zipformer_vi(**kwargs):
    from onevoice.stt_sherpa import ZipformerViSttEngine
    return ZipformerViSttEngine(**kwargs)


_REGISTRY = {
    "sherpa": _make_sherpa,           # song ngữ, tự định tuyến — mặc định
    "sensevoice": _make_sensevoice,   # chỉ tiếng Hoa
    "zipformer-vi": _make_zipformer_vi,  # chỉ tiếng Việt
    "mock": MockSttEngine,
    "scripted": ScriptedSttEngine,
}

# Tham số nào hợp lệ với engine nào — để `create_stt_engine` lọc bớt, người gọi
# không phải nhớ engine nào nhận gì.
_ACCEPTS = {
    "sherpa": ("mode", "force_lang", "num_threads", "zh_engine", "vi_engine"),
    "sensevoice": ("model_dir", "num_threads", "language", "use_itn"),
    "zipformer-vi": ("model_dir", "num_threads", "decoding_method"),
    "mock": ("script", "fake_latency_ms"),
    "scripted": ("path", "fake_latency_ms"),
}


def create_stt_engine(name, **kwargs):
    """Tạo engine STT theo tên, tự bỏ tham số không hợp lệ với engine đó."""
    key = (name or "sherpa").lower()
    if key not in _REGISTRY:
        raise ValueError("Chưa có engine STT tên %r. Đang có: %s"
                         % (name, ", ".join(sorted(_REGISTRY))))
    allowed = _ACCEPTS.get(key, ())
    clean = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    return _REGISTRY[key](**clean)
