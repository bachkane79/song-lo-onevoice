"""Module 3 — STT bằng sherpa-onnx: SenseVoice (tiếng Hoa) + Zipformer (tiếng Việt).

Hai model, mỗi model một thứ tiếng:

| | SenseVoice | Zipformer-30M-RNNT-6000h |
|---|---|---|
| Tiếng | zh / en / ja / ko / yue | **vi** |
| Nguồn | k2-fsa/sherpa-onnx releases | HF `hynt/Zipformer-30M-RNNT-6000h` |
| Kích thước int8 | 228 MB | **29 MB** |
| Kiến trúc | encoder-only + CTC-ish | RNN-Transducer |
| Có nhận diện ngôn ngữ | ✅ (`result.lang`) | ❌ |

Không có model nào làm được cả hai tiếng, nên phải **định tuyến**: nghe một câu
rồi quyết định đưa vào model nào. Đó là việc của `BilingualSttEngine` — xem
docstring của class đó để biết chiến lược và vì sao chọn nó.

⚖️ **Giấy phép**: Zipformer tiếng Việt phát hành theo **CC BY-NC-ND 4.0**
(phi thương mại, không phái sinh). Dùng cho nghiên cứu/đồ án thì được; nếu thiết
bị này định bán thì phải xin phép tác giả hoặc đổi model tiếng Việt khác.
"""

import os
import time

from onevoice.stt import SttEngine, SttResult

__all__ = ["SenseVoiceSttEngine", "ZipformerViSttEngine", "BilingualSttEngine",
           "SENSEVOICE_DIR", "ZIPFORMER_VI_DIR"]

_STT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "stt")
SENSEVOICE_DIR = os.environ.get("ONEVOICE_STT_ZH", os.path.join(_STT_ROOT, "sensevoice-zh"))
ZIPFORMER_VI_DIR = os.environ.get("ONEVOICE_STT_VI",
                                  os.path.join(_STT_ROOT, "zipformer-vi-30m"))

_MISSING = ("Không tìm thấy model STT tại %s\n"
            "→ Chạy: python scripts/download_stt.py")


def _require(path, what):
    if not os.path.exists(path):
        raise FileNotFoundError(_MISSING % ("%s (%s)" % (path, what)))
    return path


class SenseVoiceSttEngine(SttEngine):
    """SenseVoice — nhận dạng tiếng Hoa (và en/ja/ko/yue).

    `language=""` là chế độ tự nhận diện; model trả mã ngôn ngữ ở `result.lang`
    dưới dạng `<|zh|>`. `BilingualSttEngine` dùng chính tín hiệu này để định
    tuyến nên **đừng ép `language` nếu đang chạy song ngữ**.
    """

    name = "sensevoice"

    def __init__(self, model_dir=None, num_threads=2, language="", use_itn=True):
        import sherpa_onnx

        model_dir = model_dir or SENSEVOICE_DIR
        self.model_dir = model_dir
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=_require(os.path.join(model_dir, "model.int8.onnx"), "SenseVoice"),
            tokens=_require(os.path.join(model_dir, "tokens.txt"), "tokens SenseVoice"),
            num_threads=num_threads,
            language=language,     # "" = tự nhận diện, "zh" = ép tiếng Hoa
            use_itn=use_itn,       # thêm dấu câu + chuẩn hóa số → tốt cho máy dịch
        )

    def transcribe(self, audio, sample_rate):
        t0 = time.perf_counter()
        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio)
        self.recognizer.decode_stream(stream)
        res = stream.result
        return SttResult(
            text=(res.text or "").strip(),
            lang=_normalize_lang_tag(getattr(res, "lang", "")),
            latency_ms=(time.perf_counter() - t0) * 1000.0)

    def warmup(self):
        import numpy as np
        self.transcribe(np.zeros(16000, dtype=np.float32), 16000)


class ZipformerViSttEngine(SttEngine):
    """Zipformer-30M RNN-Transducer — nhận dạng tiếng Việt.

    Lưu ý về tên file: repo gốc đặt bảng token vào `config.json` (không phải
    `tokens.txt` như thường lệ). Đó là **file token đúng**, đã kiểm tra nội dung
    là định dạng `<token> <id>` của k2 — không phải file cấu hình JSON.
    """

    name = "zipformer-vi"

    def __init__(self, model_dir=None, num_threads=2, decoding_method="greedy_search"):
        import sherpa_onnx

        model_dir = model_dir or ZIPFORMER_VI_DIR
        self.model_dir = model_dir
        p = lambda f: os.path.join(model_dir, f)   # noqa: E731
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=_require(p("encoder-epoch-20-avg-10.int8.onnx"), "encoder vi"),
            decoder=_require(p("decoder-epoch-20-avg-10.int8.onnx"), "decoder vi"),
            joiner=_require(p("joiner-epoch-20-avg-10.int8.onnx"), "joiner vi"),
            tokens=_require(p("config.json"), "tokens vi"),
            bpe_vocab=_require(p("bpe.model"), "bpe vi"),
            num_threads=num_threads,
            decoding_method=decoding_method,
        )

    def transcribe(self, audio, sample_rate):
        t0 = time.perf_counter()
        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio)
        self.recognizer.decode_stream(stream)
        text = (stream.result.text or "").strip()
        return SttResult(text=_tidy_vi(text), lang="vi",
                         latency_ms=(time.perf_counter() - t0) * 1000.0)

    def warmup(self):
        import numpy as np
        self.transcribe(np.zeros(16000, dtype=np.float32), 16000)


class BilingualSttEngine(SttEngine):
    """Định tuyến câu nói vào đúng model: SenseVoice (zh) hay Zipformer (vi).

    ## Vì sao cần định tuyến và vì sao KHÔNG dùng cách hiển nhiên

    Không model nào làm được cả hai tiếng, nên phải chọn. Ba cách đã thử, đo trên
    16 câu FLEURS thật (8 zh + 8 vi) và 233 đoạn cắt ngắn 1.3-6.0s:

    **❌ Cách 1 — dùng nhận diện ngôn ngữ của SenseVoice.** SenseVoice có sẵn LID
    và trả `result.lang`. Nhưng nó chỉ biết zh/en/ja/ko/yue, **không có tiếng
    Việt**, nên nó báo `<|zh|>` cho **8/8 câu tiếng Việt**. Hoàn toàn không dùng được.

    **✅ Cách 2 — dùng chính sản lượng của Zipformer làm tín hiệu.** Zipformer là
    RNN-Transducer chỉ huấn luyện tiếng Việt. Gặp audio tiếng Hoa nó không tìm
    được gì khớp nên **gần như không xuất token nào**:

        audio tiếng Hoa 8-15s  → Zipformer ra: "", "", "Chào", "", "Tại sao"
        audio tiếng Việt 8-15s → Zipformer ra: câu đầy đủ, gần đúng hoàn toàn

    Đo bằng *số từ trên mỗi giây audio*: tiếng Hoa 0.00-0.13, tiếng Việt
    1.94-2.74 — cách nhau khoảng 17 lần.

    **Luật cuối cùng** (kết hợp thêm mật độ Hán tự của SenseVoice để đỡ nhầm ở
    câu rất ngắn):

        vi_rate = số_từ(Zipformer) / độ_dài_audio
        zh_rate = số_Hán_tự(SenseVoice) / độ_dài_audio

        nếu max(vi_rate, zh_rate) < FLOOR → không đủ tiếng nói, bỏ câu
        ngược lại: vi nếu vi_rate > ALPHA * zh_rate + MARGIN, còn lại là zh

    Độ chính xác đo được (ALPHA=0.3, MARGIN=0.8, FLOOR=0.8):

    | bộ | đúng | sai | bỏ |
    |---|---|---|---|
    | câu đầy đủ 7-22s | **16/16 (100%)** | 0 | 0 |
    | đoạn ngắn 1.3-6.0s | 205 | 5 | 23 |
    | **tổng** | **221 (97.8%)** | 5 | 23 |

    **0 lỗi ở chiều tiếng Việt.** 5 lỗi đều là đoạn tiếng Hoa rất ngắn (1.3-2.0s)
    mà Zipformer "nghe nhầm" ra vài từ tiếng Việt. Nếu biết trước ai đang nói thì
    đặt `force_lang` — lúc đó độ chính xác định tuyến là tuyệt đối và còn nhanh
    hơn vì chỉ chạy một model.

    ## Chi phí

    Chế độ `dual` chạy **cả hai** model mỗi câu. Trên x86 tổng RTF ≈ 0.07
    (Zipformer 0.02 + SenseVoice 0.05) nên rẻ. Trên Jetson chậm hơn nhiều lần —
    đo bằng `scripts/bench_stt.py`; nếu không kịp thì dùng `mode="vi-first"`
    (bỏ qua SenseVoice khi Zipformer đã chắc chắn là tiếng Việt) hoặc `force_lang`.
    """

    name = "bilingual"

    # Hằng số đã hiệu chỉnh trên 249 mẫu — xem docstring. Đừng đổi bằng cảm giác,
    # sửa xong phải chạy lại scripts/bench_stt.py --routing.
    ALPHA = 0.3       # trọng số của bằng chứng tiếng Hoa
    MARGIN = 0.8      # tiếng Việt phải vượt bằng chứng tiếng Hoa ít nhất chừng này
    FLOOR = 0.8       # cả hai dưới mức này = không đủ tiếng nói để quyết
    FAST_ACCEPT = 3.0  # mode "vi-first": vi_rate cao hơn mức này thì khỏi chạy SenseVoice

    def __init__(self, zh_engine=None, vi_engine=None, mode="dual",
                 force_lang=None, num_threads=2):
        """
        :param mode: "dual" (chính xác nhất) | "vi-first" (nhanh hơn, xem FAST_ACCEPT)
        :param force_lang: "zh" | "vi" — bỏ hẳn định tuyến, chỉ nạp một model
        """
        if mode not in ("dual", "vi-first"):
            raise ValueError("mode phải là 'dual' hoặc 'vi-first', không phải %r" % mode)
        self.mode = mode
        self.force_lang = force_lang
        self.stats = {"zh": 0, "vi": 0, "skipped": 0}

        # Ép ngôn ngữ thì chỉ nạp một model — tiết kiệm 228 MB RAM và một lần chạy
        self.zh = None
        self.vi = None
        if force_lang != "vi":
            self.zh = zh_engine or SenseVoiceSttEngine(
                num_threads=num_threads,
                language="zh" if force_lang == "zh" else "")
        if force_lang != "zh":
            self.vi = vi_engine or ZipformerViSttEngine(num_threads=num_threads)

    def transcribe(self, audio, sample_rate):
        t0 = time.perf_counter()
        duration = max(len(audio) / float(sample_rate), 1e-3)

        if self.force_lang == "zh":
            res = self.zh.transcribe(audio, sample_rate)
            res.lang = "zh"
            self.stats["zh"] += 1
            return res
        if self.force_lang == "vi":
            res = self.vi.transcribe(audio, sample_rate)
            # Router tự đặt lang thay vì tin engine con đặt đúng — nếu sau này
            # ai cắm engine tiếng Việt khác vào thì pipeline vẫn định chiều đúng.
            res.lang = "vi"
            self.stats["vi"] += 1
            return res

        # ── Bước 1: Zipformer (rẻ hơn 2.5×) ─────────────────────────────────
        r_vi = self.vi.transcribe(audio, sample_rate)
        vi_rate = _word_count(r_vi.text) / duration

        # ── Bước 2: SenseVoice, trừ khi chắc chắn đã là tiếng Việt ──────────
        if self.mode == "vi-first" and vi_rate >= self.FAST_ACCEPT:
            self.stats["vi"] += 1
            r_vi.lang = "vi"
            r_vi.latency_ms = (time.perf_counter() - t0) * 1000.0
            r_vi.detail = {"route": "vi", "vi_rate": round(vi_rate, 2),
                           "zh_rate": None, "fast_accept": True}
            return r_vi

        r_zh = self.zh.transcribe(audio, sample_rate)
        zh_rate = _cjk_count(r_zh.text) / duration

        # ── Bước 3: quyết định ──────────────────────────────────────────────
        detail = {"vi_rate": round(vi_rate, 2), "zh_rate": round(zh_rate, 2),
                  "vi_text": r_vi.text, "zh_text": r_zh.text}
        total_ms = (time.perf_counter() - t0) * 1000.0

        if max(vi_rate, zh_rate) < self.FLOOR:
            # Không model nào nghe ra gì đáng kể → nhiều khả năng là tiếng ồn.
            # Trả text rỗng để pipeline bỏ qua, thay vì hiển thị một câu bịa.
            self.stats["skipped"] += 1
            detail["route"] = "skip"
            return SttResult(text="", lang=None, latency_ms=total_ms, detail=detail)

        if vi_rate > self.ALPHA * zh_rate + self.MARGIN:
            self.stats["vi"] += 1
            detail["route"] = "vi"
            return SttResult(text=r_vi.text, lang="vi",
                             confidence=_route_margin(vi_rate, zh_rate, self.ALPHA,
                                                      self.MARGIN),
                             latency_ms=total_ms, detail=detail)

        self.stats["zh"] += 1
        detail["route"] = "zh"
        return SttResult(text=r_zh.text, lang="zh",
                         confidence=_route_margin(zh_rate * self.ALPHA + self.MARGIN,
                                                  vi_rate, 1.0, 0.0),
                         latency_ms=total_ms, detail=detail)

    def warmup(self):
        for eng in (self.vi, self.zh):
            if eng is not None:
                eng.warmup()

    def summary(self):
        return "STT định tuyến: %d câu zh, %d câu vi, %d bỏ (không đủ tiếng nói)" % (
            self.stats["zh"], self.stats["vi"], self.stats["skipped"])


def _word_count(text):
    return len([w for w in (text or "").split() if w])


def _cjk_count(text):
    return sum(1 for c in (text or "") if 0x3400 <= ord(c) <= 0x9FFF)


def _route_margin(winner, loser, alpha, margin):
    """Khoảng cách tương đối giữa hai bên, quy về [0, 1] để log cho dễ đọc."""
    lo = alpha * loser + margin
    if winner <= 0:
        return 0.0
    return round(min(1.0, max(0.0, (winner - lo) / winner)), 2)


def _normalize_lang_tag(tag):
    """`<|zh|>` → `"zh"`. Trả None nếu không đọc được."""
    if not tag:
        return None
    tag = tag.strip().strip("<>|").strip()
    if not tag:
        return None
    tag = tag.lower()
    return "zh" if tag in ("zh", "yue", "cmn") else tag


def _tidy_vi(text):
    """Zipformer xuất chữ IN HOA không dấu câu — chuẩn hóa nhẹ cho dễ đọc/dễ dịch.

    Chỉ đổi về dạng viết thường rồi hoa chữ đầu. **Không** cố thêm dấu câu: đoán
    sai dấu câu sẽ đổi nghĩa và làm máy dịch sai theo.
    """
    if not text:
        return text
    if text.isupper():
        text = text.lower()
        text = text[0].upper() + text[1:]
    return text
