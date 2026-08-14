"""Cấu hình tập trung cho pipeline OneVoice.

Mọi đường dẫn và tham số tinh chỉnh nằm ở đây. Có thể override bằng biến môi
trường để cùng một code chạy trên máy dev (Windows) và trên Jetson mà không sửa
file:

    ONEVOICE_MT_MODEL      — thư mục model CTranslate2 int8
    ONEVOICE_MT_TOKENIZER  — file tokenizer.json (NLLB đã cắt vocab)
    ONEVOICE_MT_THREADS    — số thread CPU cho máy dịch
    ONEVOICE_MT_BEAM       — beam size (1 = nhanh nhất, gần như không mất chất lượng)
    ONEVOICE_MT_DEVICE     — "cpu" hoặc "cuda"
    ONEVOICE_MT_COMPUTE    — "int8", "int8_float32", "float32"...
"""

import os
from dataclasses import dataclass, field

# ── Mã ngôn ngữ NLLB (FLORES-200) ────────────────────────────────────────────
LANG_ZH = "zho_Hans"   # Tiếng Hoa giản thể
LANG_VI = "vie_Latn"   # Tiếng Việt

# Mã ngắn dùng trong toàn pipeline (khớp với mã của Whisper/Sherpa: "zh", "vi")
SHORT_TO_NLLB = {
    "zh": LANG_ZH,
    "zh-cn": LANG_ZH,
    "zh-hans": LANG_ZH,
    "cmn": LANG_ZH,
    "yue": LANG_ZH,        # Tiếng Quảng: STT ra chữ Hán → vẫn dịch bằng zho_Hans
    "vi": LANG_VI,
    "vie": LANG_VI,
}
NLLB_TO_SHORT = {LANG_ZH: "zh", LANG_VI: "vi"}

# Chiều dịch mặc định của thiết bị: nghe tiếng nào thì dịch sang tiếng còn lại
OPPOSITE_SHORT = {"zh": "vi", "vi": "zh"}

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MT_MODEL_DIR = os.path.join(_REPO_ROOT, "models", "mt", "nllb-zhvi-int8")
DEFAULT_MT_TOKENIZER = os.path.join(_REPO_ROOT, "models", "mt", "tokenizer.json")
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEFAULT_GLOSSARY = os.path.join(_DATA_DIR, "glossary_vi_zh.json")
DEFAULT_ZH_UNK_MAP = os.path.join(_DATA_DIR, "zh_unk_map.json")
# Manifest của GitHub Release chứa model dịch (tên file + kích thước + sha256)
DEFAULT_MT_RELEASE = os.path.join(_DATA_DIR, "mt_release.json")


def _env_int(name, default):
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def repo_root():
    """Thư mục gốc của repo (dùng để nạp VAD-Segmenter, models/...)."""
    return _REPO_ROOT


@dataclass
class TranslateConfig:
    """Tham số của máy dịch NLLB-600M vocab-trim + CTranslate2.

    Mặc định `beam_size=4`. Trên bộ test 30 câu (toàn câu dài) beam1 chỉ kém
    beam4 0.2 chrF++, nhưng trên **câu hội thoại ngắn** — đúng loại câu thiết bị
    này gặp — beam1 sai rõ rệt:

        "Nhà vệ sinh ở đâu ạ?"  beam1 → 所在哪里?   ✗ (mất chữ)
                                beam4 → 卫生间在哪里? ✓
        "太贵了。"               beam1 → "- Đủ đắt tiền."  ✗
                                beam4 → "Rất đắt tiền."   ✓

    Với câu ngắn thì beam4 chỉ chậm hơn beam1 ~11% (184ms vs 166ms trên i9), nên
    đánh đổi này rẻ. Hạ về 1 nếu Jetson quá chậm — dùng
    `scripts/bench_translate.py` để đo trên máy thật rồi mới quyết.
    """

    model_dir: str = field(default_factory=lambda: os.environ.get(
        "ONEVOICE_MT_MODEL", DEFAULT_MT_MODEL_DIR))
    tokenizer_path: str = field(default_factory=lambda: os.environ.get(
        "ONEVOICE_MT_TOKENIZER", DEFAULT_MT_TOKENIZER))
    device: str = field(default_factory=lambda: os.environ.get(
        "ONEVOICE_MT_DEVICE", "cpu"))
    compute_type: str = field(default_factory=lambda: os.environ.get(
        "ONEVOICE_MT_COMPUTE", "int8"))
    intra_threads: int = field(default_factory=lambda: _env_int(
        "ONEVOICE_MT_THREADS", 4))
    inter_threads: int = 1
    beam_size: int = field(default_factory=lambda: _env_int(
        "ONEVOICE_MT_BEAM", 4))

    # Giới hạn an toàn — Segmenter cắt câu tối đa 8s nên input luôn ngắn,
    # các số này chỉ để chặn trường hợp STT sinh output lặp vô hạn.
    max_input_tokens: int = 256
    max_decoding_length: int = 256
    # Bật = 3, ĐÃ ĐO, đừng hạ về 0 mà không đo lại.
    #
    # Trước đây để 0 vì lo hại tiếng Hoa. Đo trên 126 câu STT thật (mic DGM20,
    # phiên 99 câu) + 14 câu bench của scripts/bench_translate.py:
    #
    #                          nrng=0    nrng=3
    #   câu có cụm lặp >=4 ký tự   7        1
    #   chuỗi chỉ có dấu câu       2        1
    #   câu bench đổi output       —        0/14
    #
    # Nghĩa là dẹp 6/7 ca lặp mà không động tới bộ câu đã benchmark, và không
    # phát sinh ca lặp mới. Ca còn lại là dương tính giả của bộ dò ("Nếu bạn
    # nhìn thấy nó, bạn sẽ không thấy nó" — tiếng Việt hợp lệ).
    # Đo lại: python scripts/bench_translate.py
    no_repeat_ngram_size: int = 3

    # Hậu xử lý địa danh / tên riêng Hán-Việt cho chiều vi→zh.
    # NLLB phiên âm "Đà Nẵng" → 达南 thay vì 岘港; glossary sửa lại.
    use_glossary: bool = True
    glossary_path: str = field(default_factory=lambda: os.environ.get(
        "ONEVOICE_MT_GLOSSARY", DEFAULT_GLOSSARY))

    # Chuẩn hóa đầu vào tiếng Hoa để tránh <unk> (chiều zh→vi).
    # Vocab NLLB thiếu hẳn 11 ký tự thường dùng: 厕咸啤汤烫虾蟹辣酱醋鸡
    # → xem onevoice/zh_norm.py để biết chi tiết.
    use_zh_unk_map: bool = True
    zh_unk_map_path: str = field(default_factory=lambda: os.environ.get(
        "ONEVOICE_MT_ZH_UNK_MAP", DEFAULT_ZH_UNK_MAP))


@dataclass
class PipelineConfig:
    """Tham số nối VAD → Segmenter → STT → Translate → Display."""

    # ── VAD ──────────────────────────────────────────────────────────────
    vad_threshold: float = 0.35
    # None = tự dò micro dùng được (onevoice/mic.py). Đặt số để khóa cứng.
    mic_device_id: int = None
    # Nuốt visualizer 31 dòng/giây của VAD-Segmenter. Tắt (--verbose) khi cần
    # xem lại log gốc để gỡ lỗi phần cứng âm thanh.
    quiet_vad_output: bool = True
    # In 2 mốc mỗi lượt nói (bắt đầu / hết lượt) thay cho visualizer.
    log_vad_turns: bool = True

    # ── Segmenter ────────────────────────────────────────────────────────
    # Số dưới đây chọn theo đo thật trên mic Realtek (scripts/diag_vad.py):
    #
    #                       phòng yên      đang nói
    #   VAD prob (median)     0.031          0.891
    #   năng lượng (median)   0.101          0.124
    #   chuỗi im dài nhất     5.15s          0.10s
    #
    # Đọc bảng này ra hai kết luận, cả hai đều ngược với mặc định gốc:
    #
    # [1] Khoảng lặng dài nhất BÊN TRONG một câu liền mạch chỉ 0.10s → mọi giá
    #     trị silence_timeout >= 0.4s đều không thể cắt giữa câu. Tầng 1 chưa
    #     bao giờ là thủ phạm; nâng nó lên chỉ thêm độ trễ.
    #
    # [2] Energy Valley KHÔNG dùng được trên phần cứng này. AGC trong
    #     VAD_process kéo mọi thứ về biên độ mục tiêu 0.3, nên năng lượng lúc im
    #     (0.101) gần bằng lúc nói (0.124) — đúng cái dải động mà valley cần đã
    #     bị san phẳng. Với peak≈0.40: ratio 0.25 → ngưỡng 0.10, nằm NGAY DƯỚI
    #     median lúc nói → bắn liên tục giữa câu (đây là lỗi "cắt hết giữa câu").
    #     Hạ xuống ratio 0.12 → ngưỡng 0.056, dưới cả sàn ồn → không bao giờ bắn,
    #     mọi câu rơi xuống Max Duration 8s. Không có giá trị nào ở giữa chạy
    #     được, vì khoảng nghỉ thật và tiếng ồn nền cùng nằm ở mức 0.10.
    #     → Tắt hẳn tầng 2, dựa vào VAD silence (0.031 vs 0.891, tách rất rõ).

    # ⚠️ Cộng THẲNG vào độ trễ của mọi câu. 0.7s = 0.10s đo được + biên cho
    # người thật ngập ngừng giữa câu (máy đọc thì đều, người thì không).
    silence_timeout: float = 0.7
    max_segment_duration: float = 8.0

    # Tầng 2 tắt mặc định — xem [2] ở trên. Bật lại bằng --energy-valley nếu
    # chạy trên mic khác (hoặc sau khi AGC được chỉnh) rồi đo lại bằng
    # scripts/diag_vad.py: chỉ có nghĩa khi năng lượng lúc im thấp hơn hẳn lúc
    # nói. Hai số dưới chỉ dùng khi đã bật.
    use_energy_valley: bool = False
    min_segment_duration: float = 2.5   # chưa đủ dài thì không tìm valley
    valley_ratio: float = 0.12
    valley_min_frames: int = 8          # 8 frame = 256ms
    # Segmenter vẫn có thể nhả mẩu rất ngắn qua nhánh Silence Timeout (nhánh đó
    # không kiểm tra min_segment_duration). Dưới ngưỡng này thì bỏ, xem
    # OneVoicePipeline.on_segment_ready.
    min_utterance_s: float = 0.4
    save_wav_debug: bool = False        # Zero I/O khi nhúng thiết bị

    # ── Hàng đợi STT+MT ─────────────────────────────────────────────────
    # Worker chạy STT+MT chậm hơn người nói. Queue có hạn + bỏ câu cũ nhất để
    # phụ đề không trôi càng lúc càng trễ so với người đang nói.
    queue_maxsize: int = 4
    drop_oldest_when_full: bool = True

    # ── STT ─────────────────────────────────────────────────────────────
    # "sherpa"   = SenseVoice (zh) + Zipformer (vi), tự định tuyến — MẶC ĐỊNH
    # "mock"     = engine giả lập, chạy pipeline khi chưa tải model STT
    # "scripted" = đọc câu từ file text
    stt_engine: str = "sherpa"
    stt_threads: int = 2
    # "dual" chạy cả hai model mỗi câu (chính xác nhất);
    # "vi-first" bỏ qua SenseVoice khi Zipformer đã chắc chắn là tiếng Việt.
    stt_router_mode: str = "dual"

    # ── Ngôn ngữ ────────────────────────────────────────────────────────
    # None = tự nhận diện (theo lang của STT, dự phòng bằng script detector).
    # Đặt "zh" hoặc "vi" để khóa cứng chiều dịch (chính xác hơn khi biết trước
    # ai đang nói, vd chế độ hướng dẫn viên).
    force_src_lang: str = None

    translate: TranslateConfig = field(default_factory=TranslateConfig)
