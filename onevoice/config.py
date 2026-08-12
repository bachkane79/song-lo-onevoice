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
    # 0 = tắt. Bật (vd 3) nếu thấy STT/MT lặp cụm từ; có thể hại tiếng Hoa nên
    # mặc định để tắt, giống đúng cấu hình đã benchmark.
    no_repeat_ngram_size: int = 0

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
    mic_device_id: int = None

    # ── Segmenter (giữ đúng mặc định đã tinh chỉnh của Module 2) ─────────
    silence_timeout: float = 0.6
    min_segment_duration: float = 1.2
    max_segment_duration: float = 8.0
    valley_ratio: float = 0.25
    valley_min_frames: int = 3
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
