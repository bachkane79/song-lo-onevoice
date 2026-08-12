"""OneVoice — thiết bị phiên dịch song ngữ Việt ⇄ Hoa chạy hoàn toàn offline.

Luồng xử lý:  Mic → VAD → Segmenter → STT → Translate → Display

Package này chứa các tầng SAU Segmenter (module VAD/Segmenter nằm nguyên trong
`VAD-Segmenter/`, không sửa đổi):

    onevoice.translate  — máy dịch NLLB-600M vocab-trim (CTranslate2 int8), zh ⇄ vi
    onevoice.langid     — nhận diện ngôn ngữ theo script (dự phòng khi STT không trả lang)
    onevoice.glossary   — hậu xử lý địa danh / tên riêng Hán-Việt (vi→zh)
    onevoice.stt        — interface + engine giả lập, chỗ cắm cho thành viên làm STT
    onevoice.display    — tầng hiển thị ra màn hình
    onevoice.pipeline   — nối toàn bộ, chạy dịch trên worker thread riêng
"""

from onevoice.config import (  # noqa: F401
    LANG_VI,
    LANG_ZH,
    PipelineConfig,
    TranslateConfig,
)
from onevoice.langid import detect_lang  # noqa: F401
from onevoice.translate import NllbTranslator, TranslationResult  # noqa: F401

__all__ = [
    "LANG_VI",
    "LANG_ZH",
    "PipelineConfig",
    "TranslateConfig",
    "NllbTranslator",
    "TranslationResult",
    "detect_lang",
]

__version__ = "0.1.0"
