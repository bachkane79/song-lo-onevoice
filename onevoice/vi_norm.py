"""Chuẩn hóa câu tiếng Việt (đặc biệt từ STT Zipformer) trước khi đưa vào máy dịch.

## Vì sao cần module này (đã đo trên `final_verify.jsonl`)

STT tiếng Việt (`Zipformer-30M-RNNT-6000h`) xuất ra chuỗi chữ **không có dấu câu**
(không có `.`, `,`, `?`) và **viết thường toàn bộ từ giữa câu**. Trong khi đó,
model dịch `NLLB-200-distilled-600M` được huấn luyện trên văn viết chuẩn nên gặp
3 lỗi nghiêm trọng khi nhận câu STT thô:

1. **Thiếu dấu hỏi `?` hoặc dấu chấm `.` cuối câu làm rơi mất vế câu hoặc đứt đuôi**:
   - `"Bạn có biết nhà hàng nào nổi tiếng với món đó không"` (không có `?`)
     → `你知道哪些餐厅都很有名?` (mất hẳn vế `"với món đó"`)
   - Khi thêm dấu `?` cuối câu:
     → `你知道有哪些餐厅以这种食物而闻名?` (đầy đủ 100% nghĩa gốc!)
   - `"Ở việt nam chúng tôi có một món đặc sản đó chính là bún bò"` (không có `.` và `,`)
     → `在越南,我们有一种特色,` (bị đứt cụt đuôi ở dấu phẩy!)
   - Khi khôi phục dấu `,` và `.`:
     → `在越南,我们有一个特色,就是牛肉粉.` (đầy đủ cả câu!)

2. **Thiếu dấu phẩy `,` ngăn cách lời chào / trạng ngữ chỉ nơi chốn mở đầu**:
   - `"Xin chào bạn đây là..."` → `"Xin chào bạn, đây là..."`
   - `"Ở Bắc Kinh tôi..."`     → `"Ở Bắc Kinh, tôi..."`

3. **Các cụm khẩu ngữ Việt bị NLLB dịch word-by-word (đen nghĩa)**:
   - `"chỉ cho tôi (quán ăn / nhà hàng / chỗ...)"` bị dịch đen chữ `"chỉ"` = `"nhìn"`
     (`给我看` = cho tôi xem) thay vì `"giới thiệu / chỉ dẫn"` (`向我介绍` / `告诉我`).
   - `"tôi biết được là"` bị dịch lệch thành `"tôi nghe nói"`; chuẩn hóa về `"tôi biết là"`.
"""

import re

__all__ = ["ViSourceNormalizer", "restore_vi_punctuation"]

# Các từ/cụm từ nghi vấn cuối câu tiếng Việt → cần gắn dấu '?'
_QUESTION_END_RE = re.compile(
    r"(?:^|\s)(?:"
    r"không|được không|phải không|đúng không|có phải không|"
    r"chưa|xong chưa|được chưa|"
    r"hả|nhỉ|sao|thế nào|như thế nào|ra sao|"
    r"ở đâu|đâu|nào|mấy giờ|bao giờ|khi nào|"
    r"bao nhiêu|bao nhiêu tiền|mấy|gì|cái gì|là gì|vì sao|tại sao"
    r")(?:\s+(?:ạ|nhé|nha|vậy|thế|hả|nhỉ))?\s*$",
    re.IGNORECASE | re.UNICODE,
)

# Ngắt dấu phẩy sau lời chào mở đầu câu nếu theo sau là đại từ / chủ ngữ mới
_GREETING_COMMA_RE = re.compile(
    r"^(xin chào(?:\s+(?:bạn|anh|chị|em|cô|chú|bác|mọi người|các bạn))?|"
    r"chào\s+(?:bạn|anh|chị|em|cô|chú|bác|mọi người|các bạn)|"
    r"cho\s+(?:tôi|mình|em)\s+hỏi)"
    r"\s+(?=(?:đây|tôi|mình|chúng tôi|bạn|anh|chị|ở|có|làm ơn|cho))",
    re.IGNORECASE | re.UNICODE,
)

# Ngắt dấu phẩy sau trạng ngữ nơi chốn mở đầu ("Ở <Địa danh> tôi/bạn/chúng tôi...")
# Hỗ trợ cả địa danh tiếng Việt và Hán tự/carrier sau khi Glossary đã thay thế.
_LOCATIVE_COMMA_RE = re.compile(
    r"^((?:ở|tại)\s+(?:[^\s,?.!]+(?:\s+[^\s,?.!]+){0,2}?))"
    r"\s+(?=(?:tôi|mình|chúng tôi|chúng ta|bạn|anh|chị|em|người ta|họ)\b)",
    re.IGNORECASE | re.UNICODE,
)

# Ngắt dấu phẩy trước mệnh đề giải thích ", đó chính là" / ", nhưng hóa ra"
_MID_CLAUSE_COMMA_RE = re.compile(
    r"(?<![,?.!])\s+(?=(?:đó chính là|nhưng hóa ra|nhưng thật ra|vì vậy|cho nên)\b)",
    re.IGNORECASE | re.UNICODE,
)

# Chuẩn hóa các cụm khẩu ngữ hay bị NLLB dịch word-by-word
_COLLOQUIAL_RULES = [
    # "chỉ cho tôi quán ăn / nhà hàng / một người..." -> "giới thiệu cho tôi..."
    # (Tránh bị NLLB dịch chữ "chỉ" thành "给我看" = "cho tôi nhìn")
    (
        re.compile(
            r"\bchỉ\s+cho\s+(tôi|mình|chúng tôi|em)\s+"
            r"(?=(?:một\s+|vài\s+|những\s+)?(?:quán|nhà hàng|khách sạn|chỗ|nơi|địa điểm|món|người)\b)",
            re.IGNORECASE | re.UNICODE,
        ),
        r"giới thiệu cho \1 ",
    ),
    # "nói cho tôi một người / một quán / một nhà hàng..." -> "giới thiệu cho tôi..."
    (
        re.compile(
            r"\bnói\s+cho\s+(tôi|mình|chúng tôi|em)\s+"
            r"(?=(?:một\s+|vài\s+)?(?:người|quán|nhà hàng|khách sạn|địa điểm|món)\b)",
            re.IGNORECASE | re.UNICODE,
        ),
        r"giới thiệu cho \1 ",
    ),
    # "biết được là" -> "biết là" (giúp cấu trúc câu gọn, tránh dịch sót chủ ngữ sau đó)
    (
        re.compile(r"\bbiết\s+được\s+là\b", re.IGNORECASE | re.UNICODE),
        "biết là",
    ),
    # Sửa lỗi nghe nhầm âm vị rất phổ biến của STT Zipformer với món ăn:
    # "vịt oai" -> "vịt quay", "mì cu long" / "mì cua long" -> "mì cua lông"
    (
        re.compile(r"\bvịt\s+oai\b", re.IGNORECASE | re.UNICODE),
        "vịt quay",
    ),
    (
        re.compile(r"\bmì\s+cu(?:a)?\s+long\b", re.IGNORECASE | re.UNICODE),
        "mì cua lông",
    ),
]


def restore_vi_punctuation(text: str, normalize_colloquial: bool = True) -> tuple:
    """Khôi phục dấu câu (`,`, `?`, `.`) và chuẩn hóa khẩu ngữ cho câu tiếng Việt.

    Chỉ can thiệp khi câu chưa có dấu kết câu (như output thô từ STT Zipformer)
    hoặc khớp các mẫu khẩu ngữ rõ ràng. Trả về ``(text_đã_chuẩn_hóa, số_thay_đổi)``.
    """
    if not text:
        return "", 0
    s = text.strip()
    if not s:
        return "", 0

    edits = 0

    # 1. Chuẩn hóa cụm khẩu ngữ dễ bị dịch word-by-word
    if normalize_colloquial:
        for pattern, repl in _COLLOQUIAL_RULES:
            s, n = pattern.subn(repl, s)
            edits += n

    # 2. Chỉ tự động thêm dấu phẩy & dấu kết câu nếu là câu thô chưa có dấu kết câu (. ? ! …)
    has_end_punct = s[-1] in ".?!…;:"
    if not has_end_punct:
        s, n1 = _GREETING_COMMA_RE.subn(r"\1, ", s)
        s, n2 = _LOCATIVE_COMMA_RE.subn(r"\1, ", s)
        s, n3 = _MID_CLAUSE_COMMA_RE.subn(r", ", s)
        edits += n1 + n2 + n3

        if _QUESTION_END_RE.search(s):
            s = s + "?"
        else:
            s = s + "."
        edits += 1

    return s, edits


class ViSourceNormalizer(object):
    """Bộ chuẩn hóa câu nguồn tiếng Việt trước khi đưa vào NLLB."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def normalize(self, text: str) -> tuple:
        if not self.enabled or not text:
            return text, 0
        return restore_vi_punctuation(text, normalize_colloquial=True)
