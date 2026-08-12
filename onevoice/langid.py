"""Nhận diện ngôn ngữ zh / vi theo hệ chữ viết (script-based).

Đây là lớp **dự phòng**, không phải nguồn thông tin chính. Thứ tự ưu tiên trong
pipeline:

    1. `PipelineConfig.force_src_lang` — người dùng khóa cứng chiều dịch
    2. Mã ngôn ngữ do STT trả về (Whisper / Sherpa-ONNX đều có)
    3. Hàm `detect_lang()` ở file này

Vì sao dùng script chứ không dùng model nhận diện ngôn ngữ: tiếng Việt viết bằng
Latin, tiếng Hoa viết bằng Hán tự — hai tập ký tự rời nhau hoàn toàn. Đếm ký tự
cho độ chính xác gần như 100% với chi phí 0 và không cần thêm model nào lên
thiết bị. Một model langid nhỏ vẫn có thể đoán sai giữa vi và các thứ tiếng
Latin khác, còn cách này thì không thể.
"""

import unicodedata

__all__ = ["detect_lang", "count_scripts", "has_cjk"]

# Các dải Unicode chứa Hán tự (bao gồm cả phồn thể, ext-A/B và ký tự tương thích)
_CJK_RANGES = (
    (0x3400, 0x4DBF),    # CJK ext-A
    (0x4E00, 0x9FFF),    # CJK cơ bản
    (0xF900, 0xFAFF),    # CJK compatibility ideographs
    (0x3000, 0x303F),    # Dấu câu CJK (。、《》…)
    (0xFF00, 0xFFEF),    # Ký tự full-width
    (0x20000, 0x2A6DF),  # CJK ext-B
)

# Ký tự chỉ xuất hiện trong tiếng Việt (không có trong các thứ tiếng Latin khác
# thường gặp) — dùng làm bằng chứng mạnh cho "vi".
_VI_MARKS = set("ăâđêôơưĂÂĐÊÔƠƯ")


def _is_cjk(ch):
    cp = ord(ch)
    for lo, hi in _CJK_RANGES:
        if lo <= cp <= hi:
            return True
    return False


def has_cjk(text):
    """True nếu chuỗi chứa ít nhất một Hán tự."""
    return any(_is_cjk(ch) for ch in (text or ""))


def count_scripts(text):
    """Đếm ký tự theo hệ chữ.

    Trả về dict ``{"cjk": n, "latin": n, "vi_marks": n, "other": n}``.
    Dấu câu ASCII, chữ số và khoảng trắng bị bỏ qua vì xuất hiện ở cả hai tiếng.
    """
    counts = {"cjk": 0, "latin": 0, "vi_marks": 0, "other": 0}
    for ch in text or "":
        if ch.isspace() or ch.isdigit():
            continue
        if _is_cjk(ch):
            counts["cjk"] += 1
        elif ch.isalpha():
            counts["latin"] += 1
            # Chuẩn hóa NFD để bắt cả nguyên âm có dấu tổ hợp (ế = e + ́)
            base = unicodedata.normalize("NFD", ch)
            if ch in _VI_MARKS or (len(base) > 1 and base[0].isalpha()):
                counts["vi_marks"] += 1
        elif not unicodedata.category(ch).startswith("P"):
            counts["other"] += 1
    return counts


def detect_lang(text, default=None, cjk_ratio_threshold=0.15):
    """Đoán ngôn ngữ của `text`: trả về ``"zh"``, ``"vi"`` hoặc `default`.

    :param cjk_ratio_threshold: tỉ lệ Hán tự / (Hán tự + Latin) tối thiểu để kết
        luận là tiếng Hoa. Ngưỡng thấp (0.15) vì câu tiếng Hoa rất hay lẫn từ
        Latin ("GPU 有 8GB 显存") trong khi câu tiếng Việt gần như không bao giờ
        chứa Hán tự.

    Trả về `default` khi chuỗi không có ký tự chữ nào (chỉ số, dấu câu, rỗng) —
    lúc đó pipeline nên bỏ qua câu thay vì dịch bừa.
    """
    if not text:
        return default

    c = count_scripts(text)
    letters = c["cjk"] + c["latin"]
    if letters == 0:
        return default

    if c["cjk"] > 0:
        if c["cjk"] / float(letters) >= cjk_ratio_threshold:
            return "zh"
        # Rất ít Hán tự lẫn trong câu Latin dài → coi là tiếng Việt
        return "vi"

    return "vi"
