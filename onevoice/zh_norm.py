"""Chuẩn hóa văn bản tiếng Hoa ở ĐẦU VÀO để tránh `<unk>`.

## Vấn đề (đã đo, không phải phỏng đoán)

Bộ từ vựng SentencePiece của NLLB-200 **thiếu hẳn 11 ký tự Hán rất thường dùng**
trong hội thoại du lịch / ăn uống. Không phải do bước cắt vocab — model NLLB gốc
cũng thiếu y hệt (đã đối chiếu, output giống nhau từng ký tự), và không có piece
nào chứa các ký tự đó:

    厕 (nhà vệ sinh)  咸 (mặn)   啤 (bia)    汤 (canh)   烫 (nóng bỏng)
    虾 (tôm)          蟹 (cua)   辣 (cay)    酱 (tương)  醋 (giấm)   鸡 (gà)

NLLB không có `byte_fallback`, nên thiếu piece 1 ký tự = ký tự đó thành `<unk>`
và model **mất hoàn toàn** thông tin. Hậu quả thật:

    你好，请问厕所在哪里？  →  "Xin chào, hãy hỏi bà mẹ ở đâu?"    (厕 = <unk>)
    我对虾过敏。            →  "Tôi bị dị ứng với bơ."             (虾 = <unk>)
    这个鸡肉太辣了。         →  "Con gà này quá ngốc."             (鸡, 辣 = <unk>)

Với câu dài (tin tức, kỹ thuật, thương mại) model dịch rất tốt — đây là lỗi tập
trung ở đúng mảng từ vựng du lịch/ăn uống, tức đúng mảng thiết bị này cần.

## Cách xử lý

Thay cụm chứa ký tự `<unk>` bằng cụm **tương đương mà vocab có**, trước khi
tokenize. Mỗi phương án dưới đây đều đã chạy thử thật và chọn bản cho kết quả tốt
nhất; xem `scripts/audit_vocab.py` để đo lại và mở rộng bảng.

Có hai loại thay thế:

  1. **Đồng nghĩa tiếng Hoa** — tốt nhất khi có từ đồng nghĩa đúng nghĩa:
     厕所 → 卫生间  ⇒  "Xin chào, xin hỏi nhà vệ sinh ở đâu?"  ✅
     太辣 → 太辛辣  ⇒  "quá cay"  ✅

  2. **Chèn từ tiếng Anh** — khi tiếng Hoa không có từ đồng nghĩa nào nằm trong
     vocab. NLLB là model 200 thứ tiếng nên nó dịch thẳng từ tiếng Anh nằm giữa
     câu tiếng Hoa sang tiếng Việt rất chính xác:
     虾 → shrimp  ⇒  "Tôi bị dị ứng với tôm."  ✅  (trước đó: "bơ")
     咸 → salty   ⇒  "Bữa ăn quá mặn."         ✅  (trước đó: "quá đậm")

⚠️ **Giới hạn quan trọng:** cách này chỉ chữa được chiều **zh→vi**. Chiều
**vi→zh** không chữa được bằng hậu xử lý, vì model không có token để *sinh ra*
các ký tự đó — nó bỏ hẳn từ:

    "Tôi bị dị ứng với tôm."  →  我对有过敏.     (mất "tôm")
    "Món này cay quá."        →  这个很苦.       ("đắng", sai nghĩa)
    "Món này mặn quá."        →  这太了.         (mất "mặn")

Đây là trần cứng của vocab NLLB. Xem `docs/TRANSLATE_GUIDE.md` mục "Hạn chế đã
biết" để biết hướng khắc phục thật (đổi model có vocab phủ đủ).
"""

import json
import os
import re

__all__ = ["ZhSourceNormalizer"]

_WS_RE = re.compile(r"\s{2,}")


class ZhSourceNormalizer(object):
    """Thay cụm tiếng Hoa chứa ký tự `<unk>` bằng cụm tương đương có trong vocab."""

    def __init__(self, rules=None):
        # Sắp theo độ dài giảm dần để cụm dài được ưu tiên:
        # "太咸" phải khớp trước "咸", nếu không "咸" thay lẻ sẽ tạo chuỗi lạ.
        self.rules = sorted((rules or []), key=lambda r: len(r[0]), reverse=True)

    @classmethod
    def empty(cls):
        return cls([])

    @classmethod
    def load(cls, path):
        """Nạp bảng thay thế từ JSON. Thiếu file → normalizer rỗng (không crash)."""
        if not path or not os.path.isfile(path):
            return cls.empty()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (ValueError, OSError) as exc:
            print("⚠️  Không đọc được bảng chuẩn hóa zh %s: %s" % (path, exc))
            return cls.empty()

        rules = []
        for item in data.get("rules", []):
            src = (item.get("from") or "").strip()
            dst = item.get("to")
            if src and dst is not None:
                rules.append((src, dst))
        return cls(rules)

    def normalize(self, text):
        """Trả về ``(text_đã_thay, số_lần_thay)``."""
        if not text or not self.rules:
            return text, 0
        n = 0
        for src, dst in self.rules:
            if src in text:
                n += text.count(src)
                text = text.replace(src, dst)
        if n:
            # Từ tiếng Anh chèn vào có khoảng trắng hai bên (đã đo: thiếu khoảng
            # trắng thì tokenizer ghép nó vào ký tự Hán liền kề và model dịch sai
            # — "菜太salty了" ra "quá ngọt", còn "菜太 salty 了" ra "quá mặn").
            # Nhiều luật liên tiếp sinh khoảng trắng kép → gom lại.
            text = _WS_RE.sub(" ", text).strip()
        return text, n

    def __len__(self):
        return len(self.rules)

    def __repr__(self):
        return "ZhSourceNormalizer(%d rules)" % len(self.rules)
