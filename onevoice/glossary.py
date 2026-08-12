"""Địa danh / tên riêng Hán-Việt cho chiều vi→zh.

## Vấn đề (đã đo trên 30 địa danh × 5 ngữ cảnh)

NLLB-200 là model 200 thứ tiếng, **không có kiến thức Hán-Việt**. Nó không dịch
địa danh Việt Nam sang Hán tự người Trung Quốc thật sự dùng, mà sinh ra hai loại
lỗi — loại thứ hai nguy hiểm hơn nhiều:

  1. **Phiên âm theo âm Latin**:  Hà Nội → 哈诺伊 (đúng: 河内), Sa Pa → 萨帕
  2. **Dịch nghĩa bừa thành một từ tiếng Hoa thông dụng khác** — nghe rất trôi
     chảy nên người dùng không thể phát hiện:

        Sài Gòn    → 七个子    ("bảy đứa con")
        Nha Trang  → 时尚      ("thời trang")
        Cần Thơ    → 需要诗歌  ("cần thơ ca")
        Ninh Bình  → 和平      ("hòa bình")
        Bình Dương → 太平洋    ("Thái Bình Dương")

**26/30 địa danh sai ít nhất một lần.**

## Vì sao KHÔNG dùng hậu xử lý đơn thuần

Ý tưởng đầu tiên là sửa chuỗi ở đầu ra (哈诺伊 → 河内). Cách này hỏng với loại lỗi
thứ hai: luật `和平 → 宁平` sẽ phá mọi câu nói về "hòa bình" chẳng liên quan gì
đến Ninh Bình. Không thể phân biệt được ở tầng chuỗi.

## Cách làm ở đây: thay ở ĐẦU VÀO (source-side), 2 bước

  **Bước 1** — trước khi dịch, thay tên tiếng Việt trong câu nguồn bằng Hán tự
  đúng. Model chỉ việc chép chuỗi Hán tự đó sang output (đã đo: nó chép rất
  đáng tin, vì output đang là tiếng Hoa):

      "Tôi ở Ninh Bình ba ngày"  →  "Tôi ở 宁平 ba ngày"  →  我已经在宁平了三天了 ✅
      (không thay thì ra 宁波 = Ninh Ba, một thành phố Trung Quốc khác)

  **Bước 2** — với 5 tên mà chính Hán tự đúng lại chứa ký tự không có trong vocab
  NLLB (岘港, 大叻, 芹苴, 邦美蜀, 芒街 — model không thể chép thứ nó không có
  token), ta chèn một chuỗi "vật mang" (`carrier`) nằm trong vocab rồi đổi lại ở
  đầu ra:

      "đi Đà Nẵng" → "đi 达港" → 我想去达港 → hậu xử lý → 我想去岘港 ✅

  Chuỗi carrier phải hiếm gặp trong tiếng Hoa tự nhiên để bước hậu xử lý không
  đụng câu khác — đó là lý do chọn 达港 chứ không phải một từ thông dụng.

Bảng dữ liệu: `onevoice/data/glossary_vi_zh.json`.
Đo lại / mở rộng: `python scripts/probe_glossary.py`.
"""

import json
import os
import re

__all__ = ["Glossary", "GlossaryEntry"]


class GlossaryEntry(object):
    """Một địa danh.

    :param vi: tên tiếng Việt xuất hiện trong câu nguồn (khớp không phân biệt hoa/thường)
    :param zh: Hán tự đúng, sẽ xuất hiện trong bản dịch cuối cùng
    :param carrier: chuỗi Hán tự chèn vào câu nguồn. Mặc định = `zh`. Chỉ cần
        khác `zh` khi `zh` chứa ký tự không có trong vocab model.
    :param wrong: các chuỗi model từng sinh ra sai — chỉ để tham khảo/tài liệu,
        KHÔNG dùng để thay thế (xem docstring module).
    """

    __slots__ = ("vi", "zh", "carrier", "wrong")

    def __init__(self, vi, zh, carrier=None, wrong=None):
        self.vi = vi
        self.zh = zh
        self.carrier = carrier or zh
        self.wrong = list(wrong or [])

    @property
    def needs_post_edit(self):
        return self.carrier != self.zh

    def __repr__(self):
        return "GlossaryEntry(%r → %r%s)" % (
            self.vi, self.zh,
            ", carrier=%r" % self.carrier if self.needs_post_edit else "")


class Glossary(object):
    """Bảng địa danh áp dụng 2 bước quanh lần gọi model (chỉ chiều vi→zh)."""

    def __init__(self, entries=None):
        self.entries = list(entries or [])

        # Khớp tên dài trước: "Vịnh Hạ Long" phải thắng "Hạ Long", nếu không thì
        # thay xong sẽ còn lại chuỗi rác "Vịnh 下龙".
        ordered = sorted(self.entries, key=lambda e: len(e.vi), reverse=True)
        self._src_rules = []
        for e in ordered:
            if not e.vi:
                continue
            # (?<![^\W\d_]) / (?![^\W\d_]) = ranh giới "không kề chữ cái".
            # Không dùng \b vì tiếng Việt có dấu, \b hoạt động sai ở rìa dấu.
            pattern = re.compile(
                r"(?<![^\W\d_])" + re.escape(e.vi) + r"(?![^\W\d_])",
                re.IGNORECASE | re.UNICODE)
            self._src_rules.append((pattern, e.carrier))

        self._tgt_rules = sorted(
            ((e.carrier, e.zh) for e in self.entries if e.needs_post_edit),
            key=lambda p: len(p[0]), reverse=True)

    # ── Khởi tạo ────────────────────────────────────────────────────────────
    @classmethod
    def empty(cls):
        return cls([])

    @classmethod
    def load(cls, path):
        """Nạp từ JSON. Thiếu/hỏng file → glossary rỗng, không làm chết thiết bị."""
        if not path or not os.path.isfile(path):
            return cls.empty()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (ValueError, OSError) as exc:
            print("⚠️  Không đọc được glossary %s: %s" % (path, exc))
            return cls.empty()

        entries = []
        for item in data.get("entries", []):
            vi = (item.get("vi") or "").strip()
            zh = (item.get("zh") or "").strip()
            if not vi or not zh:
                continue
            entries.append(GlossaryEntry(
                vi=vi, zh=zh,
                carrier=(item.get("carrier") or "").strip() or None,
                wrong=[w.strip() for w in item.get("wrong", []) if w and w.strip()]))
        return cls(entries)

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "version": 2,
            "note": ("Địa danh Hán-Việt cho chiều vi→zh, thay ở ĐẦU VÀO. "
                     "'carrier' chỉ cần khi 'zh' chứa ký tự không có trong vocab "
                     "NLLB. 'wrong' chỉ để tham khảo, không dùng để thay thế. "
                     "Đo lại bằng scripts/probe_glossary.py"),
            "entries": [],
        }
        for e in sorted(self.entries, key=lambda x: x.vi):
            row = {"vi": e.vi, "zh": e.zh}
            if e.needs_post_edit:
                row["carrier"] = e.carrier
            if e.wrong:
                row["wrong"] = e.wrong
            payload["entries"].append(row)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    # ── Áp dụng ─────────────────────────────────────────────────────────────
    def apply_source(self, vi_text):
        """Bước 1: thay tên tiếng Việt trong câu nguồn bằng Hán tự (hoặc carrier).

        Trả về ``(text, số_lần_thay)``.
        """
        if not vi_text or not self._src_rules:
            return vi_text, 0
        hits = 0
        for pattern, carrier in self._src_rules:
            vi_text, n = pattern.subn(carrier, vi_text)
            hits += n
        return vi_text, hits

    def apply_target(self, zh_text):
        """Bước 2: đổi carrier trong bản dịch về Hán tự đúng."""
        if not zh_text or not self._tgt_rules:
            return zh_text, 0
        hits = 0
        for carrier, zh in self._tgt_rules:
            if carrier in zh_text:
                hits += zh_text.count(carrier)
                zh_text = zh_text.replace(carrier, zh)
        return zh_text, hits

    def lookup_vi(self, vi_term):
        """Tra Hán tự đúng của một địa danh tiếng Việt."""
        key = (vi_term or "").strip().lower()
        for e in self.entries:
            if e.vi.lower() == key:
                return e.zh
        return None

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        return "Glossary(%d mục, %d cần hậu xử lý)" % (
            len(self.entries), len(self._tgt_rules))
