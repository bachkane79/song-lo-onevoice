"""Địa danh / tên riêng & từ vựng ẩm thực Hán-Việt cho chiều vi→zh.

## Vấn đề (đã đo trên 30 địa danh × 5 ngữ cảnh + bộ từ vựng ẩm thực du lịch)

NLLB-200 là model 200 thứ tiếng, **không có kiến thức Hán-Việt** và bộ vocab
SentencePiece của nó thiếu 331 chữ Hán thông dụng (đặc biệt là bộ chữ ẩm thực/đời
sống như 鸭 vịt, 鸡 gà, 蟹 cua, 虾 tôm, 锅 lẩu, 啤 bia, 圳 Quyến, 滩 Bến...).
Nó sinh ra ba loại lỗi:

  1. **Phiên âm theo âm Latin**:  Hà Nội → 哈诺伊 (đúng: 河内), Sa Pa → 萨帕
  2. **Dịch nghĩa word-by-word thành từ khác**:
        Thượng Hải → 海上      ("trên biển")
        Bắc Kinh   → 在北/北方 ("phía Bắc")
        Sài Gòn    → 七个子    ("bảy đứa con")
        Nha Trang  → 时尚      ("thời trang")
        Cần Thơ    → 需要诗歌  ("cần thơ ca")
        Ninh Bình  → 和平      ("hòa bình")
        Bình Dương → 太平洋    ("Thái Bình Dương")
  3. **Bỏ trống hoặc cắt cụt chữ khi gặp ký tự ngoài vocab (OOV)**:
        vịt quay    → 子       (vì thiếu chữ 鸭)
        mì cua lông → 毛饼     (vì thiếu chữ 蟹)

## Cách làm ở đây: Thay ở ĐẦU VÀO kết hợp Khóa Thực Thể Bất Biến (`ITEM01`, `ITEM02`...)

  **Bước 1** — với các địa danh có toàn bộ ký tự nằm trong vocab NLLB và model
  chép ổn định (như 河内, 上海, 北京), thay trực tiếp vào câu nguồn:

      "Tôi ở Ninh Bình ba ngày"  →  "Tôi ở 宁平 ba ngày"  →  我已经在宁平了三天了 ✅

  **Bước 2 (Khóa thực thể bất biến — Invariant Entity Locking)** — với các địa
  danh hoặc món ăn chứa ký tự ngoài vocab NLLB (như 岘港, 大叻, 芹苴, 外滩, 烤鸭,
  蟹黄面, 火锅, 啤酒...), nếu chèn một cụm Hán tự giả (như "深海鱼面") thì model
  NLLB đôi khi tự ý cắt bớt chữ cuối ("深海鱼") khiến bước hậu xử lý trượt khớp!
  Để bảo toàn tuyệt đối 100%, hệ thống gán mã khóa bất biến dạng `ITEM01`, `ITEM02`...
  (hoặc `carrier` chỉ định) khi đưa vào NLLB, sau đó khôi phục chính xác về Hán tự
  chuẩn `zh` ở đầu ra:

      "Tôi đến Thượng Hải để thử món mì cua lông."
      → "Tôi đến 上海 để thử món ITEM01."
      → NLLB dịch: "我去上海试试ITEM01."
      → Khôi phục: "我去上海试试蟹黄面." ✅ (Giữ nguyên cả động từ 'thử' 试试 và '蟹黄面'!)

Bảng dữ liệu: `onevoice/data/glossary_vi_zh.json`.
Đo lại / mở rộng: `python scripts/probe_glossary.py`.
"""

import json
import os
import re

__all__ = ["Glossary", "GlossaryEntry"]


class GlossaryEntry(object):
    """Một mục từ điển (địa danh hoặc món ăn/thuật ngữ du lịch).

    :param vi: tên tiếng Việt xuất hiện trong câu nguồn (khớp không phân biệt hoa/thường)
    :param zh: Hán tự đúng, sẽ xuất hiện trong bản dịch cuối cùng
    :param carrier: chuỗi thay thế ở câu nguồn. Mặc định = `zh`. Khi `zh` chứa ký tự
        ngoài vocab NLLB hoặc dễ bị NLLB cắt chữ, đặt `"LOCK"` (hoặc mã `ITEMxx` /
        carrier cụ thể) để kích hoạt cơ chế khóa thực thể bất biến 2 bước.
    :param wrong: các chuỗi model từng sinh ra sai — chỉ để tham khảo/tài liệu,
        KHÔNG dùng để thay thế.
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
    """Bảng từ điển áp dụng 2 bước quanh lần gọi model (chỉ chiều vi→zh)."""

    def __init__(self, entries=None):
        self.entries = list(entries or [])

        # Khớp tên dài trước: "Vịnh Hạ Long" phải thắng "Hạ Long",
        # "vịt quay Bắc Kinh" phải thắng "vịt quay" và "Bắc Kinh".
        ordered = sorted(self.entries, key=lambda e: len(e.vi), reverse=True)
        self._src_rules = []
        self._tgt_map = {}

        lock_idx = 1
        for e in ordered:
            if not e.vi:
                continue
            pattern = re.compile(
                r"(?<![^\W\d_])" + re.escape(e.vi) + r"(?![^\W\d_])",
                re.IGNORECASE | re.UNICODE)

            if e.needs_post_edit:
                # Nếu carrier đặt là "LOCK" (hoặc chứa ký tự Hán có nguy cơ bị NLLB
                # cắt bớt ở câu ngắn), tự động cấp mã khóa bất biến ITEM01, ITEM02...
                if e.carrier.upper() == "LOCK":
                    eff_carrier = "ITEM%02d" % lock_idx
                    lock_idx += 1
                else:
                    eff_carrier = e.carrier
                self._tgt_map[eff_carrier] = e.zh
            else:
                eff_carrier = e.zh

            self._src_rules.append((pattern, eff_carrier))

        # Sắp xếp khóa dài trước để ITEM10 không bị ITEM1 ăn mất số 0
        self._tgt_rules = sorted(
            self._tgt_map.items(),
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
            "version": 3,
            "note": ("Từ điển Địa danh & Ẩm thực cho chiều vi→zh, thay ở ĐẦU VÀO. "
                     "'carrier': 'LOCK' kích hoạt Khóa Thực Thể Bất Biến (ITEMxx) "
                     "khi 'zh' chứa ký tự ngoài vocab NLLB hoặc dễ bị cắt chữ."),
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
        """Bước 1: thay tên tiếng Việt trong câu nguồn bằng Hán tự (hoặc mã khóa ITEMxx).

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
        """Bước 2: đổi mã khóa ITEMxx / carrier trong bản dịch về Hán tự đúng."""
        if not zh_text or not self._tgt_rules:
            return zh_text, 0
        hits = 0
        for carrier, zh in self._tgt_rules:
            if carrier in zh_text:
                hits += zh_text.count(carrier)
                # Nếu trước/sau mã ITEMxx có khoảng trắng thừa do tokenizer sinh ra
                # giữa chữ Hán và mã Latin (vd: "吃 ITEM01"), dọn sạch khoảng trắng đó
                zh_text = re.sub(r"\s*" + re.escape(carrier) + r"\s*", zh, zh_text)
        return zh_text, hits

    def lookup_vi(self, vi_term):
        """Tra Hán tự đúng của một mục tiếng Việt."""
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
