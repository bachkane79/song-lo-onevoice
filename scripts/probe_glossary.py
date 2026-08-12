"""Đo xem model phiên âm sai địa danh Việt Nam thành chữ gì (chiều vi→zh).

NLLB không có kiến thức Hán-Việt: nó phiên âm âm Latin thay vì dùng Hán tự mà
người Trung Quốc thật sự dùng (Đà Nẵng → 达南 thay vì 岘港). Script này chạy thật
từng địa danh qua model, ghi lại đúng chuỗi model xuất ra, rồi sinh
`onevoice/data/glossary_vi_zh.json` để tầng hậu xử lý sửa lại.

    # Xem model dịch ra gì (không ghi file)
    python scripts/probe_glossary.py

    # Ghi kết quả vào glossary (gộp với các mục đã có)
    python scripts/probe_glossary.py --write

⚠️ ĐỌC LẠI file glossary sau khi ghi. Một chuỗi phiên âm sai có thể tình cờ là
một từ tiếng Hoa hợp lệ khác (vd 达南 cũng là tên riêng khác), lúc đó luật thay
thế sẽ gây lỗi ở câu không liên quan. Script tự loại chuỗi dài < 2 ký tự vì rủi
ro quá cao, nhưng phần còn lại vẫn cần người xem.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from onevoice.config import TranslateConfig  # noqa: E402
from onevoice.glossary import Glossary, GlossaryEntry  # noqa: E402

# (tên tiếng Việt, Hán tự người Trung Quốc thật sự dùng)
# Nguồn: tên gọi tiêu chuẩn trong tiếng Trung cho địa danh Việt Nam.
PLACES = [
    ("Hà Nội", "河内"),
    ("Hồ Chí Minh", "胡志明市"),
    ("Sài Gòn", "西贡"),
    ("Đà Nẵng", "岘港"),
    ("Hải Phòng", "海防"),
    ("Huế", "顺化"),
    ("Hạ Long", "下龙"),
    ("Vịnh Hạ Long", "下龙湾"),
    ("Nha Trang", "芽庄"),
    ("Đà Lạt", "大叻"),
    ("Phú Quốc", "富国岛"),
    ("Cần Thơ", "芹苴"),
    ("Hội An", "会安"),
    ("Sa Pa", "沙巴"),
    ("Ninh Bình", "宁平"),
    ("Vũng Tàu", "头顿"),
    ("Quảng Ninh", "广宁"),
    ("Phan Thiết", "潘切"),
    ("Mũi Né", "美奈"),
    ("Quy Nhơn", "归仁"),
    ("Buôn Ma Thuột", "邦美蜀"),
    ("Lào Cai", "老街"),
    ("Lạng Sơn", "谅山"),
    ("Móng Cái", "芒街"),
    ("Cao Bằng", "高平"),
    ("Thanh Hóa", "清化"),
    ("Bình Dương", "平阳"),
    ("Đồng Nai", "同奈"),
    ("Bắc Ninh", "北宁"),
    ("Việt Nam", "越南"),
]

# Nhiều ngữ cảnh khác nhau: model có thể phiên âm khác nhau tùy câu, cần bắt hết.
TEMPLATES = [
    "{}",
    "Tôi muốn đi {}.",
    "Cho tôi một vé đi {}.",
    "{} có gì đẹp không?",
    "Chuyến bay đến {} mấy giờ khởi hành?",
]


def cjk_only(text):
    """Lọc lấy phần Hán tự liên tục dài nhất trong chuỗi."""
    runs, cur = [], []
    for ch in text:
        if 0x3400 <= ord(ch) <= 0x9FFF:
            cur.append(ch)
        else:
            if cur:
                runs.append("".join(cur))
            cur = []
    if cur:
        runs.append("".join(cur))
    return runs


def main():
    ap = argparse.ArgumentParser(description="Đo phiên âm sai địa danh vi→zh")
    ap.add_argument("--write", action="store_true",
                    help="Ghi vào onevoice/data/glossary_vi_zh.json")
    ap.add_argument("--min-len", type=int, default=2,
                    help="Bỏ qua chuỗi sai ngắn hơn N ký tự (mặc định 2)")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from onevoice.translate import NllbTranslator

    cfg = TranslateConfig()
    cfg.use_glossary = False       # phải tắt, nếu không sẽ đo chính output đã sửa
    tr = NllbTranslator(cfg)
    tr.warmup()

    entries, n_wrong = [], 0
    for vi, zh in PLACES:
        wrong = []
        for tpl in TEMPLATES:
            out = tr.translate(tpl.format(vi), "vi").target
            for run in cjk_only(out):
                # Bỏ qua nếu model đã dùng đúng Hán tự (hoặc chứa nó)
                if zh in run or run in zh:
                    continue
                # Chỉ giữ chuỗi có độ dài tương đương tên riêng, tránh bắt cả
                # phần còn lại của câu ("我想要去" ...)
                if not (args.min_len <= len(run) <= len(zh) + 2):
                    continue
                if run not in wrong:
                    wrong.append(run)

        status = "✅ đúng" if not wrong else "❌ sai: " + " ".join(wrong)
        print("  %-16s → %-8s %s" % (vi, zh, status))
        if wrong:
            n_wrong += 1
        entries.append(GlossaryEntry(vi=vi, zh=zh, wrong=wrong))

    print("\n%d/%d địa danh bị phiên âm sai ít nhất một lần." % (n_wrong, len(PLACES)))

    if args.write:
        path = TranslateConfig().glossary_path
        old = Glossary.load(path)
        merged = {e.vi: e for e in old.entries}
        for e in entries:
            if e.vi in merged:
                seen = merged[e.vi].wrong
                for w in e.wrong:
                    if w not in seen:
                        seen.append(w)
            else:
                merged[e.vi] = e
        g = Glossary(sorted(merged.values(), key=lambda e: e.vi))
        g.save(path)
        print("\n✅ Đã ghi %d mục vào %s" % (len(g), path))
        print("⚠️  MỞ FILE RA ĐỌC LẠI trước khi ship — xem docstring đầu file.")
    else:
        print("\n(Chạy lại với --write để ghi vào glossary)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
