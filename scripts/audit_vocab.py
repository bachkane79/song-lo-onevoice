"""Kiểm tra lỗ hổng từ vựng của model dịch — công cụ chẩn đoán quan trọng nhất.

NLLB không có `byte_fallback`: ký tự nào không có piece trong vocab sẽ thành
`<unk>` và model **mất hẳn** thông tin đó, nhưng vẫn dịch ra một câu trôi chảy —
người dùng không có cách nào biết. Script này để phát hiện trước khi ship.

Ba chế độ:

    # 1. Ký tự nào trong danh sách từ vựng bị <unk>?
    python scripts/audit_vocab.py --chars

    # 2. Bảng zh_unk_map.json có thật sự chữa được không? (dịch trước/sau)
    python scripts/audit_vocab.py --check-rules

    # 3. Quét file văn bản của mình xem có ký tự nào bị <unk>
    python scripts/audit_vocab.py --scan my_phrases.txt
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from onevoice.config import TranslateConfig  # noqa: E402

# Từ vựng cốt lõi của thiết bị phiên dịch du lịch: chào hỏi, đi lại, lưu trú,
# tiền, y tế, ăn uống. Thêm ký tự vào đây khi mở rộng phạm vi sử dụng.
TRAVEL_CHARS = (
    "你好谢谢对不起请问哪里多少钱贵便宜"
    "厕所卫生间洗手间机场火车站公交地铁出租车"
    "酒店旅馆房间预订护照签证行李寄存换钱银行"
    "医院药店医生疼痛发烧感冒过敏"
    "菜单点菜牛肉猪肉鸡肉鱼虾蟹蔬菜米饭面条汤"
    "辣咸甜酸苦盐糖醋酱油啤酒咖啡茶水冰热烫凉"
    "炒煎炸煮蒸烤鸭锅碗筷勺辣椒葱姜蒜"
    "打包外卖买单发票现金刷卡支付宝微信"
    # Họ người Trung Quốc phổ biến + chữ trong tên tỉnh — nhóm này cũng thiếu
    # nhiều trong vocab NLLB, xem docs/TRANSLATE_GUIDE.md
    "王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林郑谢邓许"
    "浙湘陕闽粤沪皖赣鄂桂黔滇渝苏鲁"
)

# Câu thử cho --check-rules: mỗi câu chứa ít nhất một ký tự bị <unk>
RULE_TEST_SENTENCES = [
    "你好，请问厕所在哪里？",
    "菜太咸了，能不能换一份？",
    "我对虾过敏。",
    "这个鸡肉太辣了。",
    "我要一份炒鸡蛋。",
    "请给我一瓶啤酒。",
    "这个汤很好喝。",
    "小心，很烫！",
    "请给我酱油和醋。",
    "我要一份炒蟹。",
    "这家餐厅的辣椒酱很有名。",
    "汤太咸了。",
    "来两只烤鸭和一碗鸡汤。",
    "我们去吃火锅吧，请给我一双筷子。",
]


def load_vocab(tokenizer_path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(tokenizer_path).get_vocab()


def mode_chars(vocab):
    chars = sorted(set(TRAVEL_CHARS))
    missing = [c for c in chars if c not in vocab]
    print("Đã quét %d ký tự du lịch/ăn uống riêng biệt." % len(chars))
    print("Thiếu piece đơn (→ <unk>): %d ký tự\n" % len(missing))
    if missing:
        print("  " + " ".join(missing))
        print("\n⚠️  Mỗi ký tự trên đều làm model MẤT HẲN nghĩa của từ chứa nó.")
        print("    → Thêm luật thay thế vào onevoice/data/zh_unk_map.json")
        print("    → Rồi chạy: python scripts/audit_vocab.py --check-rules")
    else:
        print("  ✅ Không thiếu ký tự nào.")
    return 0 if not missing else 1


def mode_scan(vocab, path):
    if not os.path.isfile(path):
        print("❌ Không tìm thấy file: %s" % path)
        return 1
    with open(path, encoding="utf-8") as f:
        text = f.read()
    # Chỉ quan tâm ký tự Hán (tiếng Việt dùng Latin, NLLB phủ đủ)
    cjk = {c for c in text if 0x3400 <= ord(c) <= 0x9FFF}
    missing = sorted(c for c in cjk if c not in vocab)
    print("File %s: %d ký tự Hán riêng biệt, %d bị <unk>" % (path, len(cjk), len(missing)))
    if missing:
        print("  " + " ".join(missing))
        for line in text.splitlines():
            bad = [c for c in line if c in missing]
            if bad:
                print("  ⚠️  %s   ← thiếu: %s" % (line.strip(), "".join(sorted(set(bad)))))
    return 0 if not missing else 1


def mode_check_rules(cfg):
    """Dịch từng câu thử 2 lần: có bảng chuẩn hóa và không, rồi so kết quả."""
    from onevoice.translate import NllbTranslator

    off = TranslateConfig(**dict(cfg.__dict__))
    off.use_zh_unk_map = False

    tr_on = NllbTranslator(cfg)
    tr_off = NllbTranslator(off)
    tr_on.warmup()

    print("So sánh CÓ / KHÔNG bảng onevoice/data/zh_unk_map.json (%d luật)\n"
          % len(tr_on.zh_norm))
    changed = 0
    for sent in RULE_TEST_SENTENCES:
        a = tr_off.translate(sent, "zh")
        b = tr_on.translate(sent, "zh")
        flag = "≠" if a.target != b.target else "="
        if a.target != b.target:
            changed += 1
        print("  %s" % sent)
        print("    KHÔNG : %s%s" % (a.target,
                                    "   [<unk>: %s]" % " ".join(a.unk_chars)
                                    if a.unk_chars else ""))
        print("    CÓ  %s : %s%s" % (flag, b.target,
                                     "   [<unk> còn lại: %s]" % " ".join(b.unk_chars)
                                     if b.unk_chars else ""))
        print()

    print("Bảng chuẩn hóa làm đổi kết quả ở %d/%d câu." % (changed, len(RULE_TEST_SENTENCES)))
    print("→ Đọc từng dòng và tự đánh giá bản 'CÓ' có đúng nghĩa hơn không. "
          "Không có metric tự động nào thay được việc này.")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Audit lỗ hổng vocab của model dịch")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--chars", action="store_true",
                   help="Liệt kê ký tự du lịch bị <unk>")
    g.add_argument("--check-rules", action="store_true",
                   help="Dịch thử trước/sau khi áp bảng zh_unk_map.json")
    g.add_argument("--scan", metavar="FILE",
                   help="Quét file văn bản tìm ký tự bị <unk>")
    ap.add_argument("--json", action="store_true", help="In vocab stats dạng JSON")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cfg = TranslateConfig()
    if args.check_rules:
        return mode_check_rules(cfg)

    if not os.path.isfile(cfg.tokenizer_path):
        print("❌ Không tìm thấy tokenizer: %s\n"
              "→ Chạy scripts/prepare_model.py trước." % cfg.tokenizer_path)
        return 1
    vocab = load_vocab(cfg.tokenizer_path)
    if args.json:
        print(json.dumps({"vocab_size": len(vocab)}, ensure_ascii=False))
    if args.chars:
        return mode_chars(vocab)
    return mode_scan(vocab, args.scan)


if __name__ == "__main__":
    sys.exit(main())
