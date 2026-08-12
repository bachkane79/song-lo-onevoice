"""Đưa model dịch vào repo / vào thiết bị Jetson.

Model 394 MB **không commit vào git được** (GitHub giới hạn 100 MB/file), nên nó
được copy vào `models/mt/` — thư mục đã có trong `.gitignore`. Script này làm việc
copy đó và verify nội dung.

Chỉ cần đúng 2 thứ để chạy máy dịch (tổng ~397 MB):

    models/mt/nllb-zhvi-int8/{model.bin, config.json, shared_vocabulary.json}
    models/mt/tokenizer.json

Dùng trên máy dev (lấy từ thư mục benchmark test-mt):

    python scripts/prepare_model.py --source D:\\Coding\\BK\\test-mt

Dùng trên Jetson (sau khi scp thư mục sang):

    python3 scripts/prepare_model.py --source ~/onevoice-model
    python3 scripts/prepare_model.py --verify-only
"""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from onevoice.config import DEFAULT_MT_MODEL_DIR, DEFAULT_MT_TOKENIZER  # noqa: E402

CT2_FILES = ("model.bin", "config.json", "shared_vocabulary.json")

# Các vị trí thường gặp của model CT2 trong thư mục nguồn, ưu tiên bản trim
CT2_CANDIDATES = (
    os.path.join("ct2", "nllb-trim-int8"),
    "nllb-trim-int8",
    os.path.join("ct2", "nllb-int8"),
    ".",
)
TOKENIZER_CANDIDATES = (
    os.path.join("nllb-zhvi-trimmed", "tokenizer.json"),
    "tokenizer.json",
)

# Mã ngôn ngữ bắt buộc phải có trong tokenizer — nếu thiếu là lấy sai file
REQUIRED_LANG_TOKENS = ("zho_Hans", "vie_Latn")


def _mb(path):
    if os.path.isfile(path):
        return os.path.getsize(path) / 1024.0 ** 2
    return sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(path) for f in fs) / 1024.0 ** 2


def find_ct2_dir(source):
    for rel in CT2_CANDIDATES:
        cand = os.path.normpath(os.path.join(source, rel))
        if all(os.path.isfile(os.path.join(cand, f)) for f in CT2_FILES):
            return cand
    return None


def find_tokenizer(source):
    for rel in TOKENIZER_CANDIDATES:
        cand = os.path.normpath(os.path.join(source, rel))
        if os.path.isfile(cand):
            return cand
    return None


def verify(model_dir, tokenizer_path):
    """Kiểm tra model đã ở đúng chỗ và đúng nội dung. Trả về True/False."""
    ok = True

    print("── Kiểm tra model CTranslate2 ──")
    for f in CT2_FILES:
        p = os.path.join(model_dir, f)
        if os.path.isfile(p):
            print("  ✅ %-24s %8.1f MB" % (f, _mb(p)))
        else:
            print("  ❌ THIẾU %s" % p)
            ok = False

    cfg_path = os.path.join(model_dir, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        # NLLB dùng </s> làm decoder_start_token; nếu khác thì đây là model khác
        # (vd Opus-MT) và cách gọi target_prefix trong translate.py sẽ sai.
        start = cfg.get("decoder_start_token")
        if start == "</s>":
            print("  ✅ decoder_start_token = </s>  (đúng chuẩn NLLB)")
        else:
            print("  ⚠️  decoder_start_token = %r — không phải NLLB?" % start)
            ok = False

    print("── Kiểm tra tokenizer ──")
    if not os.path.isfile(tokenizer_path):
        print("  ❌ THIẾU %s" % tokenizer_path)
        return False
    print("  ✅ %-24s %8.1f MB" % ("tokenizer.json", _mb(tokenizer_path)))
    with open(tokenizer_path, encoding="utf-8") as f:
        tok = json.load(f)
    vocab = tok.get("model", {}).get("vocab", {})
    added = {t["content"] for t in tok.get("added_tokens", [])}
    n_vocab = len(vocab) + len(added - set(vocab))
    print("  ℹ️  vocab ≈ %d token" % n_vocab)
    for lang in REQUIRED_LANG_TOKENS:
        if lang in vocab or lang in added:
            print("  ✅ có mã ngôn ngữ %s" % lang)
        else:
            print("  ❌ THIẾU mã ngôn ngữ %s — sai file tokenizer!" % lang)
            ok = False

    # Cắt vocab và model phải khớp nhau: shared_vocabulary.json của bản CT2
    # cũng phải có đúng số dòng như vocab tokenizer, lệch = ghép sai cặp file.
    sv = os.path.join(model_dir, "shared_vocabulary.json")
    if os.path.isfile(sv):
        with open(sv, encoding="utf-8") as f:
            n_ct2 = len(json.load(f))
        if abs(n_ct2 - n_vocab) <= 8:
            print("  ✅ vocab CT2 (%d) khớp tokenizer (%d)" % (n_ct2, n_vocab))
        else:
            print("  ❌ LỆCH vocab: CT2 có %d, tokenizer có %d — ghép sai cặp "
                  "model/tokenizer, sẽ dịch ra rác!" % (n_ct2, n_vocab))
            ok = False

    return ok


def main():
    ap = argparse.ArgumentParser(description="Copy + verify model dịch zh<->vi")
    ap.add_argument("--source", help="Thư mục chứa ct2/nllb-trim-int8 và "
                                     "nllb-zhvi-trimmed/tokenizer.json")
    ap.add_argument("--dest-model", default=DEFAULT_MT_MODEL_DIR)
    ap.add_argument("--dest-tokenizer", default=DEFAULT_MT_TOKENIZER)
    ap.add_argument("--verify-only", action="store_true",
                    help="Chỉ kiểm tra, không copy")
    ap.add_argument("--force", action="store_true",
                    help="Ghi đè model đã có ở đích")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not args.verify_only:
        if not args.source:
            ap.error("cần --source (hoặc dùng --verify-only)")
        if not os.path.isdir(args.source):
            ap.error("không tìm thấy thư mục nguồn: %s" % args.source)

        ct2_src = find_ct2_dir(args.source)
        tok_src = find_tokenizer(args.source)
        if ct2_src is None:
            print("❌ Không tìm thấy model CTranslate2 trong %s" % args.source)
            print("   Cần một thư mục chứa: %s" % ", ".join(CT2_FILES))
            return 1
        if tok_src is None:
            print("❌ Không tìm thấy tokenizer.json trong %s" % args.source)
            return 1

        print("📦 Nguồn model     : %s  (%.1f MB)" % (ct2_src, _mb(ct2_src)))
        print("📦 Nguồn tokenizer : %s  (%.1f MB)" % (tok_src, _mb(tok_src)))

        if os.path.isdir(args.dest_model) and os.listdir(args.dest_model):
            if not args.force:
                print("⚠️  %s đã có dữ liệu — dùng --force để ghi đè. "
                      "Bỏ qua bước copy." % args.dest_model)
            else:
                shutil.rmtree(args.dest_model)

        if not os.path.isdir(args.dest_model) or args.force:
            os.makedirs(args.dest_model, exist_ok=True)
            for f in CT2_FILES:
                shutil.copy2(os.path.join(ct2_src, f),
                             os.path.join(args.dest_model, f))
            print("✅ Đã copy model → %s" % args.dest_model)

        os.makedirs(os.path.dirname(os.path.abspath(args.dest_tokenizer)),
                    exist_ok=True)
        if os.path.isfile(args.dest_tokenizer) and not args.force:
            print("⚠️  tokenizer.json đã có — bỏ qua (dùng --force để ghi đè)")
        else:
            shutil.copy2(tok_src, args.dest_tokenizer)
            print("✅ Đã copy tokenizer → %s" % args.dest_tokenizer)
        print()

    ok = verify(args.dest_model, args.dest_tokenizer)
    total = 0.0
    if os.path.isdir(args.dest_model):
        total += _mb(args.dest_model)
    if os.path.isfile(args.dest_tokenizer):
        total += _mb(args.dest_tokenizer)
    print("\n%s  Tổng dung lượng cần mang lên Jetson: %.1f MB"
          % ("✅ SẴN SÀNG" if ok else "❌ CHƯA SẴN SÀNG", total))
    if ok:
        print("→ Thử ngay: python -m onevoice.translate \"这个地方怎么走?\"")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
