"""Đưa model dịch vào `models/mt/` — tải từ mạng hoặc copy từ thư mục có sẵn.

Model 394 MB **không commit vào git được** (GitHub giới hạn 100 MB/file), nên nó
được phát hành qua **GitHub Release** và script này tải về + kiểm tra sha256.

    python scripts/prepare_model.py              # tải về (mặc định)
    python scripts/prepare_model.py --source DIR # copy từ thư mục có sẵn
    python scripts/prepare_model.py --verify-only

Chỉ cần đúng 2 thứ để chạy máy dịch (tổng ~397 MB):

    models/mt/nllb-zhvi-int8/{model.bin, config.json, shared_vocabulary.json}
    models/mt/tokenizer.json

**Vì sao phải kiểm tra sha256**: file `model.bin` tải thiếu hoặc hỏng vẫn nạp
được vào CTranslate2 và vẫn dịch ra chữ — chỉ là dịch ra rác. Không có bước
kiểm tra thì lỗi này chỉ lộ ra khi người dùng cuối đọc bản dịch.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from onevoice.config import (  # noqa: E402
    DEFAULT_MT_MODEL_DIR,
    DEFAULT_MT_TOKENIZER,
    DEFAULT_MT_RELEASE,
)

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


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_release_manifest(path=None):
    with open(path or DEFAULT_MT_RELEASE, encoding="utf-8") as f:
        return json.load(f)


def download(url, dest, expect_size=None, expect_sha=None, force=False,
             attempts=4):
    """Tải một file, thử lại khi đứt mạng, kiểm tra sha256 sau khi xong.

    Ba cơ chế chống hỏng, đều cần thiết với file 394 MB:

    - **File tạm `.part`**: chỉ đổi tên sang tên thật khi đã tải đủ và đúng
      sha256. Không bao giờ để lại một file cụt trông như đã tải xong.
    - **Tải tiếp (HTTP Range)**: mạng đứt giữa chừng thì lần thử sau tải tiếp từ
      chỗ dở, không tải lại từ đầu. Cần thiết khi cắm Jetson qua wifi.
    - **Kiểm tra sha256**: `model.bin` thiếu vài MB vẫn nạp được vào CTranslate2
      và vẫn sinh ra chữ — chỉ là chữ rác. Không kiểm tra thì lỗi chỉ lộ ra ở
      người dùng cuối.
    """
    import time
    import urllib.error
    import urllib.request

    name = os.path.basename(dest)
    if os.path.isfile(dest) and not force:
        if expect_sha and sha256_of(dest) == expect_sha:
            print("   ⏭  %-24s đã có, sha256 khớp" % name)
            return True
        print("   ♻️  %-24s đã có nhưng sai sha256 → tải lại" % name)

    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    tmp = dest + ".part"
    # Chỉ vẽ thanh tiến trình khi ra terminal thật. Khi output bị pipe vào file
    # hay log thì "\r" không ghi đè được, mỗi 1 MB sẽ thành một dòng — file
    # model 394 MB sinh ra 400 dòng rác.
    interactive = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    for attempt in range(1, attempts + 1):
        resume_from = os.path.getsize(tmp) if os.path.isfile(tmp) else 0
        if expect_size and resume_from >= expect_size:
            resume_from = 0          # file tạm đã quá dài → hỏng, bỏ tải lại
            _cleanup(tmp)

        label = "   ⬇  %-24s" % name
        if attempt > 1:
            label += " (thử lần %d/%d" % (attempt, attempts)
            label += ", tải tiếp từ %.0f MB)" % (resume_from / 1024.0 ** 2) \
                if resume_from else ")"
        if not interactive:
            if expect_size:
                label += " %.1f MB..." % (expect_size / 1024.0 ** 2)
            print(label)
            sys.stdout.flush()
        elif attempt > 1:
            print(label)

        try:
            req = urllib.request.Request(url)
            if resume_from:
                req.add_header("Range", "bytes=%d-" % resume_from)
            with urllib.request.urlopen(req, timeout=60) as resp:
                # Server bỏ qua Range (trả 200 thay vì 206) → phải ghi lại từ đầu,
                # nếu cứ nối vào file cũ sẽ tạo ra file hỏng mà sha256 mới bắt được.
                if resume_from and resp.status != 206:
                    resume_from = 0
                mode = "ab" if resume_from else "wb"
                total = int(resp.headers.get("Content-Length") or 0) + resume_from
                total = total or expect_size or 0
                done = resume_from
                with open(tmp, mode) as out:
                    while True:
                        block = resp.read(1 << 20)
                        if not block:
                            break
                        out.write(block)
                        done += len(block)
                        if total and interactive:
                            sys.stdout.write(
                                "\r   ⬇  %-24s %5.1f%%  (%.1f/%.1f MB)"
                                % (name, 100.0 * done / total,
                                   done / 1024.0 ** 2, total / 1024.0 ** 2))
                            sys.stdout.flush()
            if total and interactive:
                sys.stdout.write("\r" + " " * 78 + "\r")
                sys.stdout.flush()

        except urllib.error.HTTPError as exc:
            _cleanup(tmp)
            if exc.code == 404:
                print("   ❌ %s: không tìm thấy trên GitHub Release (404)" % name)
                return False       # 404 thì thử lại cũng vô ích
            print("   ⚠️  %s: HTTP %s" % (name, exc.code))
            if attempt == attempts:
                return False
            time.sleep(2 * attempt)
            continue
        except Exception as exc:                      # noqa: BLE001
            # Đứt mạng / DNS lỗi / timeout — GIỮ lại .part để lần sau tải tiếp
            print("   ⚠️  %s: %s: %s" % (name, type(exc).__name__, exc))
            if attempt == attempts:
                print("   ❌ %s: thử %d lần đều lỗi mạng" % (name, attempts))
                return False
            time.sleep(2 * attempt)
            continue

        if expect_sha:
            got = sha256_of(tmp)
            if got != expect_sha:
                _cleanup(tmp)      # hỏng thật → xoá hẳn, lần sau tải lại từ đầu
                print("   ⚠️  %s: sha256 không khớp\n"
                      "        mong đợi %s\n        nhận được %s"
                      % (name, expect_sha[:16], got[:16]))
                if attempt == attempts:
                    print("   ❌ %s: file hỏng hoặc release đã đổi nội dung.\n"
                          "        Nếu bạn vừa phát hành model mới, cập nhật "
                          "sha256 trong onevoice/data/mt_release.json." % name)
                    return False
                continue

        if os.path.isfile(dest):
            os.remove(dest)
        os.rename(tmp, dest)
        print("   ✅ %-24s %8.1f MB  sha256 khớp" % (name, _mb(dest)))
        return True

    return False


def _cleanup(path):
    if os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass


def download_release(manifest, dest_model, dest_tokenizer, force=False):
    """Tải toàn bộ file model từ GitHub Release theo manifest."""
    base = "https://github.com/%s/releases/download/%s" % (
        manifest["repo"], manifest["tag"])
    total_mb = sum(f["size"] for f in manifest["files"]) / 1024.0 ** 2
    print("📥 Tải model dịch từ GitHub Release  %s@%s  (%.0f MB)"
          % (manifest["repo"], manifest["tag"], total_mb))
    print("   %s" % base)

    ok = True
    for item in manifest["files"]:
        if item["dest"] == "tokenizer":
            dest = dest_tokenizer
        else:
            dest = os.path.join(dest_model, item["name"])
        ok = download("%s/%s" % (base, item["name"]), dest,
                      item.get("size"), item.get("sha256"), force) and ok

    if not ok:
        print("\n❌ Tải không đủ file.\n"
              "   Nếu lỗi 404: release %r chưa được publish.\n"
              "   → Xem CLAUDE.md mục 'Phát hành model' để biết cách tạo release,\n"
              "   → hoặc copy tay rồi chạy: python scripts/prepare_model.py --source <dir>"
              % manifest["tag"])
    return ok


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
    ap = argparse.ArgumentParser(
        description="Tải/copy + verify model dịch zh<->vi")
    ap.add_argument("--source", help="Copy từ thư mục có sẵn thay vì tải mạng "
                                     "(thư mục chứa ct2/nllb-trim-int8 và "
                                     "nllb-zhvi-trimmed/tokenizer.json)")
    ap.add_argument("--dest-model", default=DEFAULT_MT_MODEL_DIR)
    ap.add_argument("--dest-tokenizer", default=DEFAULT_MT_TOKENIZER)
    ap.add_argument("--verify-only", action="store_true",
                    help="Chỉ kiểm tra, không tải/copy")
    ap.add_argument("--force", action="store_true",
                    help="Tải/copy lại đè lên file đã có")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Không chỉ định --source → tải từ GitHub Release. Đây là đường mặc định để
    # máy mới clone repo về là chạy được ngay, không cần ai copy tay cho.
    if not args.verify_only and not args.source:
        manifest = load_release_manifest()
        if not download_release(manifest, args.dest_model, args.dest_tokenizer,
                                args.force):
            return 1
        print()
        ok = verify(args.dest_model, args.dest_tokenizer)
        _print_total(ok, args.dest_model, args.dest_tokenizer)
        return 0 if ok else 1

    if not args.verify_only:
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
    _print_total(ok, args.dest_model, args.dest_tokenizer)
    return 0 if ok else 1


def _print_total(ok, dest_model, dest_tokenizer):
    total = 0.0
    if os.path.isdir(dest_model):
        total += _mb(dest_model)
    if os.path.isfile(dest_tokenizer):
        total += _mb(dest_tokenizer)
    print("\n%s  Model dịch: %.1f MB"
          % ("✅ SẴN SÀNG" if ok else "❌ CHƯA SẴN SÀNG", total))
    if ok:
        print("→ Thử ngay: python -m onevoice.translate \"这个地方怎么走?\"")


if __name__ == "__main__":
    sys.exit(main())
