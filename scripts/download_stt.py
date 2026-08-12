"""Tải 2 model STT về `models/stt/` — chạy một lần trên mỗi máy/thiết bị.

    python scripts/download_stt.py            # tải + kiểm tra
    python scripts/download_stt.py --verify-only
    python scripts/download_stt.py --force    # tải lại đè lên

Tải gì:

| model | tiếng | dung lượng | nguồn |
|---|---|---|---|
| SenseVoice int8 | zh (+en/ja/ko/yue) | 228 MB | HF `csukuangfj/sherpa-onnx-sense-voice-...` |
| Zipformer-30M-RNNT | vi | 29 MB | HF `hynt/Zipformer-30M-RNNT-6000h` |

Chỉ tải **bản int8**, không tải bản fp32. Bản tarball trên GitHub của SenseVoice
nặng 1 GB vì gói cả fp32 (937 MB) — thứ mà thiết bị không bao giờ dùng đến. Tải
lẻ từng file qua HF nên chỉ tốn 228 MB.

⚖️ Zipformer tiếng Việt phát hành theo **CC BY-NC-ND 4.0** — phi thương mại,
không phái sinh. Xem `docs/STT_GUIDE.md`.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from onevoice.stt_sherpa import SENSEVOICE_DIR, ZIPFORMER_VI_DIR  # noqa: E402

SENSEVOICE_REPO = "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
SENSEVOICE_FILES = ["model.int8.onnx", "tokens.txt", "LICENSE",
                    "test_wavs/zh.wav", "test_wavs/yue.wav"]

ZIPFORMER_REPO = "hynt/Zipformer-30M-RNNT-6000h"
ZIPFORMER_FILES = ["encoder-epoch-20-avg-10.int8.onnx",
                   "decoder-epoch-20-avg-10.int8.onnx",
                   "joiner-epoch-20-avg-10.int8.onnx",
                   "config.json", "bpe.model", "README.md"]

# File bắt buộc phải có thì engine mới chạy được
REQUIRED = {
    SENSEVOICE_DIR: ["model.int8.onnx", "tokens.txt"],
    ZIPFORMER_VI_DIR: ["encoder-epoch-20-avg-10.int8.onnx",
                       "decoder-epoch-20-avg-10.int8.onnx",
                       "joiner-epoch-20-avg-10.int8.onnx",
                       "config.json", "bpe.model"],
}


def _mb(path):
    if os.path.isfile(path):
        return os.path.getsize(path) / 1024.0 ** 2
    if not os.path.isdir(path):
        return 0.0
    return sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(path) for f in fs) / 1024.0 ** 2


def fetch(repo, files, dest, force):
    import shutil
    from huggingface_hub import hf_hub_download

    os.makedirs(dest, exist_ok=True)
    print("📦 %s → %s" % (repo, dest))
    for rel in files:
        out = os.path.join(dest, rel.replace("/", os.sep))
        if os.path.isfile(out) and not force:
            print("   ⏭  %-42s đã có (%.1f MB)" % (rel, _mb(out)))
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        try:
            cached = hf_hub_download(repo, rel)
        except Exception as exc:                      # noqa: BLE001
            # File phụ (test_wavs, LICENSE) thiếu thì bỏ qua; file bắt buộc sẽ
            # bị bước verify bắt lại nên không cần dừng ở đây.
            print("   ⚠️  %-42s tải lỗi: %s" % (rel, type(exc).__name__))
            continue
        shutil.copy2(cached, out)
        print("   ✅ %-42s %8.1f MB" % (rel, _mb(out)))


def prune():
    """Xóa bản fp32 nếu ai đó giải nén từ tarball GitHub thay vì dùng script này.

    Bản fp32 nặng 937 MB và thiết bị không bao giờ dùng đến — chỉ chạy int8.
    """
    junk = [os.path.join(SENSEVOICE_DIR, "model.onnx"),
            os.path.join(SENSEVOICE_DIR, "export-onnx.py"),
            os.path.join(ZIPFORMER_VI_DIR, "jit_script.pt")]
    freed = 0.0
    for p in junk:
        if os.path.isfile(p):
            freed += _mb(p)
            os.remove(p)
            print("  🗑  xóa %s" % os.path.basename(p))
    if freed:
        print("  ✅ giải phóng %.0f MB (bản fp32, thiết bị chỉ dùng int8)" % freed)
    return freed


def verify():
    ok = True
    print("\n── Kiểm tra ──")
    for dest, files in REQUIRED.items():
        label = os.path.basename(dest)
        missing = [f for f in files if not os.path.isfile(os.path.join(dest, f))]
        if missing:
            ok = False
            print("  ❌ %-18s thiếu: %s" % (label, ", ".join(missing)))
        else:
            print("  ✅ %-18s %8.1f MB" % (label, _mb(dest)))

    if not ok:
        return False

    # Nạp thật một lần: file tồn tại không có nghĩa là dùng được (tải dở, sai
    # kiến trúc, thiếu thư viện...). Đây là bước bắt lỗi thật sự.
    print("\n── Nạp thử recognizer ──")
    try:
        import numpy as np

        from onevoice.stt_sherpa import SenseVoiceSttEngine, ZipformerViSttEngine

        silence = np.zeros(16000, dtype=np.float32)
        for cls, label in ((SenseVoiceSttEngine, "SenseVoice (zh)"),
                           (ZipformerViSttEngine, "Zipformer (vi)")):
            eng = cls()
            eng.transcribe(silence, 16000)
            print("  ✅ %s nạp và chạy được" % label)
    except ImportError as exc:
        print("  ❌ Thiếu thư viện: %s\n     → pip install sherpa-onnx" % exc)
        return False
    except Exception as exc:                          # noqa: BLE001
        print("  ❌ %s: %s" % (type(exc).__name__, exc))
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description="Tải model STT cho OneVoice")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="Tải lại đè lên file cũ")
    ap.add_argument("--zh-only", action="store_true")
    ap.add_argument("--vi-only", action="store_true")
    ap.add_argument("--no-prune", action="store_true",
                    help="Giữ lại bản fp32 (mặc định xóa để tiết kiệm 937 MB)")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not args.verify_only:
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            print("❌ Thiếu huggingface_hub → pip install huggingface_hub")
            return 1
        if not args.vi_only:
            fetch(SENSEVOICE_REPO, SENSEVOICE_FILES, SENSEVOICE_DIR, args.force)
        if not args.zh_only:
            fetch(ZIPFORMER_REPO, ZIPFORMER_FILES, ZIPFORMER_VI_DIR, args.force)

    if not args.no_prune:
        prune()

    ok = verify()
    total = _mb(SENSEVOICE_DIR) + _mb(ZIPFORMER_VI_DIR)
    print("\n%s  STT tổng %.1f MB" % ("✅ SẴN SÀNG" if ok else "❌ CHƯA XONG", total))
    if ok:
        print("→ Chạy pipeline: python run_pipeline.py")
        print("⚖️  Nhắc: Zipformer-vi là CC BY-NC-ND 4.0 (phi thương mại).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
