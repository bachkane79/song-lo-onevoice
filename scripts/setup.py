"""Cài đặt toàn bộ OneVoice bằng MỘT lệnh — chạy cái này sau khi clone repo.

    python scripts/setup.py

Nó làm tuần tự:

  1. Kiểm tra Python và các thư viện bắt buộc
  2. Tải model STT (258 MB) — SenseVoice zh + Zipformer vi
  3. Tải model dịch (397 MB) từ GitHub Release, kiểm tra sha256
  4. Chạy thử một câu mỗi chiều để chắc chắn cả pipeline dùng được

Tổng ~655 MB, mất vài phút tuỳ mạng. Chạy lại nhiều lần vô hại: file nào đã có và
đúng sha256 thì bỏ qua.

Tuỳ chọn:

    python scripts/setup.py --check          # chỉ kiểm tra, không tải gì
    python scripts/setup.py --mt-source DIR  # lấy model dịch từ thư mục có sẵn
    python scripts/setup.py --skip-stt       # chỉ lo phần máy dịch
"""

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("ctranslate2", "ctranslate2"),
    ("tokenizers", "tokenizers"),
    ("sherpa_onnx", "sherpa-onnx"),
    ("huggingface_hub", "huggingface_hub"),
]
# Chỉ cần khi chạy với micro thật; thiếu vẫn setup và test được bằng --wav
AUDIO_PACKAGES = [
    ("onnxruntime", "onnxruntime"),
    ("sounddevice", "sounddevice"),
    ("scipy", "scipy"),
]


def step(n, total, title):
    print("\n" + "═" * 70)
    print("  [%d/%d] %s" % (n, total, title))
    print("═" * 70)


def check_python():
    v = sys.version_info
    print("Python %d.%d.%d  (%s)" % (v.major, v.minor, v.micro, sys.executable))
    if v < (3, 8):
        print("❌ Cần Python >= 3.8 (ctranslate2 và sherpa-onnx không có wheel cho 3.6/3.7).")
        print("   Jetson Nano đời cũ dùng Python 3.6 → xem docs/JETSON_SETUP.md mục 1.")
        return False
    return True


def check_packages(install_hint=True):
    missing = []
    for mod, pkg in REQUIRED_PACKAGES:
        try:
            __import__(mod)
            print("  ✅ %s" % pkg)
        except ImportError:
            print("  ❌ %s  (thiếu)" % pkg)
            missing.append(pkg)

    audio_missing = []
    for mod, pkg in AUDIO_PACKAGES:
        try:
            __import__(mod)
            print("  ✅ %s" % pkg)
        except ImportError:
            print("  ⚠️  %s  (thiếu — chỉ cần khi chạy với micro)" % pkg)
            audio_missing.append(pkg)

    if missing and install_hint:
        print("\n❌ Thiếu thư viện bắt buộc. Chạy:\n"
              "     pip install -r requirements.txt")
        return False
    if audio_missing:
        print("\n⚠️  Thiếu thư viện âm thanh → chưa chạy được với micro, nhưng vẫn "
              "test được bằng:\n     python run_pipeline.py --wav file.wav")
    return not missing


def run(script, args):
    cmd = [sys.executable, os.path.join(ROOT, "scripts", script)] + list(args)
    print("$ %s\n" % " ".join(["python", "scripts/" + script] + list(args)))
    # Phải flush trước khi gọi tiến trình con: khi output bị pipe vào file/lệnh
    # khác, stdout của Python là block-buffered nên các dòng của tiến trình cha
    # sẽ dồn đến cuối và output của con hiện sai chỗ, đọc rất khó hiểu.
    sys.stdout.flush()
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return subprocess.call(cmd, cwd=ROOT, env=env) == 0


def smoke_test():
    """Dịch thử một câu mỗi chiều — bước duy nhất chứng minh mọi thứ thật sự chạy."""
    try:
        from onevoice.translate import NllbTranslator
    except ImportError as exc:
        print("❌ Không import được máy dịch: %s" % exc)
        return False

    try:
        tr = NllbTranslator()
        a = tr.translate("你好，请问厕所在哪里？", "zh")
        b = tr.translate("Tôi muốn đi Đà Nẵng.", "vi")
    except Exception as exc:                          # noqa: BLE001
        print("❌ Máy dịch lỗi: %s: %s" % (type(exc).__name__, exc))
        return False

    print("  🇨🇳 → 🇻🇳  你好，请问厕所在哪里？")
    print("            %s" % a.target)
    print("  🇻🇳 → 🇨🇳  Tôi muốn đi Đà Nẵng.")
    print("            %s" % b.target)

    ok = True
    # Hai câu này đi qua đúng hai cơ chế dễ hỏng nhất nếu ghép sai model/tokenizer
    # hoặc thiếu file dữ liệu, nên dùng luôn làm bài kiểm tra.
    if "vệ sinh" not in a.target.lower():
        print("  ⚠️  Kết quả zh→vi bất thường — kiểm tra onevoice/data/zh_unk_map.json")
        ok = False
    if "岘港" not in b.target:
        print("  ⚠️  Kết quả vi→zh bất thường — kiểm tra onevoice/data/glossary_vi_zh.json")
        ok = False

    try:
        from onevoice.stt_sherpa import BilingualSttEngine
        import numpy as np
        stt = BilingualSttEngine(num_threads=1)
        stt.transcribe(np.zeros(16000, dtype="float32"), 16000)
        print("  ✅ STT song ngữ nạp và chạy được")
    except Exception as exc:                          # noqa: BLE001
        print("  ❌ STT lỗi: %s: %s" % (type(exc).__name__, exc))
        ok = False
    return ok


def main():
    ap = argparse.ArgumentParser(description="Cài đặt OneVoice bằng một lệnh")
    ap.add_argument("--check", action="store_true", help="Chỉ kiểm tra, không tải")
    ap.add_argument("--mt-source", help="Lấy model dịch từ thư mục có sẵn "
                                        "thay vì tải từ GitHub Release")
    ap.add_argument("--skip-stt", action="store_true")
    ap.add_argument("--skip-mt", action="store_true")
    ap.add_argument("--force", action="store_true", help="Tải lại đè lên file cũ")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("""
╔══════════════════════════════════════════════════════════════════╗
║  🌏  OneVoice — cài đặt                                          ║
║  Phiên dịch song ngữ Việt ⇄ Hoa, chạy hoàn toàn offline          ║
╚══════════════════════════════════════════════════════════════════╝""")

    total = 4
    step(1, total, "Kiểm tra môi trường")
    if not check_python():
        return 1
    if not check_packages():
        return 1

    extra = ["--force"] if args.force else []

    step(2, total, "Model STT (258 MB)")
    if args.skip_stt:
        print("(bỏ qua theo yêu cầu)")
    elif args.check:
        run("download_stt.py", ["--verify-only"])
    elif not run("download_stt.py", extra):
        print("\n❌ Tải model STT thất bại.")
        return 1

    step(3, total, "Model dịch (397 MB)")
    if args.skip_mt:
        print("(bỏ qua theo yêu cầu)")
    else:
        mt_args = ["--verify-only"] if args.check else (
            ["--source", args.mt_source] if args.mt_source else [])
        if not run("prepare_model.py", mt_args + extra):
            print("\n❌ Chuẩn bị model dịch thất bại.")
            return 1

    step(4, total, "Chạy thử")
    if not smoke_test():
        print("\n⚠️  Có cảnh báo ở bước chạy thử — đọc kỹ phía trên.")
        return 1

    print("""
╔══════════════════════════════════════════════════════════════════╗
║  ✅  XONG. Chạy thiết bị bằng:                                   ║
║                                                                  ║
║      python run_pipeline.py                                      ║
║                                                                  ║
║  Chưa có micro? Thử bằng file .wav:                              ║
║      python run_pipeline.py --wav a.wav --show-route             ║
╚══════════════════════════════════════════════════════════════════╝""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
