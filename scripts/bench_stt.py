"""Đo tầng STT trên máy thật + kiểm chứng luật định tuyến ngôn ngữ.

    # Đo tốc độ trên các file .wav mẫu đi kèm model
    python3 scripts/bench_stt.py

    # Đo trên bộ .wav của mình (đặt tên bắt đầu bằng zh_ hoặc vi_ để chấm điểm)
    python3 scripts/bench_stt.py --wav-dir my_audio/

    # Chỉ kiểm tra luật định tuyến (cần file đặt tên zh_*.wav / vi_*.wav)
    python3 scripts/bench_stt.py --wav-dir my_audio/ --routing

Đọc kết quả:

- **RTF** (real-time factor) = thời gian xử lý / độ dài audio. RTF 0.05 nghĩa là
  câu 4 giây mất 0.2 giây để nhận dạng. Cộng RTF của cả STT và máy dịch rồi so
  với 1.0: tổng phải **nhỏ hơn nhiều so với 1.0**, vì thiết bị còn phải chờ người
  nói xong mới bắt đầu xử lý.
- Chế độ `dual` chạy cả hai model mỗi câu nên RTF ≈ tổng của hai. Nếu Jetson
  không kịp thì đổi `--mode vi-first` hoặc đặt `--force-lang`.
"""

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from onevoice.stt_sherpa import (  # noqa: E402
    SENSEVOICE_DIR,
    BilingualSttEngine,
    SenseVoiceSttEngine,
    ZipformerViSttEngine,
)


def read_wav(path):
    """Đọc .wav về float32 mono 16 kHz."""
    import numpy as np

    try:
        import soundfile as sf
        audio, sr = sf.read(path, dtype="float32")
        audio = np.asarray(audio, dtype=np.float32)
    except ImportError:
        import wave
        with wave.open(path, "rb") as w:
            if w.getsampwidth() != 2:
                raise ValueError("cần soundfile để đọc %s" % path)
            audio = (np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
                     .astype(np.float32) / 32768.0)
            if w.getnchannels() > 1:
                audio = audio.reshape(-1, w.getnchannels())[:, 0]
            sr = w.getframerate()
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != 16000:
        import math

        from scipy.signal import resample_poly
        g = math.gcd(16000, sr)
        audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
        sr = 16000
    return audio, sr


def collect_wavs(wav_dir):
    if wav_dir:
        files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    else:
        # Mặc định: file mẫu đi kèm SenseVoice (chỉ có tiếng Hoa/Quảng)
        files = sorted(glob.glob(os.path.join(SENSEVOICE_DIR, "test_wavs", "*.wav")))
    out = []
    for f in files:
        base = os.path.basename(f).lower()
        # Nhãn suy từ tên file: zh_xxx.wav / vi_xxx.wav / zh.wav
        truth = None
        if base.startswith("vi_") or base == "vi.wav":
            truth = "vi"
        elif base.startswith("zh_") or base in ("zh.wav", "yue.wav"):
            truth = "zh"
        out.append((f, truth))
    return out


def bench_single(engine, wavs, label):
    total_audio = total_ms = 0.0
    print("\n=== %s ===" % label)
    for path, _ in wavs:
        audio, sr = read_wav(path)
        dur = len(audio) / float(sr)
        t0 = time.perf_counter()
        res = engine.transcribe(audio, sr)
        ms = (time.perf_counter() - t0) * 1000.0
        total_audio += dur
        total_ms += ms
        print("  %-18s %5.1fs → %6.0f ms (RTF %.3f)  %s"
              % (os.path.basename(path), dur, ms, ms / 1000.0 / dur, res.text[:46]))
    rtf = (total_ms / 1000.0) / max(total_audio, 1e-6)
    print("  ── tổng: %.1fs audio, %.1fs xử lý, RTF %.3f"
          % (total_audio, total_ms / 1000.0, rtf))
    return rtf


def bench_routing(engine, wavs):
    labelled = [(p, t) for p, t in wavs if t]
    if not labelled:
        print("\n⚠️  Không có file nào đặt tên zh_*.wav / vi_*.wav → bỏ qua chấm "
              "điểm định tuyến.")
        return
    print("\n=== Kiểm chứng định tuyến ngôn ngữ ===")
    ok = wrong = skipped = 0
    for path, truth in labelled:
        audio, sr = read_wav(path)
        res = engine.transcribe(audio, sr)
        d = res.detail or {}
        route = d.get("route", res.lang)
        if route == "skip" or not res.text:
            skipped += 1
            mark = "⏭"
        elif route == truth:
            ok += 1
            mark = "✅"
        else:
            wrong += 1
            mark = "❌"
        print("  %s %-18s thật=%s đoán=%-4s vi_rate=%-5s zh_rate=%-5s  %s"
              % (mark, os.path.basename(path), truth, route,
                 d.get("vi_rate"), d.get("zh_rate"), res.text[:40]))
    decided = ok + wrong
    print("  ── đúng %d/%d (%.1f%%), bỏ %d"
          % (ok, decided, 100.0 * ok / max(decided, 1), skipped))
    if wrong:
        print("     Sai nhiều → cân nhắc chạy với --force-lang khi biết trước "
              "ai đang nói.")


def main():
    ap = argparse.ArgumentParser(description="Đo STT + kiểm chứng định tuyến")
    ap.add_argument("--wav-dir", help="Thư mục .wav (tên zh_*/vi_* để chấm điểm)")
    ap.add_argument("--mode", choices=["dual", "vi-first"], default="dual")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--routing", action="store_true",
                    help="Chỉ kiểm chứng định tuyến, không đo từng model")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    wavs = collect_wavs(args.wav_dir)
    if not wavs:
        print("❌ Không tìm thấy file .wav nào.\n"
              "   → python scripts/download_stt.py  (lấy file mẫu)\n"
              "   → hoặc dùng --wav-dir <thư mục của bạn>")
        return 1
    print("Đo trên %d file .wav (%d file có nhãn ngôn ngữ)"
          % (len(wavs), sum(1 for _, t in wavs if t)))

    router = BilingualSttEngine(mode=args.mode, num_threads=args.threads)
    router.warmup()

    if not args.routing:
        rtf_zh = bench_single(router.zh, wavs, "SenseVoice (zh) — %d thread" % args.threads)
        rtf_vi = bench_single(router.vi, wavs, "Zipformer (vi) — %d thread" % args.threads)
        rtf_both = bench_single(router, wavs, "Song ngữ, mode=%s" % args.mode)
        print("\n%-34s RTF" % "")
        print("  %-32s %.3f" % ("SenseVoice (zh)", rtf_zh))
        print("  %-32s %.3f" % ("Zipformer (vi)", rtf_vi))
        print("  %-32s %.3f" % ("Song ngữ (mode=%s)" % args.mode, rtf_both))
        print("\nNgân sách: STT + dịch phải gọn hơn nhiều so với độ dài câu nói "
              "(2-4s), nếu không hàng đợi sẽ dồn và pipeline bỏ câu.")

    bench_routing(router, wavs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
