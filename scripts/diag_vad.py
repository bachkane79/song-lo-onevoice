"""Đo VAD trên mic thật — chạy TRƯỚC khi chỉnh tham số Segmenter.

    python scripts/diag_vad.py              # Phát câu mẫu qua loa ở Giai đoạn 2 (mặc định)
    python scripts/diag_vad.py --mic-only   # Bạn tự nói vào micro ở Giai đoạn 2 (dùng khi đeo tai nghe)

Đo hai giai đoạn và in ra bảng so sánh:

    [1] Phòng yên  — giữ im lặng hoàn toàn (không nói gì)
    [2] Đang nói   — phát câu mẫu 5.6s ra loa ngoài (hoặc bạn tự nói nếu dùng --mic-only)
"""

import argparse
import os
import sys
import threading
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "VAD-Segmenter"))

SAMPLE_WAV = os.path.join(_REPO, "models", "stt", "sensevoice-zh",
                          "test_wavs", "zh.wav")
QUIET_SECONDS = 6.0
SPEAK_SECONDS = 6.0
FRAME_S = 512.0 / 16000.0
_REAL_STDOUT = sys.stdout


def _say(msg):
    _REAL_STDOUT.write(msg + "\n")
    _REAL_STDOUT.flush()


def _countdown(label, seconds, probs):
    for rem in range(int(seconds), 0, -1):
        cur_p = probs[-1] if probs else 0.0
        _REAL_STDOUT.write(f"\r   ⏳ {label}: còn {rem}s  (VAD prob tức thời: {cur_p:.3f})   ")
        _REAL_STDOUT.flush()
        time.sleep(1.0)
    _REAL_STDOUT.write("\r" + " " * 72 + "\r")
    _REAL_STDOUT.flush()


def _summarize(tag, probs, energies, speech):
    if not probs:
        _say("%s  ❌ không nhận được frame nào" % tag)
        return
    p, e, s = np.array(probs), np.array(energies), np.array(speech)

    longest = run = 0
    for v in s:
        run = 0 if v else run + 1
        longest = max(longest, run)

    # Để đo chuỗi im lặng BÊN TRONG câu nói (không tính khoảng im trước khi bắt đầu nói
    # và sau khi đã nói xong), chỉ cắt từ frame có tiếng nói đầu tiên tới frame có tiếng nói cuối cùng
    speech_idx = np.where(s)[0]
    if len(speech_idx) >= 2:
        inner_s = s[speech_idx[0]:speech_idx[-1] + 1]
        inner_longest = inner_run = 0
        for v in inner_s:
            inner_run = 0 if v else inner_run + 1
            inner_longest = max(inner_longest, inner_run)
    else:
        inner_longest = longest

    _say("%s  (%d frame)" % (tag, len(p)))
    _say("   VAD prob   : median %.3f | p90 %.3f | max %.3f"
         % (np.median(p), np.percentile(p, 90), p.max()))
    _say("   Năng lượng : median %.5f | p90 %.5f | max %.5f"
         % (np.median(e), np.percentile(e, 90), e.max()))
    _say("   is_speech  : %.1f%% số frame" % (100.0 * s.mean()))
    _say("   Chuỗi im dài nhất (tổng / giữa câu): %.2fs / %.2fs"
         % (longest * FRAME_S, inner_longest * FRAME_S))
    return {
        "prob_med": float(np.median(p)),
        "energy_med": float(np.median(e)),
        "longest_silence_s": inner_longest * FRAME_S,
    }


def main():
    global _REAL_STDOUT
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _REAL_STDOUT = sys.stdout

    ap = argparse.ArgumentParser(description="Đo VAD thực tế trên Micro")
    ap.add_argument("--mic-only", action="store_true",
                    help="Bạn tự nói vào micro ở Giai đoạn 2 thay vì phát file mẫu qua loa "
                         "(bắt buộc dùng nếu bạn đang đeo tai nghe Headset)")
    ap.add_argument("--device", type=int, default=None, help="ID thiết bị Micro")
    args = ap.parse_args()

    from VAD_process import SileroVADDetector
    from onevoice.mic import pick_input_device
    from onevoice.vadlog import muted_stdout

    probs, energies, speech = [], [], []

    def collect(chunk, is_speech, prob, volume):
        probs.append(prob)
        energies.append(float(np.sqrt(np.mean(chunk ** 2))))
        speech.append(bool(is_speech))

    device_id, device_name = pick_input_device(args.device)
    _say("🎙️ Mic: [%d] %s" % (device_id, device_name))

    with muted_stdout(enabled=True):
        vad = SileroVADDetector(threshold=0.35, device_id=device_id,
                                on_vad_frame=collect)
        threading.Thread(target=vad.start, daemon=True).start()
        time.sleep(2.0)                  # bỏ qua lúc luồng audio ổn định

        # ── [1] Phòng yên ────────────────────────────────────────────────
        _say("\n[Giai đoạn 1/2] GIỮ YÊN LẶNG HOÀN TOÀN trong %.0f giây (Đừng nói gì)..."
             % QUIET_SECONDS)
        probs.clear(); energies.clear(); speech.clear()
        _countdown("Đang đo phòng yên", QUIET_SECONDS, probs)
        quiet = (list(probs), list(energies), list(speech))
        _say("   ✓ Đã đo xong Giai đoạn 1 (Phòng yên).")

        # ── [2] Đang nói ─────────────────────────────────────────────────
        if args.mic_only:
            _say("\n[Giai đoạn 2/2] BÂY GIỜ BẠN HÃY NÓI LIÊN TỤC VÀO MICRO trong %.0f giây!"
                 % SPEAK_SECONDS)
            probs.clear(); energies.clear(); speech.clear()
            _countdown("Đang thu giọng nói của bạn", SPEAK_SECONDS, probs)
        else:
            _say("\n[Giai đoạn 2/2] Đang phát câu mẫu tiếng Trung (zh.wav) qua loa...")
            import soundfile as sf
            import sounddevice as sd

            audio, sr = sf.read(SAMPLE_WAV, dtype="float32")
            audio = audio / max(1e-6, np.abs(audio).max()) * 0.95
            probs.clear(); energies.clear(); speech.clear()
            sd.play(audio, sr)
            sd.wait()

        loud = (list(probs), list(energies), list(speech))
        _say("   ✓ Đã đo xong Giai đoạn 2 (Đang nói).")

        vad.is_running = False
        time.sleep(0.3)

    _say("\n" + "=" * 62)
    q = _summarize("[1] PHÒNG YÊN", *quiet)
    _say("")
    l = _summarize("[2] ĐANG NÓI ", *loud)
    _say("=" * 62)

    if not (q and l):
        return 1

    # ── Kết luận ─────────────────────────────────────────────────────────
    _say("\n📌 Đọc kết quả:")
    _say("   • Khoảng nghỉ dài nhất GIỮA CÂU khi nói: %.2fs (cận dưới của silence_timeout)"
         % l["longest_silence_s"])

    gap = l["energy_med"] / max(q["energy_med"], 1e-6)
    _say("   • Tỉ lệ Năng lượng (Nói / Im): gấp %.1f lần (%.5f vs %.5f)"
         % (gap, l["energy_med"], q["energy_med"]))
    _say("   • Tỉ lệ Xác suất VAD (Nói / Im): %.3f (nói) vs %.3f (im)"
         % (l["prob_med"], q["prob_med"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
