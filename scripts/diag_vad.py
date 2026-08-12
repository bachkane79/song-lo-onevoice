"""Đo VAD trên mic thật — chạy TRƯỚC khi chỉnh tham số Segmenter.

    python scripts/diag_vad.py

Đo hai giai đoạn và in ra bảng so sánh:

    [1] Phòng yên  — không phát gì, 8 giây
    [2] Đang nói   — phát một câu mẫu 5.6s qua loa

Ba con số quyết định cách chỉnh Segmenter:

* **VAD prob** — chênh lệch giữa hai giai đoạn càng lớn thì VAD càng đáng tin.
  Median 0.03 (yên) vs 0.89 (nói) là rất tốt; nếu hai số xích lại gần nhau thì
  phải nâng `--vad-threshold` hoặc xử lý tiếng ồn trước.

* **Năng lượng** — quyết định tầng Energy Valley có dùng được không. Tầng đó
  cắt câu khi `energy < valley_ratio * peak`, nên nó chỉ có nghĩa khi năng
  lượng lúc im **thấp hơn hẳn** lúc nói. Nếu hai số xấp xỉ nhau (AGC đã san
  phẳng dải động) thì không có `valley_ratio` nào chạy đúng — để tầng đó tắt.

* **Chuỗi im lặng dài nhất** ở giai đoạn [2] — đây là cận dưới của
  `--silence-timeout`. Đặt thấp hơn số này thì câu sẽ bị cắt giữa chừng.

Chạy trên máy khác / mic khác thì đo lại: các mặc định trong
`onevoice/config.py` chọn theo đúng bảng này trên mic Realtek của máy dev.
"""

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
QUIET_SECONDS = 8.0
FRAME_S = 512.0 / 16000.0


def _summarize(tag, probs, energies, speech):
    if not probs:
        print("%s  ❌ không nhận được frame nào" % tag)
        return
    p, e, s = np.array(probs), np.array(energies), np.array(speech)

    longest = run = 0
    for v in s:
        run = 0 if v else run + 1
        longest = max(longest, run)

    print("%s  (%d frame)" % (tag, len(p)))
    print("   VAD prob   : median %.3f | p90 %.3f | max %.3f"
          % (np.median(p), np.percentile(p, 90), p.max()))
    print("   Năng lượng : median %.5f | p90 %.5f | max %.5f"
          % (np.median(e), np.percentile(e, 90), e.max()))
    print("   is_speech  : %.1f%% số frame" % (100.0 * s.mean()))
    print("   Chuỗi im lặng dài nhất: %d frame = %.2fs"
          % (longest, longest * FRAME_S))
    return {"prob_med": float(np.median(p)), "energy_med": float(np.median(e)),
            "longest_silence_s": longest * FRAME_S}


def main():
    from VAD_process import SileroVADDetector

    probs, energies, speech = [], [], []

    def collect(chunk, is_speech, prob, volume):
        probs.append(prob)
        energies.append(float(np.sqrt(np.mean(chunk ** 2))))
        speech.append(bool(is_speech))

    from onevoice.mic import pick_input_device
    device_id, device_name = pick_input_device()
    print("🎙️ Mic: [%d] %s\n" % (device_id, device_name))

    vad = SileroVADDetector(threshold=0.35, device_id=device_id,
                            on_vad_frame=collect)
    threading.Thread(target=vad.start, daemon=True).start()
    time.sleep(2.0)                      # bỏ qua lúc luồng audio ổn định

    # ── [1] Phòng yên ────────────────────────────────────────────────────
    print("\n🤫 Giai đoạn 1: GIỮ YÊN LẶNG trong %.0f giây..." % QUIET_SECONDS)
    probs.clear(); energies.clear(); speech.clear()
    time.sleep(QUIET_SECONDS)
    quiet = (list(probs), list(energies), list(speech))

    # ── [2] Đang nói ─────────────────────────────────────────────────────
    print("🔊 Giai đoạn 2: đang phát câu mẫu qua loa...")
    import soundfile as sf
    import sounddevice as sd

    audio, sr = sf.read(SAMPLE_WAV, dtype="float32")
    audio = audio / max(1e-6, np.abs(audio).max()) * 0.95
    probs.clear(); energies.clear(); speech.clear()
    sd.play(audio, sr)
    sd.wait()
    loud = (list(probs), list(energies), list(speech))

    vad.is_running = False
    time.sleep(0.3)

    print("\n" + "=" * 62)
    q = _summarize("[1] PHÒNG YÊN", *quiet)
    print()
    l = _summarize("[2] ĐANG NÓI ", *loud)
    print("=" * 62)

    if not (q and l):
        return 1

    # ── Kết luận ─────────────────────────────────────────────────────────
    print("\n📌 Đọc kết quả:")
    print("   • --silence-timeout phải > %.2fs (chuỗi im dài nhất trong câu), "
          "hiện đặt 0.7s" % l["longest_silence_s"])

    gap = l["energy_med"] / max(q["energy_med"], 1e-6)
    if gap < 1.6:
        print("   • Năng lượng lúc nói chỉ gấp %.2f lần lúc im → Energy Valley "
              "KHÔNG dùng được\n     trên mic này (AGC đã san phẳng dải động). "
              "Để mặc định tắt, đừng bật --energy-valley." % gap)
    else:
        print("   • Năng lượng lúc nói gấp %.2f lần lúc im → có thể thử "
              "--energy-valley,\n     đặt --valley-ratio quanh %.2f."
              % (gap, 0.5 * q["energy_med"] / max(l["energy_med"], 1e-6)))

    if q["prob_med"] > 0.2:
        print("   • Phòng khá ồn (prob median lúc im %.3f) → cân nhắc nâng "
              "--vad-threshold" % q["prob_med"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
