"""Điểm chạy chính của thiết bị OneVoice.

    Mic → VAD → Segmenter → STT → Translate → Display

    # Chạy thật với micro — đầy đủ VAD → Seg → STT → MT → Display
    python run_pipeline.py

    # Chạy STT + dịch trên file .wav có sẵn (không cần micro)
    python run_pipeline.py --wav a.wav b.wav

    # Chạy KHÔNG cần micro, KHÔNG cần model STT — chỉ kiểm tra tầng dịch
    python run_pipeline.py --offline

    # Khóa cứng ngôn ngữ (biết trước ai đang nói → nhanh hơn, chính xác hơn)
    python run_pipeline.py --force-lang zh

    # Xem danh sách micro
    python run_pipeline.py --list-devices
"""

import argparse
import os
import sys

from onevoice.config import PipelineConfig
from onevoice.display import CompositeDisplay, ConsoleDisplay, JsonlDisplay
from onevoice.pipeline import OneVoicePipeline
from onevoice.stt import create_stt_engine

BANNER = """
======================================================================
  🌏  OneVoice — Thiết bị phiên dịch song ngữ Việt ⇄ Hoa (offline)
======================================================================
  Mic → VAD → Segmenter → STT → Translate → Display
  Toàn bộ chạy local, không cần mạng.
======================================================================
"""


def list_devices():
    import sounddevice as sd
    print("Các thiết bị đầu vào:")
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            print("  [%2d] %-45s %d kênh @ %.0f Hz"
                  % (idx, dev["name"][:45], dev["max_input_channels"],
                     dev["default_samplerate"]))
    return 0


def build_args():
    ap = argparse.ArgumentParser(
        description="OneVoice — phiên dịch song ngữ Việt ⇄ Hoa offline",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)

    ap.add_argument("--list-devices", action="store_true",
                    help="Liệt kê micro rồi thoát")
    ap.add_argument("--wav", nargs="+", metavar="FILE",
                    help="Chạy STT + dịch trên file .wav thay vì micro "
                         "(bỏ qua VAD/Segmenter — mỗi file coi như một câu)")
    ap.add_argument("--offline", action="store_true",
                    help="Không micro, không model STT — chạy câu mẫu để kiểm tra "
                         "tầng dịch + hiển thị")

    g = ap.add_argument_group("STT")
    g.add_argument("--stt", default=None,
                   help="sherpa (mặc định) | sensevoice | zipformer-vi | mock | scripted")
    g.add_argument("--stt-mode", choices=["dual", "vi-first"], default=None,
                   help="dual = chạy cả 2 model mỗi câu (chính xác nhất); "
                        "vi-first = bỏ qua SenseVoice khi đã chắc là tiếng Việt")
    g.add_argument("--stt-threads", type=int, default=None)
    g.add_argument("--stt-script", help="File câu mẫu cho --stt scripted")
    g.add_argument("--stt-latency-ms", type=float, default=0.0,
                   help="Giả lập độ trễ STT để thử tình huống quá tải")
    g.add_argument("--show-route", action="store_true",
                   help="In thêm điểm định tuyến zh/vi của từng câu")

    g = ap.add_argument_group("Dịch")
    g.add_argument("--force-lang", choices=["zh", "vi"], default=None,
                   help="Khóa cứng ngôn ngữ nguồn (mặc định: tự nhận diện)")
    g.add_argument("--beam", type=int, default=None,
                   help="Beam size (mặc định 4; hạ về 1 nếu Jetson chậm)")
    g.add_argument("--threads", type=int, default=None,
                   help="Số thread CPU cho máy dịch (mặc định 4)")
    g.add_argument("--no-glossary", action="store_true",
                   help="Tắt bảng địa danh Hán-Việt (để so sánh)")
    g.add_argument("--no-zh-unk-map", action="store_true",
                   help="Tắt bảng chuẩn hóa <unk> tiếng Hoa (để so sánh)")

    g = ap.add_argument_group("Âm thanh")
    g.add_argument("--device", type=int, default=None, help="ID micro")
    g.add_argument("--vad-threshold", type=float, default=None)
    g.add_argument("--save-wav", action="store_true",
                   help="Lưu từng câu ra output_segments/ (chỉ để debug)")

    g = ap.add_argument_group("Hiển thị")
    g.add_argument("--log-jsonl", metavar="FILE",
                   help="Ghi mỗi câu thành 1 dòng JSON để phân tích sau")
    g.add_argument("--no-timing", action="store_true",
                   help="Ẩn thông tin độ trễ trên màn hình")
    return ap


def main():
    args = build_args().parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.list_devices:
        return list_devices()

    cfg = PipelineConfig()
    # `--offline` không có audio nên engine STT thật vô nghĩa → dùng câu mẫu.
    default_stt = "mock" if args.offline else cfg.stt_engine
    cfg.stt_engine = args.stt or default_stt
    if args.offline and cfg.stt_engine in ("sherpa", "sensevoice", "zipformer-vi"):
        print("⚠️  --offline không có audio thật → chuyển sang --stt mock\n")
        cfg.stt_engine = "mock"
    if args.stt_mode:
        cfg.stt_router_mode = args.stt_mode
    if args.stt_threads:
        cfg.stt_threads = args.stt_threads
    if args.force_lang:
        cfg.force_src_lang = args.force_lang
    if args.device is not None:
        cfg.mic_device_id = args.device
    if args.vad_threshold is not None:
        cfg.vad_threshold = args.vad_threshold
    cfg.save_wav_debug = args.save_wav
    if args.beam is not None:
        cfg.translate.beam_size = args.beam
    if args.threads is not None:
        cfg.translate.intra_threads = args.threads
    if args.no_glossary:
        cfg.translate.use_glossary = False
    if args.no_zh_unk_map:
        cfg.translate.use_zh_unk_map = False

    stt_kwargs = {
        "fake_latency_ms": args.stt_latency_ms,
        "mode": cfg.stt_router_mode,
        "force_lang": cfg.force_src_lang,
        "num_threads": cfg.stt_threads,
    }
    if cfg.stt_engine == "scripted":
        if not args.stt_script:
            print("❌ --stt scripted cần kèm --stt-script FILE")
            return 1
        stt_kwargs["path"] = args.stt_script

    print(BANNER)
    try:
        stt = create_stt_engine(cfg.stt_engine, **stt_kwargs)
    except (FileNotFoundError, ImportError) as exc:
        print("❌ %s" % exc)
        print("\n→ Tải model STT : python scripts/download_stt.py")
        print("→ Chạy tạm không cần STT: python run_pipeline.py --offline")
        return 1

    display = ConsoleDisplay(show_timing=not args.no_timing,
                             show_route=args.show_route)
    if args.log_jsonl:
        display = CompositeDisplay(display, JsonlDisplay(args.log_jsonl))

    pipeline = OneVoicePipeline(config=cfg, stt_engine=stt, display=display)

    if args.wav:
        return run_wav_files(pipeline, display, args.wav)

    if args.offline:
        # Không có micro: bơm thẳng N câu rỗng, engine STT tự trả câu của nó.
        n = len(getattr(stt, "script", [])) or 8
        display.show_status("⚙️  Chế độ offline — bỏ qua micro, chạy %d câu mẫu\n" % n)
        pipeline.run_offline([(None, 16000, 2.0)] * n)
        return 0

    try:
        pipeline.run()
    except ImportError as exc:
        print("❌ Thiếu thư viện âm thanh: %s" % exc)
        print("   → pip install -r requirements.txt")
        print("   → Hoặc chạy không cần micro: python run_pipeline.py --offline")
        return 1
    except KeyboardInterrupt:
        pipeline.stop()
    return 0


def run_wav_files(pipeline, display, paths):
    """Chạy STT + dịch trên các file .wav có sẵn.

    Bỏ qua VAD/Segmenter: mỗi file được coi là **một câu đã cắt sẵn**. Dùng để
    kiểm tra STT + máy dịch bằng audio thật khi chưa có micro, hoặc để tái hiện
    lại một câu bị dịch sai (bật `--save-wav` khi chạy thật để có file).
    """
    import numpy as np

    try:
        import soundfile as sf
    except ImportError:
        sf = None

    utterances = []
    for path in paths:
        if not os.path.isfile(path):
            print("❌ Không tìm thấy %s" % path)
            return 1
        if sf is not None:
            audio, sr = sf.read(path, dtype="float32")
        else:
            audio, sr = _read_wav_stdlib(path)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio[:, 0]
        if sr != 16000:
            from scipy.signal import resample_poly
            import math
            g = math.gcd(16000, sr)
            audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
            sr = 16000
        utterances.append((audio, sr, len(audio) / float(sr)))
        print("   📂 %s  (%.1fs)" % (os.path.basename(path), utterances[-1][2]))

    display.show_status("\n⚙️  Chế độ --wav: bỏ qua VAD/Segmenter, "
                        "mỗi file = một câu\n")
    pipeline.run_offline(utterances)
    return 0


def _read_wav_stdlib(path):
    """Đọc .wav 16-bit bằng thư viện chuẩn khi không có soundfile."""
    import wave

    import numpy as np

    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("%s: chỉ hỗ trợ WAV 16-bit khi thiếu soundfile" % path)
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        audio = data.astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            audio = audio.reshape(-1, w.getnchannels())[:, 0]
        return audio, w.getframerate()


if __name__ == "__main__":
    sys.exit(main())
