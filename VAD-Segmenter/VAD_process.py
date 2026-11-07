import os
import sys
import math
import queue
import argparse
import numpy as np
import onnxruntime as ort
import sounddevice as sd
from scipy.signal import resample_poly

# Cài đặt UTF-8 cho Windows Terminal Output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ==============================================================================
# MODULE VAD (VOICE ACTIVITY DETECTION - NHẬN DIỆN GIỌNG NÓI VIỆT & HOA)
# ==============================================================================
class SileroVADDetector:
    """
    Module VAD: Nhận diện giọng nói đa ngôn ngữ (Tiếng Việt & Tiếng Hoa).
    - Silero VAD ONNX là mô hình Language-Agnostic (không phụ thuộc ngôn ngữ),
      được huấn luyện trên >60 ngôn ngữ bao gồm Tiếng Việt (Vietnamese) và Tiếng Hoa (Mandarin/Cantonese).
    - Tự động thu âm Micro thời gian thực và resample về chuẩn 16kHz (chống méo biên bằng padtype='line').
    - Tự động khóa kênh phần cứng ổn định và chuẩn hóa biên độ mượt theo từng mẫu (Sample-Level Linear AGC).
    - Hỗ trợ 3 giai đoạn cải tiến Sweetspot (stage=1, 2, 3):
        + Stage 1 & 2: Ngưỡng đơn chuẩn + Smoothed AGC.
        + Stage 3    : Context-Holding Hysteresis (Ngưỡng duy trì thấp + Speech Hangover
                       cầu nối qua các âm tiết nhỏ giữa câu, ngăn chặn ngắt câu sớm).
    - Xuất tín hiệu nhận diện (chunk_agc, is_speech, prob, volume_agc) truyền cho Segmenter.
    """

    # ── Tham số Noise Gate + Smoothed AGC ────────────────────────────────
    _NOISE_FLOOR  = 0.005   # Biên độ thô (raw) dưới mức này = tiếng ồn nền (không khuếch đại)
    _AGC_TARGET   = 0.30    # Biên độ mục tiêu sau AGC
    _AGC_MAX_GAIN = 20.0    # Gain tối đa (hạn chế đẩy noise)
    _AGC_SMOOTH   = 0.08    # Hệ số làm mượt EMA (nhỏ hơn = mượt hơn, chậm hơn)

    # [Giai đoạn 3] Số frame giữ trạng thái nói (Hangover: 3 frames = 96ms) khi giọng hạ thấp giữa câu
    _HANGOVER_FRAMES = 3

    def __init__(self, model_path=None, threshold=0.35, device_id=None, on_vad_frame=None, stage=None):
        """
        :param model_path: Đường dẫn tới silero_vad.onnx
        :param threshold: Ngưỡng xác suất phát hiện giọng nói cơ sở (mặc định 0.35)
        :param device_id: ID Micro (None = chọn mặc định hệ thống)
        :param on_vad_frame: Callback(chunk_agc, is_speech, prob, volume_agc) truyền sang Segmenter
        :param stage: Giai đoạn cải tiến (1, 2, hoặc 3). Mặc định đọc từ biến môi trường
                      ONEVOICE_VAD_STAGE (mặc định = 3).
        """
        self.target_sr = 16000
        self.chunk_size = 512
        self.context_size = 64
        self.threshold = threshold
        self.device_id = device_id
        self.on_vad_frame = on_vad_frame

        if stage is None:
            try:
                stage = int(os.environ.get("ONEVOICE_VAD_STAGE", "3"))
            except ValueError:
                stage = 3
        self.stage = int(stage)

        # [Giai đoạn 3] Ngưỡng kép bảo vệ mạch nói (Context-Holding Hysteresis):
        # - Bắt đầu/vào lại câu ngay lập tức khi prob >= start_threshold (bằng threshold gốc 0.35).
        # - Khi đang trong câu nói, giữ trạng thái nói tới tận end_threshold (0.22) kèm Hangover 3 frame (96ms)
        #   để khi người nói hạ giọng hoặc ngập ngừng nhẹ giữa câu không bị rớt trạng thái is_speech!
        self.start_threshold = self.threshold
        self.end_threshold   = max(0.15, self.threshold - 0.13)   # Vd: 0.22 khi threshold=0.35

        # Tự động tìm đường dẫn file silero_vad.onnx
        if model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.path.join(script_dir, "models", "silero_vad.onnx"),
                os.path.join(script_dir, "silero_vad.onnx"),
                "models/silero_vad.onnx",
                "silero_vad.onnx"
            ]
            for c in candidates:
                if os.path.exists(c):
                    model_path = c
                    break
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError("❌ Không tìm thấy file silero_vad.onnx trong thư mục models/!")

        self.model_path = model_path
        print(f"📦 Đã nạp mô hình Silero VAD (Hỗ trợ Việt - Hoa): {self.model_path}")

        self.session = ort.InferenceSession(self.model_path)
        self.sr_tensor = np.array(self.target_sr, dtype=np.int64)

        self._init_device()
        # maxsize=100 tương đương ~3.2s buffer, tránh tràn bộ nhớ khi ONNX chậm
        self.audio_queue = queue.Queue(maxsize=100)
        self.is_running = False
        self.reset_state()

    def _init_device(self):
        """Cấu hình Microphone phần cứng"""
        if self.device_id is None:
            default_dev = sd.default.device[0]
            if default_dev is not None and default_dev >= 0:
                self.device_id = default_dev
            else:
                for idx, dev in enumerate(sd.query_devices()):
                    if dev['max_input_channels'] > 0:
                        self.device_id = idx
                        break

        dev_info = sd.query_devices(self.device_id, 'input')
        self.native_sr = int(dev_info['default_samplerate'])
        self.max_channels = dev_info['max_input_channels']
        dev_name = dev_info['name']
        print(f"🎙️ Đã kết nối Micro [{self.device_id}]: {dev_name} ({self.max_channels} kênh)")
        print(f"🎛️ Tần số lấy mẫu phần cứng: {self.native_sr}Hz -> Resample chuẩn {self.target_sr}Hz")

        gcd = math.gcd(self.target_sr, self.native_sr)
        self.resample_up = self.target_sr // gcd
        self.resample_down = self.native_sr // gcd
        self.block_size = int(self.chunk_size * (self.native_sr / self.target_sr))

    def reset_state(self):
        """Reset trạng thái RNN, bộ đệm Context (64 mẫu), Smoothed AGC và Hysteresis"""
        self.state   = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, self.context_size), dtype=np.float32)
        self._gain   = 1.0   # Giá trị gain hiện tại (EMA), reset về 1.0 khi khởi động lại
        self._vad_triggered  = False
        self._hangover_count = 0
        self._active_ch      = 0

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"⚠️ Cảnh báo luồng âm thanh: {status}", file=sys.stderr)
        # Không block nếu queue đầy — bỏ qua frame cũ thay vì để latency tăng
        try:
            self.audio_queue.put_nowait(indata.copy())
        except queue.Full:
            pass

    def _process_audio_queue(self):
        """Vòng lặp xử lý âm thanh từ Queue"""
        audio_leftover = np.array([], dtype=np.float32)

        while self.is_running:
            try:
                indata = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Chọn kênh âm thanh ổn định (chỉ đổi kênh nếu kênh khác mạnh hơn rõ rệt > 1.5x,
            # tránh nhảy qua lại giữa 4 capsule micro mỗi 32ms gây lệch pha và tiếng rè)
            if indata.ndim > 1 and indata.shape[1] > 1:
                ch_peaks = np.max(np.abs(indata), axis=0)
                best_ch  = int(np.argmax(ch_peaks))
                curr_ch  = self._active_ch if self._active_ch < indata.shape[1] else 0
                if ch_peaks[best_ch] > ch_peaks[curr_ch] * 1.5 and ch_peaks[best_ch] > self._NOISE_FLOOR:
                    self._active_ch = best_ch
                raw = indata[:, self._active_ch].flatten()
            else:
                raw = indata.flatten()
            raw = raw.astype(np.float32)

            # Resample về 16kHz (dùng padtype='line' để triệt tiêu gai nhiễu 38% ở biên mỗi block 32ms)
            if self.native_sr != self.target_sr:
                resampled = resample_poly(
                    raw, self.resample_up, self.resample_down, padtype='line'
                ).astype(np.float32)
            else:
                resampled = raw

            combined = np.concatenate([audio_leftover, resampled]) if len(audio_leftover) > 0 else resampled

            idx = 0
            while idx + self.chunk_size <= len(combined):
                chunk_16k = combined[idx : idx + self.chunk_size]
                idx += self.chunk_size

                volume_raw = float(np.max(np.abs(chunk_16k)))  # Biên độ thô từ mic
                prev_gain  = self._gain

                # ── Soft Noise Gate + Smoothed AGC ───────────────────────────────
                # Khi dưới ngưỡng ồn nền (_NOISE_FLOOR): KHÔNG cắt cụt về np.zeros (tránh tạo xung
                # vuông lẹt xẹt 31.25Hz khi biên độ dao động quanh ngưỡng), mà kéo mượt gain về 1.0
                if volume_raw < self._NOISE_FLOOR:
                    target_gain = 1.0
                elif volume_raw < self._AGC_TARGET:
                    target_gain = min(self._AGC_MAX_GAIN, self._AGC_TARGET / volume_raw)
                else:
                    target_gain = 1.0

                self._gain = (1 - self._AGC_SMOOTH) * self._gain + self._AGC_SMOOTH * target_gain

                # Nội suy tuyến tính gain trên từng mẫu (Sample-by-Sample Linear Ramp từ prev_gain -> self._gain)
                # giúp sóng âm liên tục tuyệt đối qua ranh giới chunk, triệt tiêu hoàn toàn tiếng rè bậc thang!
                gain_curve    = np.linspace(prev_gain, self._gain, self.chunk_size, dtype=np.float32)
                chunk_for_vad = np.clip(chunk_16k * gain_curve, -1.0, 1.0).astype(np.float32)

                # Biên độ SAU AGC — mức thực tế VAD & Segmenter nhìn thấy
                volume_agc = float(np.max(np.abs(chunk_for_vad)))

                # Ghép Context 64 samples + Chunk 512 samples = Shape (1, 576)
                input_tensor = np.concatenate([self.context, chunk_for_vad.reshape(1, 512)], axis=1).astype(np.float32)

                inputs = {
                    "input": input_tensor,
                    "sr": self.sr_tensor,
                    "state": self.state
                }
                outputs = self.session.run(None, inputs)
                prob = float(outputs[0][0][0])
                self.state = outputs[1]                      # Cập nhật trạng thái RNN
                self.context = input_tensor[:, -self.context_size:]  # Cập nhật Context 64 mẫu

                # ── Xác định trạng thái tiếng nói theo Giai đoạn ────────────────
                if self.stage >= 3:
                    # [Giai đoạn 3] Context-Holding Hysteresis + Speech Hangover:
                    # - Bật ngay lập tức khi prob >= start_threshold (0.35) → không trễ hay bỏ sót chữ.
                    # - Khi đã đang nói: giữ trạng thái nói miễn là prob >= end_threshold (0.22).
                    # - Nếu prob tụt dưới 0.22 (do hạ giọng/lướt âm giữa câu), duy trì thêm _HANGOVER_FRAMES
                    #   (3 frames = 96ms) trước khi thực sự chuyển sang im lặng → chống ngắt câu sớm!
                    if not self._vad_triggered:
                        if prob >= self.start_threshold:
                            self._vad_triggered  = True
                            self._hangover_count = self._HANGOVER_FRAMES
                    else:
                        if prob >= self.end_threshold:
                            self._hangover_count = self._HANGOVER_FRAMES
                        else:
                            if self._hangover_count > 0:
                                self._hangover_count -= 1
                            else:
                                self._vad_triggered = False

                    is_speech = self._vad_triggered
                else:
                    is_speech = prob > self.threshold

                # Visualizer — hiển thị Vol sau AGC (phản ánh đúng tín hiệu thực tế)
                bar_len = int(prob * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                if is_speech:
                    sys.stdout.write(f"\r🗣️  VAD: ĐANG NÓI    [{bar}] Prob: {prob:.2f} | Vol(agc): {volume_agc:.4f}  ")
                else:
                    sys.stdout.write(f"\r🤫 VAD: KHOẢNG LẶNG  [{bar}] Prob: {prob:.2f} | Vol(agc): {volume_agc:.4f}  ")
                sys.stdout.flush()

                # Bàn giao frame AGC-normalized (512 mẫu, 16kHz) sang Segmenter
                if self.on_vad_frame:
                    self.on_vad_frame(chunk_for_vad, is_speech, prob, volume_agc)

            audio_leftover = combined[idx:]

    def start(self):
        """Khởi chạy luồng Microphone + VAD Detector"""
        self.is_running = True
        self.reset_state()

        print("\n🚀 Đang chạy VAD Engine (Nhận diện giọng nói Việt & Hoa)...")
        print("Hãy nói Tiếng Việt hoặc Tiếng Hoa vào Micro để kiểm tra! (Ctrl+C để dừng)")
        print("----------------------------------------------------------------------------")

        try:
            with sd.InputStream(
                device=self.device_id,
                samplerate=self.native_sr,
                channels=self.max_channels,
                dtype="float32",
                blocksize=self.block_size,
                callback=self._audio_callback
            ):
                self._process_audio_queue()
        except KeyboardInterrupt:
            print("\n\n🛑 Đã dừng VAD Detector thành công.")
        except Exception as e:
            print(f"\n\n❌ Lỗi Luồng âm thanh: {e}")
        finally:
            self.is_running = False

# ==============================================================================
# HƯỚNG DẪN BÀN GIAO KẾT NỐI VỚI NGƯỜI LÀM SEGMENTER
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy thử Silero VAD Detector")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=3,
                        help="Giai đoạn cải tiến VAD (1, 2, hoặc 3 - mặc định 3)")
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="Ngưỡng VAD cơ sở (mặc định 0.35)")
    parser.add_argument("--device", type=int, default=None,
                        help="ID thiết bị Micro")
    args = parser.parse_args()

    def callback_segmenter(chunk_agc, is_speech, prob, volume_agc):
        pass

    vad = SileroVADDetector(
        threshold=args.threshold,
        device_id=args.device,
        on_vad_frame=callback_segmenter,
        stage=args.stage,
    )
    vad.start()
