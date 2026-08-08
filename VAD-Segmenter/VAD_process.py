import os
import sys
import math
import queue
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
    - Tự động thu âm Micro thời gian thực và resample về chuẩn 16kHz.
    - Tự động lọc kênh phần cứng và chuẩn hóa biên độ (Auto Gain Control).
    - Xuất tín hiệu nhận diện (chunk_agc, is_speech, prob, volume_agc) truyền cho Segmenter.
      chunk_agc: audio float32 đã chuẩn hóa AGC — biên độ nhất quán để Segmenter phân tích.
    """

    # ── Tham số Noise Gate + Smoothed AGC ────────────────────────────────
    _NOISE_FLOOR  = 0.005   # Biên độ thô (raw) dưới mức này = tiếng ồn nền → zeroed out
    _AGC_TARGET   = 0.30    # Biên độ mục tiêu sau AGC
    _AGC_MAX_GAIN = 20.0    # Gain tối đa (giảm từ 25 → 20 để hạn chế đẩy noise)
    _AGC_SMOOTH   = 0.08    # Hệ số làm mượt EMA (nhỏ hơn = mượt hơn, chậm hơn)

    def __init__(self, model_path=None, threshold=0.35, device_id=None, on_vad_frame=None):
        """
        :param model_path: Đường dẫn tới silero_vad.onnx
        :param threshold: Ngưỡng xác suất phát hiện giọng nói (mặc định 0.35)
        :param device_id: ID Micro (None = chọn mặc định hệ thống)
        :param on_vad_frame: Callback(chunk_agc, is_speech, prob, volume_agc) truyền sang Segmenter
                             chunk_agc    — numpy float32, 512 mẫu 16kHz, đã chuẩn hóa AGC
                             is_speech    — True/False
                             prob         — xác suất VAD (0.0–1.0)
                             volume_agc   — biên độ max sau AGC
        """
        self.target_sr = 16000
        self.chunk_size = 512
        self.context_size = 64
        self.threshold = threshold
        self.device_id = device_id
        self.on_vad_frame = on_vad_frame

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
        """Reset trạng thái RNN, bộ đệm Context (64 mẫu) và Smoothed AGC gain"""
        self.state   = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, self.context_size), dtype=np.float32)
        self._gain   = 1.0   # Giá trị gain hiện tại (EMA), reset về 1.0 khi khởi động lại

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

            # Lấy kênh âm thanh có tín hiệu mạnh nhất (tránh kênh im lặng)
            if indata.ndim > 1 and indata.shape[1] > 1:
                best_ch = np.argmax(np.max(np.abs(indata), axis=0))
                raw = indata[:, best_ch].flatten()
            else:
                raw = indata.flatten()
            raw = raw.astype(np.float32)

            # Resample về 16kHz
            if self.native_sr != self.target_sr:
                resampled = resample_poly(raw, self.resample_up, self.resample_down).astype(np.float32)
            else:
                resampled = raw

            combined = np.concatenate([audio_leftover, resampled]) if len(audio_leftover) > 0 else resampled

            idx = 0
            while idx + self.chunk_size <= len(combined):
                chunk_16k = combined[idx : idx + self.chunk_size]
                idx += self.chunk_size

                volume_raw = np.max(np.abs(chunk_16k))  # Biên độ thô từ mic

                # ── Noise Gate ────────────────────────────────────────────────
                # Khung âm thanh dưới ngưỡng noise floor → zeroed out (không khuếch đại tiếng ồn nền)
                if volume_raw < self._NOISE_FLOOR:
                    chunk_for_vad = np.zeros(self.chunk_size, dtype=np.float32)
                    target_gain   = 1.0
                    # Tranh thủ lúc im lặng kéo gain mượt về 1.0 qua EMA
                    self._gain    = (1 - self._AGC_SMOOTH) * self._gain + self._AGC_SMOOTH * target_gain

                # ── Smoothed AGC (Exponential Moving Average) ────────────────────
                # Gain thay đổi mượt dần (không nhảy đột ngột) → khắc phục tiếng rè tại ranh giới chunk
                elif volume_raw < self._AGC_TARGET:
                    target_gain   = min(self._AGC_MAX_GAIN, self._AGC_TARGET / volume_raw)
                    self._gain    = (1 - self._AGC_SMOOTH) * self._gain + self._AGC_SMOOTH * target_gain
                    chunk_for_vad = np.clip(chunk_16k * self._gain, -1.0, 1.0).astype(np.float32)
                else:
                    # Âm lượng đã đủ mạnh → gain mượt trở về 1.0
                    target_gain   = 1.0
                    self._gain    = (1 - self._AGC_SMOOTH) * self._gain + self._AGC_SMOOTH * target_gain
                    chunk_for_vad = np.clip(chunk_16k * self._gain, -1.0, 1.0).astype(np.float32)

                # Biên độ SAU AGC + Noise Gate — mức thực tế VAD & Segmenter nhìn thấy
                volume_agc = np.max(np.abs(chunk_for_vad))

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
                # ⚠️ QUAN TRỌNG: Truyền chunk_for_vad (đã AGC) chứ không phải chunk_16k raw!
                # → Segmenter nhận audio biên độ nhất quán → Energy Valley Detection chính xác
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

    def callback_segmenter(chunk_agc, is_speech, prob, volume_agc):
        """
        HÀM BÀN GIAO TÍN HIỆU CHO NGƯỜI LÀM SEGMENTER:
        - chunk_agc  : numpy float32, 512 mẫu (32ms) @ 16kHz — đã chuẩn hóa AGC
        - is_speech  : True (Có tiếng nói) / False (Khoảng lặng)
        - prob       : Xác suất tiếng nói từ Silero VAD (0.0 → 1.0)
        - volume_agc : Biên độ max sau AGC (~0.3 khi mic hoạt động bình thường)

        Người làm Segmenter nhận stream này để ngắt đoạn câu theo thuật toán của họ.
        """
        pass

    # Khởi tạo VAD Detector
    vad = SileroVADDetector(
        threshold=0.35,
        on_vad_frame=callback_segmenter
    )

    # Chạy VAD Detector (Ctrl+C để dừng)
    vad.start()