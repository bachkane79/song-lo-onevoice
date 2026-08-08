import os
import sys
import numpy as np
from collections import deque
from scipy.io.wavfile import write

from VAD_process import SileroVADDetector

# Cài đặt UTF-8 cho Windows Terminal Output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


class AudioSegmenter:
    """
    Module Segmenter: Phân đoạn câu nói từ stream VAD bằng Energy Valley Detection.

    Chiến lược cắt theo 3 lớp ưu tiên:
      [1] Silence Timeout  : VAD phát hiện im lặng hoàn toàn >= 0.6s → cắt câu (chuẩn nhất)
      [2] Energy Valley    : Phát hiện khoảng hở năng lượng tự nhiên (~80-150ms) trong
                             dòng nói liên tục → cắt tại điểm tự nhiên nhất về mặt ngôn ngữ.
                             Chỉ áp dụng khi câu đã đủ dài (>= min_segment_duration).
      [3] Max Duration     : Nếu không tìm được valley, ép cắt cứng tại max_segment_duration
                             để tránh quá tải STT/Trans Machine phía sau.
    """

    FRAME_SIZE = 512        # samples (32ms at 16kHz)
    FRAME_DURATION = FRAME_SIZE / 16000  # 0.032s

    def __init__(
        self,
        sample_rate: int          = 16000,
        silence_timeout: float    = 0.6,    # [Lớp 1] VAD im lặng 0.6s → cắt câu
        min_segment_duration: float = 1.2,  # Không cắt trước khi đủ 1.2s
        max_segment_duration: float = 8.0,  # [Lớp 3] Ép cắt nếu vượt 8.0s
        valley_ratio: float       = 0.25,   # [Lớp 2] Năng lượng < 25% peak = valley
        valley_min_frames: int    = 3,      # Cần 3 frame liên tiếp thấp (~96ms)
        energy_window_frames: int = 40,     # Cửa sổ tính peak năng lượng (~1.28s)
        pre_speech_frames: int    = 5,      # Giữ 5 frame (~160ms) trước câu nói
        on_segment_ready          = None,   # Callback(audio, sample_rate, duration_sec) → STT nhận qua RAM
        save_wav_debug: bool      = True,   # True = lưu file .wav để nghe kiểm tra; False = chỉ RAM
        output_dir: str           = "output_segments",
    ):
        self.sample_rate          = sample_rate
        self.silence_timeout      = silence_timeout
        self.min_segment_duration = min_segment_duration
        self.max_segment_duration = max_segment_duration
        self.valley_ratio         = valley_ratio
        self.valley_min_frames    = valley_min_frames
        self.pre_speech_frames    = pre_speech_frames
        self.on_segment_ready     = on_segment_ready   # Hook cho STT — nhận audio float32 qua RAM
        self.save_wav_debug       = save_wav_debug     # Tắt khi nhúng lên thiết bị
        self.output_dir           = output_dir

        self.silence_timeout_frames = int(silence_timeout    / self.FRAME_DURATION)
        self.min_frames             = int(min_segment_duration / self.FRAME_DURATION)
        self.max_frames             = int(max_segment_duration / self.FRAME_DURATION)

        # Cửa sổ trượt năng lượng để tính đỉnh (peak) gần nhất
        self._energy_window = deque(maxlen=energy_window_frames)

        if self.save_wav_debug:
            os.makedirs(self.output_dir, exist_ok=True)
        self.segment_count = 0
        self._reset()

    # ──────────────────────────────────────────────────────────────────────────
    def _reset(self):
        """Reset trạng thái bộ đệm về ban đầu cho câu mới"""
        self.is_speaking     = False
        self.speech_buffer   = []          # Danh sách chunk_agc của câu hiện tại
        self.speech_frames   = 0           # Số frame kể từ khi bắt đầu nói
        self.silence_frames  = 0           # Số frame VAD im lặng liên tiếp
        self.valley_frames   = 0           # Số frame năng lượng thấp liên tiếp (valley)
        self.best_valley_idx = None        # Vị trí frame valley tốt nhất để cắt
        self.pre_ring        = deque(maxlen=self.pre_speech_frames)

    # ──────────────────────────────────────────────────────────────────────────
    def _frame_energy(self, chunk: np.ndarray) -> float:
        """Tính năng lượng RMS của một frame audio"""
        return float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0

    def _peak_energy(self) -> float:
        """Đỉnh năng lượng trong cửa sổ gần nhất"""
        return max(self._energy_window) if self._energy_window else 0.0

    # ──────────────────────────────────────────────────────────────────────────
    def process_vad_frame(self, chunk_agc: np.ndarray, is_speech: bool, prob: float, volume: float):
        """
        Callback chính — nhận từng frame 512 samples từ SileroVADDetector.

        Parameters
        ----------
        chunk_agc : np.ndarray  — audio float32 16kHz 512 mẫu đã AGC-normalized
        is_speech : bool        — True nếu VAD phát hiện tiếng nói
        prob      : float       — xác suất VAD (0.0–1.0)
        volume    : float       — biên độ max sau AGC
        """
        energy = self._frame_energy(chunk_agc)
        self._energy_window.append(energy)

        # ── Cập nhật pre-speech ring buffer ───────────────────────────────────
        if not self.is_speaking:
            self.pre_ring.append(chunk_agc)

        # ─────────────────────────────────────────────────────────────────────
        # [NHÁNH A] VAD phát hiện TIẾNG NÓI
        # ─────────────────────────────────────────────────────────────────────
        if is_speech:
            self.silence_frames = 0

            if not self.is_speaking:
                # Bắt đầu câu mới
                self.is_speaking = True
                print(f"\n🟢 [Segmenter] Bắt đầu câu mới — VAD prob={prob:.3f}")
                # Đưa pre-speech vào đầu câu để không mất phụ âm đầu
                self.speech_buffer.extend(self.pre_ring)
                self.speech_frames = len(self.pre_ring)

            self.speech_buffer.append(chunk_agc)
            self.speech_frames += 1

            # ── [Lớp 2] Energy Valley Detection ──────────────────────────────
            # Chỉ bắt đầu tìm valley khi câu đã đủ dài
            if self.speech_frames >= self.min_frames:
                peak = self._peak_energy()
                if peak > 1e-5 and energy < self.valley_ratio * peak:
                    # Đây là frame năng lượng thấp — ứng viên valley
                    self.valley_frames += 1
                    if self.best_valley_idx is None:
                        # Ghi nhớ vị trí bắt đầu của valley
                        self.best_valley_idx = len(self.speech_buffer) - 1
                else:
                    # Năng lượng phục hồi — kiểm tra xem valley vừa rồi có đủ không
                    if self.valley_frames >= self.valley_min_frames and self.best_valley_idx is not None:
                        valley_duration_ms = self.valley_frames * self.FRAME_DURATION * 1000
                        self._emit_segment_at_valley(
                            valley_idx=self.best_valley_idx,
                            reason=f"Energy Valley tự nhiên ({valley_duration_ms:.0f}ms thấp)"
                        )
                        return  # _emit_segment_at_valley đã gọi _reset(), thoát sớm
                    # Valley không đủ dài → bỏ qua, reset đếm valley
                    self.valley_frames  = 0
                    self.best_valley_idx = None

            # ── [Lớp 3] Max Duration — ép cắt ───────────────────────────────
            if self.speech_frames >= self.max_frames:
                self._emit_segment(reason=f"Max Duration ({self.max_segment_duration:.0f}s) — cắt cứng")

        # ─────────────────────────────────────────────────────────────────────
        # [NHÁNH B] VAD phát hiện IM LẶNG
        # ─────────────────────────────────────────────────────────────────────
        else:
            if self.is_speaking:
                self.speech_buffer.append(chunk_agc)
                self.silence_frames += 1
                self.valley_frames   = 0       # Reset valley counter khi VAD đã off
                self.best_valley_idx = None

                # ── [Lớp 1] Silence Timeout ──────────────────────────────────
                if self.silence_frames >= self.silence_timeout_frames:
                    # Cắt sạch đuôi im lặng trước khi emit
                    if self.silence_frames > 0 and self.speech_buffer:
                        self.speech_buffer = self.speech_buffer[:-self.silence_frames]
                    self._emit_segment(reason=f"Silence Timeout ({self.silence_timeout:.1f}s)")

    # ──────────────────────────────────────────────────────────────────────────
    def _emit_segment_at_valley(self, valley_idx: int, reason: str):
        """
        Cắt câu tại vị trí valley (ranh giới tự nhiên), bảo toàn phần sau
        làm pre-speech cho câu kế tiếp (không bị mất phụ âm đầu câu mới).
        """
        if not self.speech_buffer or valley_idx <= 0:
            self._reset()
            return

        first_part  = self.speech_buffer[:valley_idx]
        second_part = self.speech_buffer[valley_idx:]  # Tái sử dụng làm đầu câu kế

        self._emit_segment_data(first_part, reason)

        # Khởi động câu kế tiếp ngay với phần còn lại (không mất âm thanh!)
        self.is_speaking     = True
        self.speech_buffer   = second_part
        self.speech_frames   = len(second_part)
        self.silence_frames  = 0
        self.valley_frames   = 0
        self.best_valley_idx = None
        # Không xóa pre_ring — giữ nguyên cho trường hợp câu kế kết thúc ngay

    def _emit_segment(self, reason: str):
        """Cắt câu tại điểm hiện tại và reset hoàn toàn"""
        if not self.speech_buffer:
            self._reset()
            return
        self._emit_segment_data(self.speech_buffer, reason)
        self._reset()

    def _emit_segment_data(self, frames: list, reason: str):
        """
        Xuất dữ liệu 1 câu hoàn chỉnh theo 2 kênh:
          [1] RAM Callback  → on_segment_ready(audio, sample_rate, duration_sec) — STT nhận trực tiếp
          [2] File WAV      → output_segments/segment_NNN.wav  (chỉ khi save_wav_debug=True)
        """
        if not frames:
            return

        full_audio   = np.concatenate(frames)          # Mảng float32, 16kHz, toàn bộ 1 câu
        duration_sec = len(full_audio) / float(self.sample_rate)
        self.segment_count += 1

        print(f"\n🔴 [Segmenter] Chốt câu #{self.segment_count:03d}  |  Lý do: {reason}")
        print(f"   ⏱  Thời lượng : {duration_sec:.2f}s  |  Samples: {len(full_audio)}")

        # ── [1] Đẩy thẳng qua RAM cho STT (Zero I/O — không tốn thẻ nhớ, không latency)
        if self.on_segment_ready is not None:
            self.on_segment_ready(full_audio, self.sample_rate, duration_sec)
            print(f"   📡 Đã gửi RAM : on_segment_ready() ← float32 {len(full_audio)} samples @ {self.sample_rate}Hz")

        # ── [2] Lưu file .wav (chỉ bật khi debug — TẮT khi nhúng lên thiết bị)
        if self.save_wav_debug:
            wav_path  = os.path.join(self.output_dir, f"segment_{self.segment_count:03d}.wav")
            audio_i16 = (full_audio * 32767.0).clip(-32768, 32767).astype(np.int16)
            write(wav_path, 16000, audio_i16)
            abs_path  = os.path.abspath(wav_path).replace("\\", "/")
            print(f"   💾 Đã lưu WAV : file:///{abs_path}")

        print()


# ==============================================================================
# CHƯƠNG TRÌNH TÍCH HỢP CHẠY THỬ VAD + SEGMENTER
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  🚀  KIỂM TRA TÍCH HỢP: VAD DETECTOR  +  AUDIO SEGMENTER")
    print("=" * 70)
    print("  Chiến lược ngắt câu (ưu tiên giảm dần):")
    print("   [1] Silence Timeout 0.6s  — kết thúc câu tự nhiên")
    print("   [2] Energy Valley ~96ms   — ranh giới cụm từ trong nói liên tục")
    print("   [3] Max Duration 8.0s     — ép cắt nếu quá dài")
    print("=" * 70)
    print("  💡 Nói thử bằng tiếng Việt hoặc tiếng Hoa")
    print("  ❌  Nhấn Ctrl+C để dừng\n")

    # ─────────────────────────────────────────────────────────────────────────
    # MOCK STT CALLBACK — Người làm STT thay thế hàm này bằng mô hình thật của họ
    # Nhận: audio (np.ndarray float32, 16kHz), sample_rate (int), duration_sec (float)
    # ─────────────────────────────────────────────────────────────────────────
    def mock_stt_callback(audio: np.ndarray, sample_rate: int, duration_sec: float):
        """
        ĐÂY LÀ HÀM BÀN GIAO CHO NGƯỜI LÀM STT:
        Thay nội dung bên dưới bằng code chạy mô hình STT thực tế.

        Tham số nhận vào:
          audio       : np.ndarray, dtype=float32, shape=(N,) — dữ liệu 1 câu nói hoàn chỉnh
          sample_rate : int — luôn là 16000 (Hz)
          duration_sec: float — thời lượng câu (giây)
        """
        print(f"   🤖 [Mock STT] Nhận audio {len(audio)} samples ({duration_sec:.2f}s) @ {sample_rate}Hz")
        print(f"   🤖 [Mock STT] → Đưa vào mô hình STT để nhận diện tiếng Việt / Hoa...")
        # VÍ DỤ khi tích hợp Faster-Whisper thật:
        # segments, _ = whisper_model.transcribe(audio, language=None)
        # text = "".join([s.text for s in segments]).strip()
        # print(f"   📝 Kết quả   : {text}")

    # ─────────────────────────────────────────────────────────────────────────
    segmenter = AudioSegmenter(
        sample_rate           = 16000,
        silence_timeout       = 0.6,
        min_segment_duration  = 1.2,
        max_segment_duration  = 8.0,
        valley_ratio          = 0.25,
        valley_min_frames     = 3,
        on_segment_ready      = mock_stt_callback,  # ← STT nhận audio qua RAM tại đây
        save_wav_debug        = True,               # ← Đổi thành False khi nhúng lên thiết bị
        output_dir            = "output_segments",
    )

    vad_detector = SileroVADDetector(
        threshold    = 0.35,
        on_vad_frame = segmenter.process_vad_frame
    )

    vad_detector.start()
