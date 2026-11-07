import os
import sys
import argparse
import numpy as np
from collections import deque
from scipy.io.wavfile import write

from VAD_process import SileroVADDetector

# Cài đặt UTF-8 cho Windows Terminal Output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


class AudioSegmenter:
    """
    Module Segmenter: Phân đoạn câu nói từ stream VAD hướng tới điểm Sweetspot
    (cân bằng giữa Độ nhạy / Độ trễ thấp và Độ ổn định ngữ cảnh cho STT/MT).

    Hỗ trợ 3 Giai đoạn cải tiến (chọn qua tham số `stage` hoặc biến môi trường ONEVOICE_VAD_STAGE):
      - Giai đoạn 1 (stage=1):
          + Adaptive Silence Timeout: Ngưỡng chờ im lặng tự động co giãn theo độ dài câu đang nói.
          + Acoustic Safety Gate: Chặn các mẩu vụn ngắn (<0.45s) có độ tin cậy thấp ngay tại gốc,
            đồng thời bảo lưu vào pre_ring để tránh mất âm đầu nếu người nói ngập ngừng.
      - Giai đoạn 2 (stage=2):
          + Bao gồm Giai đoạn 1.
          + Prob-Dip & Micro-Pause Valley Detection: Phát hiện nhịp nghỉ lấy hơi ngắn dựa trên
            độ sụt xác suất VAD (Prob-Dip), cắt tại ĐÁY thung lũng khi người nói bắt đầu vế câu mới.
          + Smart Max-Duration Cut: Khi chạm trần 8.0s, tự động tìm điểm ranh giới âm tiết có xác suất
            thấp nhất trong cửa sổ cuối câu để cắt mềm thay vì cắt ngang thân từ.
      - Giai đoạn 3 (stage=3, mặc định):
          + Bao gồm Giai đoạn 1 & 2, kết hợp với Context-Holding Hysteresis + Speech Hangover (96ms)
            từ SileroVADDetector và tinh chỉnh ngưỡng chờ kiên nhẫn hơn ở giữa câu (chống ngắt câu sớm
            khi người nói hạ giọng hoặc ngập ngừng suy nghĩ chưa hết câu).
    """

    FRAME_SIZE = 512        # samples (32ms at 16kHz)
    FRAME_DURATION = FRAME_SIZE / 16000  # 0.032s

    def __init__(
        self,
        sample_rate: int              = 16000,
        silence_timeout: float        = 0.6,    # [Lớp 1] Thời gian chờ im lặng cơ sở (s)
        min_segment_duration: float   = 1.2,    # Không tách vế Valley trước khi câu đủ dài (s)
        max_segment_duration: float   = 8.0,    # [Lớp 3] Ép cắt nếu vượt quá (s)
        valley_ratio: float           = 0.25,   # [Lớp 2] Năng lượng < ratio * peak = valley (chỉ dùng ở Legacy)
        valley_min_frames: int        = 3,      # Số frame tối thiểu cho Energy Valley (~96ms)
        energy_window_frames: int     = 40,     # Cửa sổ trượt tính peak năng lượng (~1.28s)
        pre_speech_frames: int        = 5,      # Giữ 5 frame (~160ms) trước câu nói
        on_segment_ready              = None,   # Callback(audio, sample_rate, duration_sec) → STT nhận qua RAM
        save_wav_debug: bool          = True,   # True = lưu file .wav để nghe kiểm tra; False = chỉ RAM
        output_dir: str               = "output_segments",
        stage: int                    = None,   # 1, 2, hoặc 3 (None = đọc ONEVOICE_VAD_STAGE, mặc định 3)
        min_utterance_duration: float = 0.45,   # [GĐ 1] Ngưỡng chặn mẩu vụn tại gốc Segmenter (s)
        prob_valley_threshold: float  = 0.30,   # [GĐ 2-3] Ngưỡng xác suất VAD phát hiện nhịp nghỉ lấy hơi
    ):
        self.sample_rate            = sample_rate
        self.silence_timeout        = silence_timeout
        self.min_segment_duration   = min_segment_duration
        self.max_segment_duration   = max_segment_duration
        self.valley_ratio           = valley_ratio
        self.valley_min_frames      = valley_min_frames
        self.pre_speech_frames      = pre_speech_frames
        self.on_segment_ready       = on_segment_ready   # Hook cho STT — nhận audio float32 qua RAM
        self.save_wav_debug         = save_wav_debug     # Tắt khi nhúng lên thiết bị
        self.output_dir             = output_dir
        self.min_utterance_duration = min_utterance_duration
        self.prob_valley_threshold  = prob_valley_threshold

        if stage is None:
            try:
                stage = int(os.environ.get("ONEVOICE_VAD_STAGE", "3"))
            except ValueError:
                stage = 3
        self.stage = int(stage)

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
        self.is_speaking               = False
        self.speech_buffer             = []     # Danh sách chunk_agc của câu hiện tại
        self.speech_probs              = []     # Xác suất VAD tương ứng từng frame trong speech_buffer
        self.speech_frames             = 0      # Số frame kể từ khi bắt đầu nói
        self.silence_frames            = 0      # Số frame VAD im lặng liên tiếp ở cuối câu
        self.valley_frames             = 0      # Số frame liên tiếp trong vùng trũng (Prob-Dip)
        self.best_valley_idx           = None   # Vị trí frame bắt đầu vùng trũng (Valley)
        self.post_valley_speech_frames = 0      # Số frame nói rõ liên tiếp sau valley (xác nhận vế câu mới)
        self.pre_ring                  = deque(maxlen=self.pre_speech_frames)

    # ──────────────────────────────────────────────────────────────────────────
    def _frame_energy(self, chunk: np.ndarray) -> float:
        """Tính năng lượng RMS của một frame audio"""
        return float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0

    def _peak_energy(self) -> float:
        """Đỉnh năng lượng trong cửa sổ gần nhất"""
        return max(self._energy_window) if self._energy_window else 0.0

    # ──────────────────────────────────────────────────────────────────────────
    def _get_adaptive_silence_frames(self) -> int:
        """
        Tính số frame chờ im lặng động (Adaptive Silence Timeout) theo thời lượng nói thực tế:
          - Ở Stage 1 & 2:
              + Mở đầu (< 1.3s)       : 0.65s
              + Context Vàng (1.3-3.5s): giảm mượt về 0.45s
              + Câu dài (>= 3.5s)     : giảm mượt về 0.30s
          - Ở Stage 3 (Bảo vệ mạch câu tối đa, chống ngắt sớm khi đang nghĩ từ giữa câu):
              + Mở đầu (< 1.5s)       : 0.70s
              + Context Vàng (1.5-4.2s): giảm mượt về 0.52s
              + Câu dài (>= 4.2s)     : giảm mượt về 0.38s (kết hợp Hangover 96ms của VAD)
        """
        if self.stage < 1:
            return self.silence_timeout_frames

        active_frames = max(0, self.speech_frames - self.silence_frames)
        active_sec    = active_frames * self.FRAME_DURATION

        if self.stage >= 3:
            base_sec = max(self.silence_timeout, 0.70)
            if active_sec < 1.5:
                timeout_sec = base_sec
            elif active_sec < 4.2:
                ratio       = (active_sec - 1.5) / (4.2 - 1.5)
                timeout_sec = base_sec - ratio * (base_sec - 0.52)
            else:
                ratio       = min(1.0, (active_sec - 4.2) / 2.3)
                timeout_sec = 0.52 - ratio * (0.52 - 0.38)
            return max(12, int(round(timeout_sec / self.FRAME_DURATION)))

        # Stage 1 & 2
        base_sec = max(self.silence_timeout, 0.65)
        if active_sec < 1.3:
            timeout_sec = base_sec
        elif active_sec < 3.5:
            ratio       = (active_sec - 1.3) / (3.5 - 1.3)
            timeout_sec = base_sec - ratio * (base_sec - 0.45)
        else:
            ratio       = min(1.0, (active_sec - 3.5) / 2.0)
            timeout_sec = 0.45 - ratio * (0.45 - 0.30)

        return max(10, int(round(timeout_sec / self.FRAME_DURATION)))

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

        # ── Khi chưa vào câu nói: tích lũy pre-speech ring buffer ────────────
        if not self.is_speaking:
            if not is_speech:
                self.pre_ring.append(chunk_agc)
                return

            # Bắt đầu câu mới khi phát hiện tiếng nói
            self.is_speaking = True
            print(f"\n🟢 [Segmenter] Bắt đầu câu mới — VAD prob={prob:.3f}")
            # Đưa pre-speech vào đầu câu để không mất phụ âm đầu (p, t, k, s, ch...)
            self.speech_buffer.extend(self.pre_ring)
            self.speech_probs.extend([0.10] * len(self.pre_ring))
            self.speech_frames = len(self.speech_buffer)

        # ── Đang trong câu nói: ghi nhận frame hiện tại vào buffer ───────────
        self.speech_buffer.append(chunk_agc)
        self.speech_probs.append(prob)
        self.speech_frames += 1

        if is_speech:
            self.silence_frames = 0
        else:
            self.silence_frames += 1

        # ═════════════════════════════════════════════════════════════════════
        # [LỚP 1] SILENCE TIMEOUT (Giai đoạn 1+: Adaptive Silence Timeout)
        # ═════════════════════════════════════════════════════════════════════
        if not is_speech and self.silence_frames > 0:
            adaptive_timeout_frames = self._get_adaptive_silence_frames()
            if self.silence_frames >= adaptive_timeout_frames:
                # Giữ lại 2 frame (~64ms) đệm cuối câu ở Stage 1+ để không cụt phụ âm cuối (-t, -p, -c, -ng)
                trim_frames = max(0, self.silence_frames - 2) if self.stage >= 1 else self.silence_frames
                if trim_frames > 0 and len(self.speech_buffer) > trim_frames:
                    clean_buffer   = self.speech_buffer[:-trim_frames]
                    clean_probs    = self.speech_probs[:-trim_frames]
                else:
                    clean_buffer   = self.speech_buffer
                    clean_probs    = self.speech_probs

                clean_duration = len(clean_buffer) * self.FRAME_DURATION

                # ── [Giai đoạn 1+] Acoustic Safety Gate: Chặn mẩu vụn tại gốc ─
                if self.stage >= 1 and clean_duration < self.min_utterance_duration:
                    active_probs = [p for p in clean_probs if p >= 0.35]
                    mean_prob    = float(np.mean(active_probs)) if active_probs else 0.0
                    # Nếu mẩu quá ngắn (<0.28s) hoặc xác suất giọng nói yếu -> không đẩy rác sang STT,
                    # giữ lại các frame cuối vào pre_ring để ghép nối nếu người nói nói tiếp ngay sau đó
                    if clean_duration < 0.28 or mean_prob < 0.55 or len(active_probs) < 5:
                        recent_frames = clean_buffer[-self.pre_speech_frames:]
                        self._reset()
                        self.pre_ring.extend(recent_frames)
                        return

                self.speech_buffer = clean_buffer
                eff_timeout_sec    = adaptive_timeout_frames * self.FRAME_DURATION
                reason_str = (
                    f"Adaptive Silence Timeout ({eff_timeout_sec:.2f}s)"
                    if self.stage >= 1 else f"Silence Timeout ({self.silence_timeout:.1f}s)"
                )
                self._emit_segment(reason=reason_str)
                return

        # ═════════════════════════════════════════════════════════════════════
        # [LỚP 2] VALLEY DETECTION (Giai đoạn 2+: Pure Prob-Dip Valley)
        # ═════════════════════════════════════════════════════════════════════
        active_speech_frames = max(0, self.speech_frames - self.silence_frames)
        active_speech_sec    = active_speech_frames * self.FRAME_DURATION
        peak                 = self._peak_energy()

        if self.stage >= 2:
            # Ở Stage 3, ưu tiên độ ổn định mạch câu: chỉ xét tách vế khi câu đã dài >= 3.0s
            # (Stage 2 xét từ >= 2.2s). Tuyệt đối KHÔNG dùng is_energy_dip vì AGC nén dải động dễ cắt nhầm!
            min_valley_sec      = 3.0 if self.stage >= 3 else max(2.2, self.min_segment_duration)
            valley_start_frames = int(min_valley_sec / self.FRAME_DURATION)
            dip_thresh          = 0.24 if self.stage >= 3 else self.prob_valley_threshold

            if active_speech_frames >= valley_start_frames:
                is_prob_dip = (prob < dip_thresh)

                if is_prob_dip:
                    self.valley_frames += 1
                    self.post_valley_speech_frames = 0
                    if self.best_valley_idx is None:
                        self.best_valley_idx = len(self.speech_buffer) - 1
                else:
                    # Âm thanh đã phục hồi lên mức nói:
                    # - Stage 3: đòi hỏi nhịp nghỉ rõ ràng hơn (5 frames = 160ms với câu 3.0-5.0s;
                    #   4 frames = 128ms với câu > 5.0s) và 3 frames nói rõ liên tiếp xác nhận vế mới.
                    # - Stage 2: 4 frames (128ms) hoặc 3 frames (96ms khi >= 4.2s) và 2 frames xác nhận.
                    if self.stage >= 3:
                        req_valley_frames  = 4 if active_speech_sec >= 5.0 else 5
                        req_confirm_frames = 3
                    else:
                        req_valley_frames  = 3 if active_speech_sec >= 4.2 else 4
                        req_confirm_frames = 2

                    if self.valley_frames >= req_valley_frames and self.best_valley_idx is not None:
                        if prob >= 0.45:
                            self.post_valley_speech_frames += 1
                            if self.post_valley_speech_frames >= req_confirm_frames:
                                valley_ms = self.valley_frames * self.FRAME_DURATION * 1000
                                self._emit_segment_at_valley(
                                    valley_idx=self.best_valley_idx,
                                    reason=f"Prob-Dip Valley tự nhiên ({valley_ms:.0f}ms nghỉ lấy hơi)"
                                )
                                return
                        else:
                            # Không đủ xác nhận tiếng nói rõ sau valley -> xóa sạch trạng thái valley
                            # (không để treo valley_frames gây cắt nhầm ở các giây sau)
                            self.valley_frames             = 0
                            self.best_valley_idx           = None
                            self.post_valley_speech_frames = 0
                    else:
                        # Khoảng trũng ngắn (chỉ là âm lướt/âm tắc trong câu) -> xóa bộ đếm valley
                        self.valley_frames             = 0
                        self.best_valley_idx           = None
                        self.post_valley_speech_frames = 0
        else:
            # [Giai đoạn 1 / Legacy] Chỉ dùng Energy Valley truyền thống khi valley_ratio > 0
            if is_speech and self.speech_frames >= self.min_frames:
                if self.valley_ratio > 0 and peak > 1e-5 and energy < self.valley_ratio * peak:
                    self.valley_frames += 1
                    if self.best_valley_idx is None:
                        self.best_valley_idx = len(self.speech_buffer) - 1
                else:
                    if self.valley_frames >= self.valley_min_frames and self.best_valley_idx is not None:
                        valley_duration_ms = self.valley_frames * self.FRAME_DURATION * 1000
                        self._emit_segment_at_valley(
                            valley_idx=self.best_valley_idx,
                            reason=f"Energy Valley tự nhiên ({valley_duration_ms:.0f}ms thấp)"
                        )
                        return
                    self.valley_frames   = 0
                    self.best_valley_idx = None
            elif not is_speech:
                self.valley_frames   = 0
                self.best_valley_idx = None

        # ═════════════════════════════════════════════════════════════════════
        # [LỚP 3] MAX DURATION — Ép cắt (Giai đoạn 2+: Smart Soft-Cut tại ranh giới âm tiết)
        # ═════════════════════════════════════════════════════════════════════
        if self.speech_frames >= self.max_frames:
            if self.stage >= 2 and len(self.speech_probs) >= 60:
                # Tìm điểm có xác suất VAD thấp nhất trong cửa sổ ~2.0s cuối câu
                # (bỏ qua 4 frame sát cuối để phần giữ lại có nghĩa) nhằm tránh cắt ngang thân từ
                search_start = max(self.min_frames, len(self.speech_probs) - 64)
                search_end   = len(self.speech_probs) - 4
                if search_end > search_start:
                    tail_probs = np.array(self.speech_probs[search_start:search_end])
                    min_rel    = int(np.argmin(tail_probs))
                    if tail_probs[min_rel] < 0.65:
                        soft_idx = search_start + min_rel
                        self._emit_segment_at_valley(
                            valley_idx=soft_idx,
                            reason=f"Max Duration Soft-Cut ({soft_idx * self.FRAME_DURATION:.2f}s / {self.max_segment_duration:.0f}s)"
                        )
                        return

            self._emit_segment(reason=f"Max Duration ({self.max_segment_duration:.0f}s) — cắt cứng")

    # ──────────────────────────────────────────────────────────────────────────
    def _emit_segment_at_valley(self, valley_idx: int, reason: str):
        """
        Cắt câu tại vùng trũng valley (ranh giới tự nhiên), cấp đệm đuôi cho vế đầu
        và đệm đầu (pre-roll) cho vế kế tiếp mà không làm rớt bất kỳ âm tiết nào.
        """
        if not self.speech_buffer or valley_idx <= 0:
            self._reset()
            return

        # Xác định vùng bắt đầu và kết thúc của nhịp nghỉ (valley_start .. valley_end)
        valley_start = valley_idx
        if self.post_valley_speech_frames > 0 and self.valley_frames > 0:
            valley_end = max(valley_start + 1, len(self.speech_buffer) - self.post_valley_speech_frames)
        else:
            valley_end = min(len(self.speech_buffer), valley_start + max(1, self.valley_frames))

        # Cấp tối đa 2 frame (~64ms) đệm đuôi cho vế trước để giữ trọn phụ âm cuối
        first_end  = min(valley_end, valley_start + 2)
        first_part = self.speech_buffer[:first_end]

        # Đảm bảo vế đầu đủ dài tối thiểu (>= 1.2s) trước khi cho phép cắt valley
        if len(first_part) * self.FRAME_DURATION < 1.2:
            self.valley_frames             = 0
            self.best_valley_idx           = None
            self.post_valley_speech_frames = 0
            return

        # Cấp tối đa 3 frame (~96ms) pre-roll trước điểm khởi âm của vế mới
        second_start = max(valley_start, valley_end - 3)
        second_part  = self.speech_buffer[second_start:]
        second_probs = self.speech_probs[second_start:]

        self._emit_segment_data(first_part, reason)

        # Khởi động câu kế tiếp ngay với phần còn lại (không mất âm thanh đầu vế mới!)
        self.is_speaking               = True
        self.speech_buffer             = second_part
        self.speech_probs              = second_probs
        self.speech_frames             = len(second_part)
        self.silence_frames            = 0
        self.valley_frames             = 0
        self.best_valley_idx           = None
        self.post_valley_speech_frames = 0

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
            os.makedirs(self.output_dir, exist_ok=True)
            wav_path  = os.path.join(self.output_dir, f"segment_{self.segment_count:03d}.wav")
            audio_i16 = (full_audio * 32767.0).clip(-32768, 32767).astype(np.int16)
            write(wav_path, 16000, audio_i16)
            abs_path  = os.path.abspath(wav_path).replace("\\", "/")
            print(f"   💾 Đã lưu WAV : file:///{abs_path}")

        print()


# ==============================================================================
# CHƯƠNG TRÌNH TÍCH HỢP CHẠY THỬ VAD + SEGMENTER THEO TỪNG GIAI ĐOẠN
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm tra tích hợp VAD Detector + Audio Segmenter")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=3,
                        help="Chọn giai đoạn cải tiến cần test (1, 2, hoặc 3 - mặc định 3)")
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="Ngưỡng VAD cơ sở (mặc định 0.35)")
    parser.add_argument("--silence-timeout", type=float, default=0.6,
                        help="Silence timeout cơ sở (mặc định 0.6s)")
    parser.add_argument("--device", type=int, default=None,
                        help="ID thiết bị Micro")
    parser.add_argument("--no-wav", action="store_true",
                        help="Tắt lưu file WAV debug")
    args = parser.parse_args()

    stage_descriptions = {
        1: "GIAI ĐOẠN 1: Adaptive Silence Timeout (0.65s→0.45s→0.30s) + Safety Gate lọc vụn <0.45s",
        2: "GIAI ĐOẠN 2: GĐ 1 + Prob-Dip Valley (tách vế câu >=2.2s tại nhịp thở) + Smart Soft-Cut",
        3: "GIAI ĐOẠN 3: Context-Holding Hysteresis (Speech Hangover 96ms giữ mạch câu) + Prob-Dip Valley (>=3.0s)",
    }

    print("=" * 78)
    print(f"  🚀  KIỂM TRA TÍCH HỢP: VAD DETECTOR + AUDIO SEGMENTER (STAGE {args.stage})")
    print("=" * 78)
    print(f"  🎯 Chế độ đang bật: {stage_descriptions[args.stage]}")
    print("  Chiến lược ngắt câu (ưu tiên phối hợp):")
    print("   [1] Adaptive Silence Timeout — co giãn thông minh theo độ dài câu")
    print("   [2] Prob-Dip Valley          — ngắt tại đáy nhịp thở trong câu nói dài liên tục")
    print("   [3] Smart Max Duration       — tìm ranh giới âm tiết nếu câu chạm trần 8.0s")
    print("=" * 78)
    print("  💡 Hãy nói thử bằng tiếng Việt hoặc tiếng Hoa để cảm nhận điểm cắt (Ctrl+C để dừng)\n")

    def mock_stt_callback(audio: np.ndarray, sample_rate: int, duration_sec: float):
        print(f"   🤖 [Mock STT] Nhận audio {len(audio)} samples ({duration_sec:.2f}s) @ {sample_rate}Hz")

    # Tắt valley_ratio (0.0) ở Stage 1 chạy độc lập giống hệt cấu hình chuẩn của OneVoicePipeline
    # để không bị Energy Valley cũ cắt nhầm do AGC san phẳng biên độ
    segmenter = AudioSegmenter(
        sample_rate          = 16000,
        silence_timeout      = args.silence_timeout,
        min_segment_duration = 2.2,
        max_segment_duration = 8.0,
        valley_ratio         = 0.0,
        valley_min_frames    = 3,
        on_segment_ready     = mock_stt_callback,
        save_wav_debug       = not args.no_wav,
        output_dir           = "output_segments",
        stage                = args.stage,
    )

    vad_detector = SileroVADDetector(
        threshold    = args.threshold,
        device_id    = args.device,
        on_vad_frame = segmenter.process_vad_frame,
        stage        = args.stage,
    )

    vad_detector.start()
