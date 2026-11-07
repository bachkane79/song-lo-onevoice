# 📖 Hướng Dẫn Kỹ Thuật: VAD Detector & Audio Segmenter (Sweetspot Architecture)

Tài liệu chi tiết về kiến trúc xử lý tín hiệu số (DSP), thuật toán nhận diện giọng nói (**Module 1: `SileroVADDetector`**) và thuật toán phân đoạn câu thông minh 3 giai đoạn (**Module 2: `AudioSegmenter`**) hướng tới điểm cân bằng tối ưu (**Sweetspot**: Độ nhạy / Độ trễ thấp nhưng vẫn giữ trọn vẹn ngữ cảnh mệnh đề cho STT & MT).

---

## 🏗️ 1. Tổng Quan Kiến Trúc Luồng Âm Thanh

```
 🎙️ Hardware Mic (48kHz / 44.1kHz / 16kHz, Đa kênh)
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 🚀 Module 1: VAD Detector (VAD_process.py)                              │
│  1. Khóa kênh ổn định (Channel Hysteresis 1.5x — chống nhảy pha 32ms)   │
│  2. Resample chuẩn 16000Hz (scipy resample_poly với padtype='line')     │
│  3. Soft Noise Gate (_NOISE_FLOOR = 0.005 — không cắt cụt np.zeros)     │
│  4. Sample-Level Linear Ramp AGC (Nội suy tuyến tính từng mẫu 512 mẫu)  │
│  5. Silero VAD ONNX + Context-Holding Hysteresis (Speech Hangover 96ms) │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │ Callback 32ms/frame (512 samples)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ ✂️ Module 2: Audio Segmenter (Segmenter_process.py — Stage 1 / 2 / 3)   │
│  - Pre-speech Ring Buffer 160ms (Bảo toàn phụ âm bật đầu câu)           │
│  - Acoustic Safety Gate (Chặn mẩu vụn <0.45s tại gốc, hoàn trả pre_ring)│
│  - [Lớp 1] Adaptive Silence Timeout (Co giãn động 0.70s → 0.52s → 0.38s)│
│  - [Lớp 2] Pure Prob-Dip Valley (Tách vế câu dài >=3.0s tại nhịp thở)   │
│  - [Lớp 3] Smart Max-Duration Soft-Cut (Cắt mềm tại đáy âm tiết <=8.0s) │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │ Callback RAM (Zero I/O, 0ms latency)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 🤖 Module 3: STT & MT Engine (Sherpa-ONNX SenseVoice + CTranslate2)     │
│  - Nhận mảng float32 16kHz trực tiếp qua RAM                            │
│  - Nhận dạng & Dịch song ngữ Việt ↔ Trung thời gian thực                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🎙️ 2. Module 1: VAD Detector (`VAD_process.py`)

Module [`SileroVADDetector`](VAD_process.py) giao tiếp trực tiếp với phần cứng Microphone, làm sạch tín hiệu DSP chống rè/méo biên và chạy suy luận mô hình Silero VAD ONNX.

### 🔑 Các cải tiến DSP & VAD cốt lõi:

1. **Language-Agnostic (Đa ngôn ngữ Việt & Hoa)**:
   - Mô hình **Silero VAD ONNX** phân tích đặc trưng âm học giọng người độc lập ngôn ngữ, hoạt động chính xác trên cả **Tiếng Việt** và **Tiếng Hoa** (Phổ thông / Quảng Đông).
2. **Khóa kênh phần cứng có trễ (Channel Selection Hysteresis 1.5x)**:
   - Trên các cụm Micro Array (4 kênh), việc chọn kênh có đỉnh lớn nhất ở mỗi khung 32ms dễ khiến thuật toán nhảy qua lại liên tục giữa các capsule micro, gây lệch pha và tiếng lẹt xẹt.
   - Hệ thống giữ cố định kênh hiện tại (`_active_ch`) và chỉ chuyển kênh khi kênh mới có biên độ mạnh hơn **gấp 1.5 lần** kênh cũ và vượt trên sàn nhiễu `_NOISE_FLOOR`.
3. **Resample chống méo biên (`padtype='line'`)**:
   - Mặc định `scipy.signal.resample_poly` dùng `padtype='constant'` (đệm số `0` ở hai đầu khung 32ms), tạo ra vết lõm biên độ 38% lặp lại ở tần số $31.25\text{Hz}$ gây tiếng rè cơ học.
   - Sử dụng `padtype='line'` giúp ngoại suy tuyến tính biên khung, giữ sóng âm mượt liên tục.
4. **Soft Noise Gate (`_NOISE_FLOOR = 0.005`) & Sample-Level Linear Ramp AGC**:
   - **Soft Noise Gate**: Khi biên độ thô dưới `0.005`, thay vì cắt cụt mảng về `np.zeros(512)` (gây xung vuông khi âm lượng dao động quanh ngưỡng), hệ thống đưa `target_gain = 1.0` để giữ nền âm tự nhiên mà không khuếch đại tiếng ồn phòng.
   - **Sample-by-Sample Linear Ramp AGC**: Hệ số khuếch đại EMA (`_AGC_SMOOTH = 0.08`, mục tiêu `_AGC_TARGET = 0.30`, tối đa `20.0x`) được nội suy tuyến tính trên từng mẫu âm thanh bằng `np.linspace(prev_gain, self._gain, 512)`, triệt tiêu hoàn toàn hiện tượng nhảy bậc thang giữa các khung 32ms.
5. **Context-Holding Hysteresis & Speech Hangover (Stage 3)**:
   - **Vào câu tức thì (`start_threshold = 0.35`)**: Bắt ngay âm tiết đầu tiên không bị trễ frame.
   - **Giữ mạch câu (`end_threshold = 0.22` + Hangover 3 frames = `96ms`)**: Khi người nói hạ giọng hoặc lướt âm ở giữa câu, trạng thái `is_speech` vẫn được giữ vững qua các khoảng trũng ngắn, chống đứt câu sớm.

---

## ✂️ 3. Module 2: Audio Segmenter (`Segmenter_process.py`)

Module [`AudioSegmenter`](Segmenter_process.py) nhận luồng frame 32ms từ VAD và phân đoạn theo **3 Giai đoạn (Stages)** tiến dần tới điểm **Sweetspot** (mặc định chạy **Stage 3**, có thể chọn qua tham số `stage` hoặc biến môi trường `ONEVOICE_VAD_STAGE=1|2|3`):

### 📊 Bảng So Sánh 3 Giai Đoạn (3-Stage Sweetspot Roadmap)

| Tính năng / Cơ chế | Stage 1 (Nền tảng) | Stage 2 (Tách vế tự nhiên) | Stage 3 (Sweetspot Mặc định) |
| :--- | :--- | :--- | :--- |
| **1. Adaptive Silence Timeout** | `<1.3s`: `0.65s`<br>`1.3–3.5s`: `0.65s → 0.45s`<br>`>=3.5s`: `0.45s → 0.30s` | Giống Stage 1 | `<1.5s`: `0.70s` (bảo vệ đầu câu)<br>`1.5–4.2s`: `0.70s → 0.52s`<br>`>=4.2s`: `0.52s → 0.38s` |
| **2. Acoustic Safety Gate** | Lọc mẩu vụn `<0.45s` có `mean_prob < 0.55`; hoàn trả đuôi vào `pre_ring` | Giống Stage 1 | Giống Stage 1 + Đệm đuôi câu `64ms` (2 frames) giữ phụ âm cuối |
| **3. Prob-Dip Valley (Lớp 2)** | Tắt (hoặc Legacy Energy Valley nếu bật cờ) | Bật khi câu `>= 2.2s`, `prob < 0.30` trong `96–128ms` + `2` frames xác nhận | Bật khi câu `>= 3.0s`, `prob < 0.24` trong `128–160ms` + `3` frames xác nhận |
| **4. Smart Max-Duration (Lớp 3)** | Cắt cứng tại `8.0s` | Tìm đáy xác suất (`prob < 0.65`) trong cửa sổ `2.0s` cuối câu trước khi chạm `8.0s` | Giống Stage 2 |
| **5. VAD Hysteresis & Hangover** | Ngưỡng đơn (`prob > 0.35`) | Ngưỡng đơn (`prob > 0.35`) | Ngưỡng kép (`0.35` / `0.22`) + Speech Hangover `96ms` (3 frames) |

### 🛡️ Cơ chế Bảo toàn Âm tiết tại Điểm Cắt:
- **Pre-speech Ring Buffer (`160ms` = 5 frames)**: Luôn giữ 5 frame trước khi bắt đầu câu nói để không mất các phụ âm bật đầu câu (*p, t, k, tr, ch, s...*).
- **Tail Release Padding (`64ms` = 2 frames)**: Khi cắt câu bằng Silence Timeout hoặc Valley, luôn giữ lại 2 frame đệm ở cuối vế trước để âm cuối (*-t, -p, -c, -ng*) ngân hết tự nhiên.
- **Valley Pre-roll & Trim**: Khi tách hai vế câu tại vùng trũng Valley `[valley_start .. valley_end]`, vế đầu nhận tối đa `2` frame đệm đuôi, vế thứ hai nhận tối đa `3` frame (`96ms`) pre-roll ngay trước điểm khởi âm mới, còn khoảng im lặng thừa ở giữa được cắt bỏ gọn gàng.

---

## 🔌 4. Hướng Dẫn Chạy Thử & Tích Hợp

### Chạy kiểm tra độc lập từng Giai đoạn (Stage 1 / 2 / 3):
```bash
# Chạy Stage 3 (Sweetspot mặc định)
python VAD-Segmenter/Segmenter_process.py --stage 3

# So sánh với Stage 1 hoặc Stage 2
python VAD-Segmenter/Segmenter_process.py --stage 1
python VAD-Segmenter/Segmenter_process.py --stage 2
```

### Chạy trên toàn bộ Pipeline Dịch Thuật (`run_pipeline.py`):
```bash
# Mặc định chạy Stage 3
python run_pipeline.py

# Hoặc chọn Stage qua biến môi trường (PowerShell)
$env:ONEVOICE_VAD_STAGE="2"; python run_pipeline.py
```

### Code mẫu tích hợp trực tiếp vào Module STT qua RAM:
```python
from VAD_process import SileroVADDetector
from Segmenter_process import AudioSegmenter

def stt_callback(audio_float32, sample_rate, duration_sec):
    """
    Được AudioSegmenter tự động gọi mỗi khi chốt xong 1 câu/vế câu hoàn chỉnh:
    - audio_float32 : mảng numpy float32 chuẩn 16000Hz
    - sample_rate   : 16000
    - duration_sec  : thời lượng đoạn âm thanh (giây)
    """
    print(f"🎙️ [STT] Nhận câu nói {duration_sec:.2f}s từ RAM...")

segmenter = AudioSegmenter(
    on_segment_ready = stt_callback,
    save_wav_debug   = False,   # Đặt False khi chạy thực tế (Zero I/O)
    stage            = 3        # Sử dụng kiến trúc Sweetspot Stage 3
)

vad_detector = SileroVADDetector(
    threshold    = 0.35,
    on_vad_frame = segmenter.process_vad_frame,
    stage        = 3
)

vad_detector.start()
```

---

## 📦 5. Yêu Cầu Thư Viện (`requirements.txt`)

```text
numpy>=1.22.0
onnxruntime>=1.14.0
sounddevice>=0.4.5
scipy>=1.8.0
```
