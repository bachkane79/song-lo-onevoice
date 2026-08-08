# 📖 Hướng Dẫn Kỹ Thuật: VAD Detector & Audio Segmenter

Tài liệu chi tiết về cách hoạt động, kiến trúc thuật toán và hướng dẫn bàn giao cho **Module 1 (VAD Detector)** và **Module 2 (Audio Segmenter)** trong dự án thiết bị phiên dịch du lịch Việt - Hoa.

---

## 🏗️ 1. Tổng Quan Kiến Trúc Luồng Âm Thanh

```
 🎙️ Hardware Mic
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ 🚀 Module 1: VAD Detector (VAD_process.py)                  │
│  - Chọn kênh Micro tốt nhất                                 │
│  - Resample về chuẩn 16000Hz (scipy resample_poly)          │
│  - Noise Gate (Lọc ồn nền < 0.005)                          │
│  - Smoothed AGC (Khuếch đại mượt ~0.30, chống rè)           │
│  - Silero VAD ONNX Engine (Phân tích Việt & Hoa)            │
└──────────────────────────────┬──────────────────────────────┘
                               │ Callback 32ms/frame
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ ✂️ Module 2: Audio Segmenter (Segmenter_process.py)          │
│  - Pre-speech Buffer 160ms (Giữ phụ âm đầu câu)             │
│  - Lớp 1: Silence Timeout (Im lặng 0.6s → Ngắt câu)        │
│  - Lớp 2: Energy Valley Detection (Phát hiện cụm từ ~96ms)  │
│  - Lớp 3: Max Duration (Cắt cứng sau 8.0s)                  │
└──────────────────────────────┬──────────────────────────────┘
                               │ Callback RAM (0ms latency, Zero I/O)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 🤖 Module 3: STT Engine (Whisper / Sherpa-ONNX / FunASR...) │
│  - Nhận mảng float32 16kHz trực tiếp qua RAM                │
│  - Chuyển âm thanh thành văn bản Tiếng Việt / Tiếng Hoa     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎙️ 2. Module 1: VAD Detector (`VAD_process.py`)

Module `SileroVADDetector` chịu trách nhiệm giao tiếp với phần cứng Microphone, xử lý tín hiệu âm thanh thô và chạy suy luận mô hình VAD AI.

### 🔑 Các tính năng kỹ thuật cốt lõi:
1. **Language-Agnostic (Đa ngôn ngữ Việt & Hoa)**:
   - Sử dụng mô hình **Silero VAD ONNX**, phân tích tần số dây thanh âm $f_0$ của người nên hoạt động chính xác trên cả **Tiếng Việt** và **Tiếng Hoa**.
2. **Lọc kênh tự động (Best Channel Selection)**:
   - Micro đa kênh trên Windows/Linux thường chứa kênh im lặng. Hệ thống dùng `np.argmax(np.max(np.abs(indata), axis=0))` để tự chọn kênh âm thanh mạnh nhất.
3. **Bộ lọc Noise Gate (`_NOISE_FLOOR = 0.005`)**:
   - Âm thanh thô dưới ngưỡng `0.005` (tiếng quạt, điều hòa, nhiễu điện) bị zeroed về `0.0`, tránh việc khuếch đại nhiễu nền.
4. **Khuếch đại biên độ mượt - Smoothed AGC (`_AGC_SMOOTH = 0.08`)**:
   - Giúp câu nói nhỏ được đẩy lên biên độ mục tiêu `~0.30`.
   - Sử dụng thuật toán **Exponential Moving Average (EMA)** giúp gain thay đổi mượt mà giữa các frame 32ms, loại bỏ hoàn toàn tiếng rè/crackle.
5. **Duy trì Context & RNN State**:
   - Mỗi frame gửi vào mô hình Silero VAD có shape `(1, 576)` gồm `64` mẫu context + `512` mẫu audio hiện tại.
   - Trạng thái ẩn RNN `state (2, 1, 128)` được duy trì liên tục qua các khung.

---

## ✂️ 3. Module 2: Audio Segmenter (`Segmenter_process.py`)

Module `AudioSegmenter` nhận luồng tín hiệu VAD (32ms/frame) và cắt dòng âm thanh thành các phân đoạn câu hoàn chỉnh.

### 🎯 3 Lớp Ưu Tiên Ngắt Câu (Multi-layer Strategy):

| Lớp | Tên cơ chế | Điều kiện kích hoạt | Ý nghĩa |
| :--- | :--- | :--- | :--- |
| **[Lớp 1]** | **Silence Timeout** | VAD im lặng liên tiếp $\ge 0.6\text{s}$ | Người dùng ngắt nghỉ câu tự nhiên. Đuôi im lặng thừa được tự động cắt sạch trước khi xuất câu. |
| **[Lớp 2]** | **Energy Valley Detection** ⭐ | Đã nói $\ge 1.2\text{s}$ **+** năng lượng âm thanh giảm $< 25\%$ so với đỉnh gần nhất trong $\sim 96\text{ms}$ | Khi người dùng nói liên tục không nghỉ, thuật toán phát hiện khoảng hở năng lượng tự nhiên giữa các cụm từ để cắt câu. |
| **[Lớp 3]** | **Max Duration** | Đã nói liên tục đạt $8.0\text{s}$ | Ép cắt cứng an toàn để tránh quá tải cho mô hình STT phía sau. |

### 🛡️ Cơ chế Bảo toàn Âm thanh:
- **Pre-speech Buffer (160ms)**: Giữ 5 frame xoay vòng trước khi VAD báo có tiếng nói để không bị ăn mất phụ âm bật đầu câu (*p, t, k, tr, ch...*).
- **Valley Re-use**: Khi cắt câu tại điểm Energy Valley, phần audio từ vị trí valley trở đi được giữ lại làm đầu câu tiếp theo (không bị rơi rớt chữ).

---

## 🔌 4. Hướng Dẫn Tích Hợp Cho Thành viên (STT Integration)

Thành viên làm phần **STT (Speech-to-Text)** chỉ cần import `AudioSegmenter` và truyền hàm nhận dạng của họ vào tham số `on_segment_ready`:

### Code mẫu tích hợp STT:

```python
from VAD_process import SileroVADDetector
from Segmenter_process import AudioSegmenter
# import faster_whisper / sherpa_onnx...

def stt_callback(audio_float32, sample_rate, duration_sec):
    """
    HÀM NÀY SẼ ĐƯỢC SEGMENTER TỰ ĐỘNG GỌI MỖI KHI CÓ 1 CÂU HOÀN CHỈNH:
    - audio_float32 : mảng numpy float32 chuẩn 16000Hz chứa trọn vẹn 1 câu
    - sample_rate   : 16000
    - duration_sec  : thời lượng câu (giây)
    """
    print(f"🎙️ [STT] Nhận câu nói {duration_sec:.2f}s từ RAM...")
    
    # Ví dụ truyền trực tiếp vào mô hình STT (Faster-Whisper / Sherpa-ONNX):
    # segments, _ = model.transcribe(audio_float32, language="vi") # hoặc "zh"
    # text = "".join([s.text for s in segments])
    # print(f"📝 Văn bản: {text}")

# 1. Khởi tạo Segmenter (Truyền callback qua RAM)
segmenter = AudioSegmenter(
    on_segment_ready = stt_callback, # STT nhận audio qua RAM tại đây
    save_wav_debug   = False        # Đặt False khi nhúng thiết bị (Zero I/O wear)
)

# 2. Khởi tạo VAD Detector và nối với Segmenter
vad_detector = SileroVADDetector(
    threshold    = 0.35,
    on_vad_frame = segmenter.process_vad_frame
)

# 3. Chạy luồng
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

Cài đặt bằng lệnh:
```bash
pip install -r requirements.txt
```
