# 🔧 Cài đặt trên Jetson

Tài liệu này chỉ nói về phần **nhúng lên thiết bị**. Kiến thức về máy dịch nằm ở
[TRANSLATE_GUIDE.md](TRANSLATE_GUIDE.md).

---

## ⚠️ 1. Xác định đúng board trước khi làm gì khác

Hai board tên gần giống nhau nhưng cách cài **khác hẳn**:

| | Jetson Nano (2019) | Jetson Orin Nano (2023) |
|---|---|---|
| CPU | 4× Cortex-A57 @ 1.43 GHz | 6× Cortex-A78AE @ 1.5 GHz |
| JetPack cuối cùng được hỗ trợ | **4.6.x** (đã ngừng phát triển) | 5.x / 6.x |
| Ubuntu | 18.04 | 20.04 / 22.04 |
| Python hệ thống | **3.6** | 3.8 / 3.10 |
| `pip install ctranslate2` | ❌ **không chạy** (xem dưới) | ✅ chạy |

Kiểm tra board của bạn:

```bash
cat /etc/nv_tegra_release
python3 --version
uname -m          # phải là aarch64
```

### Nếu là Jetson Nano đời cũ (Python 3.6)

CTranslate2 4.x yêu cầu Python ≥ 3.8 nên **không có wheel nào cài được**. Ba lựa
chọn, theo thứ tự nên thử:

1. **Cài Python 3.8+ song song** (`deadsnakes` PPA hoặc conda/miniforge cho
   aarch64), rồi tạo venv và `pip install ctranslate2 tokenizers`. Ít rủi ro nhất.
2. **Build CTranslate2 từ nguồn** — mất 30-60 phút trên Nano, cần bật swap.
3. **Đổi sang Orin Nano.** Nếu dự án còn thời gian chọn phần cứng thì đây là lựa
   chọn đúng: A57 chậm hơn A78 khoảng 3-4 lần, mà máy dịch chạy trên CPU.

---

## 📦 2. Cài đặt

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv portaudio19-dev libsndfile1

git clone https://github.com/bachkane79/song-lo-onevoice.git
cd song-lo-onevoice

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

**`onnxruntime` trên Jetson**: đừng cài bản PyPI mặc định nếu muốn dùng GPU cho
VAD. NVIDIA phát hành wheel riêng cho từng JetPack tại
[Jetson Zoo](https://elinux.org/Jetson_Zoo#ONNX_Runtime). Với Silero VAD (model
2.3 MB, chạy 32 ms/frame) thì bản CPU đã thừa sức — chỉ đổi khi đo thấy thiếu.

---

## 🚚 3. Mang model sang thiết bị

Model **không nằm trong git** (GitHub giới hạn 100 MB/file). Tổng cần **~652 MB**.

**Cách nhanh nhất — một lệnh làm hết:**

```bash
python3 scripts/setup.py
```

Nó chạy đúng hai bước dưới đây rồi dịch thử một câu mỗi chiều. Chỉ đọc tiếp phần
này khi cần làm tay (Jetson không có mạng chẳng hạn).

### 3.1. STT (258 MB) — tự tải, không cần copy

```bash
python3 scripts/download_stt.py
```

Script tải thẳng bản int8 từ Hugging Face (SenseVoice 229 MB + Zipformer 29 MB),
xóa bản fp32 thừa, rồi **nạp thử cả hai recognizer** để chắc chắn dùng được chứ
không chỉ kiểm tra file có tồn tại.

Jetson không có mạng thì copy thư mục `models/stt/` từ máy dev sang, rồi:

```bash
python3 scripts/download_stt.py --verify-only
```

### 3.2. Máy dịch (397 MB) — tải từ GitHub Release

```bash
python3 scripts/prepare_model.py
```

Tải 4 file từ release `mt-model-v1` của chính repo này và **kiểm tra sha256** —
file `model.bin` tải thiếu vẫn nạp được vào CTranslate2 và vẫn sinh ra chữ, chỉ
là chữ rác, nên bước kiểm tra này không bỏ được.

Jetson không có mạng thì copy tay từ máy dev:

```bash
scp -r ct2/nllb-trim-int8                jetson@<ip>:~/onevoice-model/
scp    nllb-zhvi-trimmed/tokenizer.json  jetson@<ip>:~/onevoice-model/

python3 scripts/prepare_model.py --source ~/onevoice-model
```

`prepare_model.py` không chỉ copy — nó verify `decoder_start_token`, kiểm tra
tokenizer có đủ `zho_Hans`/`vie_Latn`, và **so số vocab giữa model CT2 và
tokenizer**. Ghép nhầm cặp model/tokenizer sẽ dịch ra chuỗi rác mà không báo lỗi
gì; đây là bước chặn lỗi đó.

---

## ⚡ 4. Đo trước, chỉnh sau

```bash
python3 scripts/bench_stt.py --wav-dir my_audio/     # STT
python3 scripts/bench_translate.py --sweep-beam      # máy dịch
python3 scripts/bench_translate.py --sweep-threads
```

Số tham chiếu trên máy dev **i9-9900K**:

| tầng | cấu hình | chi phí cho câu 4 giây |
|---|---|---|
| STT SenseVoice (zh) | 2 thread | 0.17 s (RTF 0.042) |
| STT Zipformer (vi) | 2 thread | 0.08 s (RTF 0.021) |
| STT song ngữ `dual` | 2 thread | **0.24 s** (RTF 0.060) |
| Máy dịch | beam 1 | 0.20 s |
| Máy dịch | **beam 4 (mặc định)** | **0.25 s** |
| **Tổng** | | **~0.5 s / câu 4 giây** |

Peak RSS máy dịch: 966–972 MB. Cộng STT (~260 MB) → **~1.3 GB**.

Jetson sẽ chậm hơn nhiều lần — **đừng ngoại suy, hãy đo**. Cách đọc kết quả:

- Câu nói thường dài 2-4 s. Nếu `STT + dịch` vượt quá độ dài câu nói thì hàng đợi
  dồn lại và pipeline bắt đầu bỏ câu (số này in ra khi dừng `run_pipeline.py`).
- Thấy chậm, thử theo thứ tự **hiệu quả giảm dần**:
  ```bash
  python3 run_pipeline.py --force-lang zh   # bỏ hẳn 1 model STT — lợi nhất
  python3 run_pipeline.py --stt-mode vi-first
  export ONEVOICE_MT_BEAM=1                 # dịch nhanh hơn ~25%
  export ONEVOICE_MT_THREADS=4              # bằng số core vật lý
  ```
- Beam 1 dịch câu ngắn kém hơn rõ rệt (xem [TRANSLATE_GUIDE.md](TRANSLATE_GUIDE.md)
  mục 4), nên hạ beam là bước cuối chứ không phải bước đầu.

### GPU có dùng được không?

Wheel `ctranslate2` cho aarch64 trên PyPI là **bản CPU-only**. Kiểm chứng trên
máy của bạn:

```bash
python3 -c "import ctranslate2; print(ctranslate2.get_supported_compute_types('cuda'))"
```

Lệnh trên báo lỗi hoặc trả tập rỗng ⇒ chỉ chạy CPU được. Muốn dùng GPU Jetson thì
phải tự build CTranslate2 với CUDA. **Hãy đo tốc độ CPU trước** — nếu đã đủ nhanh
thì đừng bỏ công build, và GPU nên để dành cho STT (tầng nặng hơn nhiều).

---

## 🎙️ 5. Micro

```bash
python3 run_pipeline.py --list-devices
python3 run_pipeline.py --device 1
```

Module VAD tự chọn kênh mạnh nhất và tự resample về 16 kHz nên đa số micro USB
cắm vào là chạy. Nếu VAD không nhận tiếng nói:

```bash
arecord -l                                    # card có được nhận không
arecord -d 3 -f cd test.wav && aplay test.wav # thu thử
python3 run_pipeline.py --vad-threshold 0.25  # hạ ngưỡng nếu mic yếu
```

---

## 🚀 6. Chạy

```bash
# Không cần micro — kiểm tra tầng dịch + hiển thị trước
python3 run_pipeline.py --offline

# Chạy thật
python3 run_pipeline.py

# Chế độ hướng dẫn viên: biết trước người nói tiếng Hoa
python3 run_pipeline.py --force-lang zh
```

### Chạy tự động khi khởi động (systemd)

```ini
# /etc/systemd/system/onevoice.service
[Unit]
Description=OneVoice translator
After=sound.target

[Service]
Type=simple
User=jetson
WorkingDirectory=/home/jetson/song-lo-onevoice
Environment=ONEVOICE_MT_THREADS=4
ExecStart=/home/jetson/song-lo-onevoice/.venv/bin/python3 run_pipeline.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now onevoice
journalctl -u onevoice -f
```

---

## 💾 7. Giữ tuổi thọ thẻ nhớ

- `save_wav_debug` mặc định đã **tắt** trong `PipelineConfig` — Segmenter đẩy
  audio qua RAM, không ghi file. Chỉ bật `--save-wav` khi đang debug.
- `--log-jsonl` cũng ghi đĩa liên tục; chỉ dùng khi thu thập dữ liệu đánh giá.
- Bật swap trên Nano 4 GB nếu chạy chung với STT lớn:
  ```bash
  sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile
  sudo mkswap /swapfile && sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
  ```

---

## 🩺 8. Sự cố thường gặp

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `FileNotFoundError: models/mt/...` | chưa copy model dịch | `python3 scripts/prepare_model.py --source <dir>` |
| `FileNotFoundError: models/stt/...` | chưa tải model STT | `python3 scripts/download_stt.py` |
| `ImportError: sherpa_onnx` | thiếu thư viện STT | `pip install sherpa-onnx` |
| Nhận nhầm tiếng Việt thành tiếng Hoa | câu quá ngắn, định tuyến kém chắc | `--force-lang`, xem [STT_GUIDE.md](STT_GUIDE.md) mục 2 |
| Thống kê báo nhiều câu "bỏ (không đủ tiếng nói)" | mic yếu / ồn nền | tăng gain mic, `--vad-threshold 0.45` |
| Dịch ra chuỗi Hán tự vô nghĩa | ghép nhầm model với tokenizer | `python3 scripts/prepare_model.py --verify-only` |
| `Illegal instruction (core dumped)` | wheel không đúng kiến trúc | cài lại `ctranslate2` cho aarch64, hoặc build từ nguồn |
| Thống kê báo nhiều câu "bỏ do quá tải" | STT+dịch chậm hơn tốc độ nói | hạ `ONEVOICE_MT_BEAM=1`, giảm `max_segment_duration` |
| Bản dịch thiếu hẳn một từ | ký tự nằm ngoài vocab | xem cảnh báo `⚠️ Model không đọc được ký tự`, thêm luật vào `onevoice/data/zh_unk_map.json` |
| Không có âm thanh | thiếu portaudio | `sudo apt install portaudio19-dev` rồi cài lại `sounddevice` |
