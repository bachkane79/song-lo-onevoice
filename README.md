# 🌏 OneVoice — Thiết bị phiên dịch song ngữ Việt ⇄ Hoa

Phiên dịch hội thoại thời gian thực, **chạy hoàn toàn offline** trên Jetson.
Không gọi API, không cần mạng.

```
🎙️ Mic ──► [1] VAD ──► [2] Segmenter ──► [3] STT ──► [4] Translate ──► [5] Display
            Silero      Energy Valley    SenseVoice/    NLLB-600M       màn hình
            ONNX        3 lớp ngắt câu   Zipformer      CT2 int8
```

| Module | Trạng thái | File |
|---|---|---|
| 1. VAD Detector | ✅ | [`VAD-Segmenter/VAD_process.py`](VAD-Segmenter/VAD_process.py) |
| 2. Audio Segmenter | ✅ | [`VAD-Segmenter/Segmenter_process.py`](VAD-Segmenter/Segmenter_process.py) |
| 3. STT song ngữ | ✅ | [`onevoice/stt_sherpa.py`](onevoice/stt_sherpa.py) |
| 4. Translate zh ⇄ vi | ✅ | [`onevoice/translate.py`](onevoice/translate.py) |
| 5. Display | ✅ | [`onevoice/display.py`](onevoice/display.py) |
| Điều phối | ✅ | [`onevoice/pipeline.py`](onevoice/pipeline.py) |

---

## 🚀 Chạy trong 2 lệnh

```bash
pip install -r requirements.txt
python scripts/setup.py          # tự tải toàn bộ model (~655 MB) + kiểm tra + chạy thử

python run_pipeline.py           # xong — nói vào micro
```

`setup.py` tải STT từ Hugging Face và model dịch từ GitHub Release của repo này,
**kiểm tra sha256 từng file** rồi dịch thử một câu mỗi chiều. Chạy lại nhiều lần
vô hại — file nào đã đúng thì bỏ qua.

Kiểm tra từng phần khi chưa có micro:

```bash
python run_pipeline.py --wav a.wav b.wav --show-route  # STT + dịch trên file thật
python run_pipeline.py --offline                       # chỉ tầng dịch + hiển thị
python scripts/setup.py --check                        # kiểm tra không tải lại
```

Trên Jetson: đọc [docs/JETSON_SETUP.md](docs/JETSON_SETUP.md) trước — bản Jetson
Nano đời cũ (Python 3.6) cần xử lý riêng.

---

## 🎤 STT — hai model, tự định tuyến

| | SenseVoice | Zipformer-30M-RNNT |
|---|---|---|
| Tiếng | zh (+en/ja/ko/yue) | **vi** |
| int8 | 229 MB | **29 MB** |
| RTF (x86, 2 thread) | 0.042 | 0.021 |

Không model nào làm được cả hai tiếng nên mỗi câu phải chọn model. **LID có sẵn
của SenseVoice không dùng được** — nó không biết tiếng Việt nên báo `<|zh|>` cho
8/8 câu tiếng Việt. Thay vào đó dùng chính sản lượng của Zipformer làm tín hiệu:
gặp audio tiếng Hoa nó gần như không xuất token nào (0.00–0.13 từ/giây) còn gặp
tiếng Việt thì ra câu đầy đủ (1.94–2.74 từ/giây).

Độ chính xác định tuyến đo được: **16/16 (100%) trên câu đầy đủ**, **97.8% trên
249 mẫu** gồm cả đoạn ngắn 1.3 s, **0 lỗi ở chiều tiếng Việt**.

Biết trước ai đang nói thì bỏ hẳn định tuyến — nhanh hơn và chính xác tuyệt đối:

```bash
python run_pipeline.py --force-lang zh     # chế độ "khách Trung Quốc nói"
```

Chi tiết: [docs/STT_GUIDE.md](docs/STT_GUIDE.md).

> ⚖️ Zipformer tiếng Việt phát hành theo **CC BY-NC-ND 4.0** (phi thương mại,
> không phái sinh). Đồ án/nghiên cứu thì được; bán thiết bị thì phải xin phép
> hoặc đổi model.

---

## 🤖 Máy dịch — NLLB-600M vocab-trim

394 MB, ~250 ms/câu (i9, beam 4), **một model làm cả hai chiều**. Runtime chỉ cần
`ctranslate2` + `tokenizers`, không cần torch/transformers.

```python
from onevoice import NllbTranslator

tr = NllbTranslator()
tr.translate("这个多少钱？", "zh").target          # → Đây là bao nhiêu tiền?
tr.translate("Tôi muốn đi Đà Nẵng.", "vi").target  # → 我想去岘港.
```

> ⚠️ **Đọc [docs/TRANSLATE_GUIDE.md](docs/TRANSLATE_GUIDE.md) mục 5 trước khi
> ship.** Vocab NLLB thiếu 11 ký tự Hán rất thường dùng trong hội thoại du lịch
> (厕 咸 啤 汤 虾 蟹 辣 酱 醋 鸡 烫). Repo đã vá cho chiều zh→vi và có bảng địa danh
> Hán-Việt cho chiều vi→zh, nhưng **chiều vi→zh vẫn còn hạn chế không vá được**.

---

## 📁 Cấu trúc

```
VAD-Segmenter/          Module 1 & 2 (giữ nguyên, không sửa)
onevoice/
  config.py             mọi tham số + override bằng biến môi trường
  stt.py                interface STT + engine giả lập + registry
  stt_sherpa.py         SenseVoice + Zipformer + bộ định tuyến ngôn ngữ
  translate.py          máy dịch NLLB
  langid.py             nhận diện zh/vi theo hệ chữ viết (dự phòng)
  glossary.py           địa danh Hán-Việt, thay ở đầu vào (vi→zh)
  zh_norm.py            vá ký tự ngoài vocab (zh→vi)
  display.py            hiển thị
  pipeline.py           điều phối, worker thread, chính sách quá tải
  data/                 bảng địa danh + bảng chuẩn hóa <unk>
scripts/
  download_stt.py       tải model STT
  prepare_model.py      copy + verify model dịch
  bench_stt.py          đo RTF + chấm điểm định tuyến ngôn ngữ
  bench_translate.py    đo tốc độ dịch trên thiết bị thật
  audit_vocab.py        tìm ký tự model không đọc được
  probe_glossary.py     đo model dịch sai địa danh thành gì
tests/                  78 test, tự bỏ qua phần cần model
docs/
  STT_GUIDE.md          STT + luật định tuyến ngôn ngữ
  TRANSLATE_GUIDE.md    máy dịch + hạn chế đã biết
  JETSON_SETUP.md       cài đặt và nhúng lên Jetson
models/                 model (KHÔNG commit — xem .gitignore)
```

Tổng dung lượng model trên thiết bị: **~652 MB** (STT 258 + dịch 394).
RAM lúc chạy: ~1.3 GB.

---

## 🔧 Thay module

Mọi tầng đều nói chuyện qua interface nhỏ, đổi một tầng không ảnh hưởng tầng khác:

```python
# STT khác
class MySttEngine(SttEngine):
    def transcribe(self, audio, sample_rate):   # float32 16 kHz
        return SttResult(text="...", lang="vi")

# Màn hình thật thay vì terminal
class LcdDisplay(Display):
    def show(self, event):
        my_screen.draw(event.source_text, event.target_text)

OneVoicePipeline(stt_engine=MySttEngine(), display=LcdDisplay()).run()
```

---

## 🧪 Kiểm thử

```bash
pytest -q                          # toàn bộ
pytest -q -m "not needs_model"     # chỉ logic, không cần model
python scripts/bench_stt.py --wav-dir my_audio/
python scripts/audit_vocab.py --check-rules
```
