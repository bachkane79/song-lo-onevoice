# 🎤 Hướng Dẫn Kỹ Thuật: STT Song Ngữ (Module 3)

Tầng nhận dạng giọng nói trong pipeline. Xem
[VAD_SEGMENTER_GUIDE.md](../VAD-Segmenter/VAD_SEGMENTER_GUIDE.md) cho Module 1 & 2,
[TRANSLATE_GUIDE.md](TRANSLATE_GUIDE.md) cho máy dịch.

---

## 🏗️ 1. Hai model, một bộ định tuyến

```
Segmenter ──audio float32 16kHz──►  BilingualSttEngine
                                          │
                            ┌─────────────┴─────────────┐
                            ▼                           ▼
                   Zipformer-30M (vi)          SenseVoice (zh)
                       29 MB, RTF 0.021         229 MB, RTF 0.042
                            └─────────────┬─────────────┘
                                          ▼
                              luật định tuyến → text + lang
                                          │
                                          ▼
                                     Translate
```

| | SenseVoice | Zipformer-30M-RNNT-6000h |
|---|---|---|
| Tiếng | zh, en, ja, ko, yue | **vi** |
| Kích thước int8 | 229 MB | **29 MB** |
| Kiến trúc | non-autoregressive | RNN-Transducer |
| RTF (x86, 2 thread) | 0.042 | 0.021 |
| Nguồn | `csukuangfj/sherpa-onnx-sense-voice-...` | `hynt/Zipformer-30M-RNNT-6000h` |
| Giấy phép | Apache-2.0 | ⚠️ **CC BY-NC-ND 4.0** |

Cả hai chạy trên **sherpa-onnx** — cùng runtime ONNX với Silero VAD ở Module 1,
nên không phải thêm backend nào.

> ⚖️ **Giấy phép cần chú ý**: Zipformer tiếng Việt là **CC BY-NC-ND 4.0** — phi
> thương mại và không phái sinh. Dùng cho đồ án/nghiên cứu thì được; nếu định bán
> thiết bị thì phải xin phép tác giả hoặc đổi sang model tiếng Việt khác
> (PhoWhisper, Whisper fine-tune, wav2vec2-vi...).

---

## 🧭 2. Định tuyến ngôn ngữ — phần khó nhất

Không model nào làm được cả hai tiếng, nên mỗi câu phải quyết định gửi vào đâu.

### ❌ Cách hiển nhiên nhưng SAI: dùng LID của SenseVoice

SenseVoice có sẵn nhận diện ngôn ngữ, trả ở `result.lang`. Nhưng nó chỉ được
huấn luyện trên zh/en/ja/ko/yue — **không có tiếng Việt**. Đo trên 8 câu tiếng
Việt FLEURS:

```
audio tiếng Việt → SenseVoice báo <|zh|>  ở 8/8 câu
```

Nó không có lựa chọn "vi" để mà báo. Dùng tín hiệu này là hỏng ngay.

### ✅ Cách đang dùng: lấy chính sản lượng của Zipformer làm tín hiệu

Zipformer là RNN-Transducer chỉ học tiếng Việt. Gặp audio tiếng Hoa, nó không tìm
được gì khớp nên **gần như không xuất token nào**:

| audio | Zipformer xuất ra |
|---|---|
| 🇨🇳 `这并不是告别...` (10.4s) | *(rỗng)* |
| 🇨🇳 `桥下垂直净空15米...` (13.9s) | `Chào` |
| 🇻🇳 `Văn hóa và bộ lạc cổ xưa...` (11.3s) | câu đầy đủ, gần đúng hoàn toàn |

Quy ra **số từ trên mỗi giây audio**: tiếng Hoa **0.00–0.13**, tiếng Việt
**1.94–2.74** — cách nhau ~17 lần.

### Luật cuối cùng

```python
vi_rate = số_từ(Zipformer)      / độ_dài_audio
zh_rate = số_Hán_tự(SenseVoice) / độ_dài_audio

nếu max(vi_rate, zh_rate) < 0.8:       →  bỏ câu (không đủ tiếng nói)
nếu vi_rate > 0.3 * zh_rate + 0.8:     →  tiếng Việt
ngược lại:                             →  tiếng Hoa
```

Ba hằng số `ALPHA=0.3`, `MARGIN=0.8`, `FLOOR=0.8` hiệu chỉnh trên 249 mẫu
(16 câu FLEURS đầy đủ + 233 đoạn cắt 1.3–6.0s). **Sửa xong phải đo lại** bằng
`python scripts/bench_stt.py --wav-dir <dir> --routing`.

### Độ chính xác đo được

| bộ mẫu | đúng | sai | bỏ |
|---|---|---|---|
| Câu đầy đủ 7–22s (FLEURS) | **16/16 — 100%** | 0 | 0 |
| Đoạn cắt ngắn 1.3–6.0s | 205 | 5 | 23 |
| **Tổng 249 mẫu** | **221 — 97.8%** | 5 | 23 |

**0 lỗi ở chiều tiếng Việt.** Cả 5 lỗi đều là đoạn tiếng Hoa **rất ngắn**
(1.3–2.0s) mà Zipformer "nghe nhầm" ra vài từ tiếng Việt. Câu càng dài định tuyến
càng chắc — mà Segmenter đã đặt `min_segment_duration = 1.2s` và thường cắt ở
2–4s, nên phần lớn câu thật nằm ở vùng an toàn.

### Vì sao có nhánh "bỏ câu"

Khi cả hai model đều gần như không nghe ra gì, câu đó gần chắc là tiếng ồn (mic
va chạm, tiếng động nền vượt VAD). Lúc đó **hiển thị một bản dịch bịa còn tệ hơn
là không hiển thị gì** — người dùng không có cách nào biết máy đang bịa. Pipeline
đếm riêng số câu bị bỏ và in ra khi dừng.

### Khi biết trước ai đang nói → dùng `--force-lang`

```bash
python run_pipeline.py --force-lang zh    # chế độ "khách Trung Quốc nói"
python run_pipeline.py --force-lang vi    # chế độ "hướng dẫn viên nói"
```

Lúc này định tuyến **chính xác tuyệt đối**, chỉ nạp **một** model (tiết kiệm
229 MB RAM nếu ép `vi`) và chỉ chạy một lần suy luận. Nếu thiết bị có nút bấm
hai chiều thì đây là chế độ nên dùng.

---

## ⚡ 3. Hiệu năng

Đo trên i9-9900K, 2 thread, 16 câu FLEURS (201 giây audio):

| cấu hình | RTF | ý nghĩa với câu 4 giây |
|---|---|---|
| SenseVoice (zh) | 0.042 | 0.17 s |
| Zipformer (vi) | 0.021 | 0.08 s |
| **Song ngữ `dual`** | **0.060** | **0.24 s** |

Cộng thêm máy dịch (~0.25 s/câu) → tổng ~0.5 s cho một câu 4 giây trên x86.

**Jetson chậm hơn nhiều lần — phải đo lại**:

```bash
python3 scripts/bench_stt.py --wav-dir my_audio/
python3 scripts/bench_translate.py --sweep-beam
```

Nếu không kịp, theo thứ tự nên thử:

1. `--force-lang zh|vi` — bỏ hẳn một model, nhanh gần gấp đôi
2. `--stt-mode vi-first` — bỏ qua SenseVoice khi Zipformer đã chắc chắn là tiếng
   Việt (`vi_rate ≥ 3.0`). Có rủi ro: câu tiếng Hoa ngắn bị nhận nhầm sẽ không
   còn được SenseVoice "cứu".
3. `--stt-threads 4` — nếu còn core rảnh
4. `ONEVOICE_MT_BEAM=1` — hạ chất lượng dịch để đổi tốc độ

---

## 🔌 4. Dùng trong code

```python
from onevoice.stt_sherpa import BilingualSttEngine

stt = BilingualSttEngine(mode="dual", num_threads=2)
stt.warmup()

res = stt.transcribe(audio_float32_16k, 16000)
print(res.lang, res.text)      # "zh" / "vi" / None (bỏ)
print(res.detail)              # {'route':..., 'vi_rate':..., 'zh_rate':...}
```

Chỉ dùng một tiếng:

```python
from onevoice.stt_sherpa import SenseVoiceSttEngine, ZipformerViSttEngine
zh = SenseVoiceSttEngine(language="zh")     # ép tiếng Hoa, chính xác hơn
vi = ZipformerViSttEngine()
```

### Thay bằng model khác

Giữ nguyên interface `SttEngine.transcribe(audio, sample_rate) -> SttResult`
(xem `onevoice/stt.py`) rồi đăng ký vào `_REGISTRY`. Pipeline không cần sửa gì.

Có thể cắm engine mới vào đúng một chiều mà giữ chiều còn lại:

```python
BilingualSttEngine(vi_engine=MyBetterVietnameseStt(), mode="dual")
```

---

## 🚨 5. Hạn chế đã biết

**5.1. Zipformer xuất chữ IN HOA, không có dấu câu.** Repo tự chuyển về chữ
thường + hoa chữ đầu (`_tidy_vi`) nhưng **không tự thêm dấu câu** — đoán sai dấu
câu sẽ đổi nghĩa và làm máy dịch sai theo. Máy dịch NLLB xử lý câu không dấu câu
vẫn ổn.

**5.2. Từ mượn / tên riêng nước ngoài bị phiên âm.** `springboks` → `spring book`,
`t rex` → `tivi`, `catalunya` → `cataludia`. Đây là hạn chế chung của ASR tiếng
Việt, không riêng model này.

**5.3. Câu rất ngắn (< 2s) định tuyến kém chắc chắn.** Xem mục 2. Dùng
`--force-lang` nếu tình huống sử dụng có nhiều câu ngắn (mặc cả giá, hỏi đường).

**5.4. SenseVoice nhận cả en/ja/ko.** Nếu có người nói tiếng Anh, SenseVoice sẽ
nhận ra tiếng Anh và máy dịch NLLB sẽ được đưa mã ngôn ngữ `zh` sai. Hiện pipeline
không xử lý trường hợp này — nằm ngoài phạm vi thiết bị Việt-Hoa.

**5.5. Hai model cùng nằm trong RAM.** SenseVoice 229 MB + Zipformer 29 MB +
máy dịch ~970 MB ≈ **1.3 GB**. Vừa với Jetson 4 GB, chật với bản 2 GB — lúc đó
nên dùng `--force-lang` để chỉ nạp một model STT.

---

## 🧪 6. Công cụ

| lệnh | dùng để |
|---|---|
| `python scripts/download_stt.py` | tải + kiểm tra + xóa bản fp32 thừa |
| `python scripts/download_stt.py --verify-only` | kiểm tra lại sau khi copy sang thiết bị |
| `python scripts/bench_stt.py --wav-dir DIR` | đo RTF + chấm điểm định tuyến |
| `python run_pipeline.py --wav a.wav b.wav --show-route` | chạy thật một vài file, xem điểm định tuyến |
| `pytest tests/test_stt.py -v` | test luật định tuyến (không cần model) |

Đặt tên file `zh_*.wav` / `vi_*.wav` thì `bench_stt.py` tự chấm điểm định tuyến
được.
