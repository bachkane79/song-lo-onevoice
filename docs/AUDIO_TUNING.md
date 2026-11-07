# Micro, VAD và cắt câu (Sweetspot Architecture)

Mọi con số trong file này là **số đo trên máy thật**, không phải số đoán. Đo lại
bằng `python scripts/diag_vad.py` (hoặc `python scripts/diag_vad.py --mic-only` khi đeo tai nghe)
trước khi chỉnh bất cứ tham số nào.

---

## 1. Micro tự nhận

`python run_pipeline.py` không cần `--device`. [`onevoice/mic.py`](../onevoice/mic.py)
**mở thử** từng thiết bị rồi mới chọn, theo thứ tự:

1. `--device N` người dùng chỉ định — mở không được thì **báo lỗi**, không tự đổi
   sang cái khác (gõ `--device 5` là có ý; im lặng chạy device khác sẽ khiến
   người dùng tưởng mic đó hỏng).
2. Thiết bị mặc định của hệ điều hành.
3. Mic thật còn lại (tên có `microphone` / `headset` / `array`...).
4. Cuối cùng mới tới cổng ảo (Stereo Mix, Sound Mapper, loopback) — thu trúng
   tiếng loa thì thiết bị sẽ **dịch lại chính bản dịch nó vừa phát ra**.

Vì sao phải mở thử: trên máy dev Windows, **8/13 thiết bị "có mặt" nhưng mở
không được** (Stereo Mix bị tắt, mic bị Zoom/Teams chiếm, card đã rút). Nếu chỉ
đọc `sd.default.device[0]` như `VAD_process` làm thì lỗi chỉ lộ ra sau khi đã nạp
655 MB model.

```bash
python run_pipeline.py --list-devices   # ✓ = đã mở thử thành công, ✗ = không dùng được
```

---

## 2. Log gọn

Mặc định log chỉ còn **lượt VAD + 2 dòng mỗi câu**:

```
🎙️  ▶ Lượt 1 — có tiếng nói
🇻🇳 STT    Cứ nghĩ chạy trốn khỏi thành phố là sẽ yên bình nhưng hóa ra
🇨🇳 DỊCH   你以为逃离这座城市是安全的,但事实证明,   #001 · 3.4s nói · 0.25s xử lý (STT 117ms + dịch 132ms)
🔇  ■ Lượt 1 — im lặng, hết lượt (6.7s nói)
🇻🇳 STT    Thứ bám theo tôi dai dẳng nhất lại chính là sự im lặng đáng sợ này
🇨🇳 DỊCH   我最坚持的是这种可怕的沉默.   #002 · 3.3s nói · 0.54s xử lý (STT 112ms + dịch 424ms)
```

`VAD-Segmenter/` in visualizer **31 dòng/giây** cộng 5 dòng mỗi lần chốt câu —
phụ đề bị đẩy trôi khỏi màn hình trước khi kịp đọc. [`onevoice/vadlog.py`](../onevoice/vadlog.py)
xử lý từ phía ngoài: tạm thay `sys.stdout` trong lúc VAD chạy và nuốt **đúng những dòng ồn đã biết**.
Lỗi luồng âm thanh và cảnh báo overflow vẫn cho qua — mất cảnh báo mic là mất
manh mối gỡ lỗi quan trọng nhất của thiết bị này.

`--verbose` trả lại log gốc. `--layout full` dùng bố cục cũ (có đường kẻ).

> **Số lượt ≠ số câu.** Một lượt nói dài có thể được Segmenter tự động tách tại nhịp thở
> (Prob-Dip Valley) thành 2 vế câu hoàn chỉnh, nên thấy "Lượt 1" ra hai câu `#001` và `#002` là hoàn toàn chuẩn xác.
>
> **Dòng "hết lượt" đôi khi in sau dòng DỊCH.** Worker dịch xong vế đầu ngay trong lúc người dùng
> đang nói vế thứ hai. Đúng bản chất chạy song song — ép tuần tự sẽ chặn luồng VAD.

---

## 3. Đo lường thực nghiệm & Kiến trúc cắt câu Sweetspot (Stage 3)

### 3.1. Hai chế độ đo của `scripts/diag_vad.py`

Công cụ [`scripts/diag_vad.py`](../scripts/diag_vad.py) hỗ trợ 2 chế độ:
- `python scripts/diag_vad.py`: Phát file mẫu `zh.wav` ra loa ngoài ở Giai đoạn 2 (dùng khi dùng loa ngoài + mic rời).
- `python scripts/diag_vad.py --mic-only`: Người dùng **tự nói trực tiếp vào micro** trong 6 giây ở Giai đoạn 2 (**bắt buộc dùng khi đeo tai nghe / Headset** hoặc dùng mic điện thoại qua SoundWire, vì nếu phát qua tai nghe thì micro sẽ không nghe thấy tiếng loa).

### 3.2. Số đo thực tế trên hệ thống (Sau khi nâng cấp DSP & VAD Stage 3)

Kết quả đo bằng `python scripts/diag_vad.py --mic-only` trên `Microphone Array on SoundWire D`:

| Chỉ số đo (189 frames = 6.05s) | [1] Phòng yên | [2] Đang nói | Độ phân tách (Tỉ lệ Nói / Im) |
|---|---|---|---|
| **VAD prob (median / p90)** | `0.001` / `0.002` | **`1.000`** / `1.000` | **Gấp 1000 lần** (`1.000` vs `0.001`) |
| **Năng lượng RMS (median / p90)** | `0.00002` / `0.00002` | **`0.09085`** / `0.18272` | **Gấp 6011 lần** (`0.09085` vs `0.00002`) |
| **Tỉ lệ frame `is_speech`** | **`0.0%`** (không nhiễu giả) | **`83.1%`** | Tách biệt hoàn toàn |
| **Chuỗi im dài nhất giữa câu** | `6.05s` | **`0.22s`** | Cận dưới an toàn của `silence_timeout` |

### 3.3. Cách kiến trúc Sweetspot (Stage 3) giải quyết bài toán Độ nhạy vs Ngữ cảnh

Trước đây, hệ thống gặp 2 vấn đề lớn:
1. **Energy Valley cũ xung đột với AGC**: Khi AGC khuếch đại biên độ về `0.30`, việc so sánh `energy < valley_ratio * peak` dễ cắt vụn câu nói ở các âm tiết nhẹ hoặc bị vô hiệu hóa hoàn toàn.
2. **Silence Timeout cố định (`0.7s`)**: Nếu đặt ngắn thì cắt cụt câu khi người nói ngập ngừng ở đầu câu; nếu đặt dài thì câu dài bị trễ phụ đề và nối dồn thành khối lớn.

Kiến trúc **Sweetspot Stage 3** trong [`VAD_process.py`](../VAD-Segmenter/VAD_process.py) và [`Segmenter_process.py`](../VAD-Segmenter/Segmenter_process.py) giải quyết triệt để qua 4 cơ chế phối hợp:

1. **DSP Sạch & Không Rè (`VAD_process.py`)**:
   - Resample với `padtype='line'` (triệt tiêu gai nhiễu 38% ở biên mỗi khung 32ms).
   - **Soft Noise Gate (`_NOISE_FLOOR = 0.005`)** + **Sample-by-Sample Linear Ramp AGC**: Không khuếch đại nhiễu nền lúc phòng yên (giữ năng lượng phòng yên ở `0.00002`, đạt tỉ lệ Nói/Im gấp **6011 lần**) và nội suy tuyến tính hệ số khuếch đại trên từng mẫu âm thanh.
   - **Context-Holding Hysteresis (`start = 0.35`, `end = 0.22` + Hangover `96ms`)**: Vào câu ngay lập tức ở khung đầu tiên (`>= 0.35`), nhưng khi đang nói thì giữ mạch câu qua các âm tiết hạ giọng (`>= 0.22`) kèm 3 khung đệm (`96ms`), giúp khoảng lặng lọt giữa câu khi nói liên tục chỉ còn tối đa **0.22s**.

2. **Adaptive Silence Timeout (Lớp 1 — Co giãn động theo độ dài câu)**:
   - Khi mới bắt đầu nói (`< 1.5s`): Chờ kiên nhẫn **`0.70s`** để người nói thoải mái ngập ngừng đầu câu mà không bị cắt cụt.
   - Khi câu vào vùng **Context Vàng (`1.5s – 4.2s`)**: Ngưỡng chờ giảm mượt từ `0.70s` về **`0.52s`** (vẫn cao hơn gấp đôi mức nghỉ giữa câu `0.22s`).
   - Khi câu đã dài (`>= 4.2s`): Ngưỡng chờ giảm mượt về **`0.38s`**, giúp chốt câu cực nhạy ngay khi người nói dứt ý.

3. **Pure Prob-Dip Valley Detection (Lớp 2 — Tách vế câu ghép tự nhiên)**:
   - Thay vì dùng Energy Valley cũ bị ảnh hưởng bởi AGC, Stage 3 dùng **Prob-Dip Valley**: Chỉ kích hoạt khi câu nói liên tục đã đủ dài (`>= 3.0s`), phát hiện nhịp nghỉ lấy hơi thật sự khi `prob < 0.24` kéo dài `128–160ms` (4–5 frames) và xác nhận người nói bắt đầu vế câu mới (`3` frames liên tiếp `prob >= 0.45`).
   - **Kết quả thực nghiệm**: Một câu ghép dài `6.7s` (*"Cứ nghĩ chạy trốn khỏi thành phố là sẽ yên bình nhưng hóa ra / Thứ bám theo tôi dai dẳng nhất lại chính là sự im lặng đáng sợ này"*) được tự động tách thành 2 vế hoàn chỉnh `3.39s` và `3.30s` ngay tại nhịp nghỉ giữa hai mệnh đề, không mất bất kỳ âm tiết nào.

4. **Acoustic Safety Gate & Smart Max-Duration Soft-Cut**:
   - **Acoustic Safety Gate**: Mọi mẩu âm thanh ngắn `< 0.45s` có độ tự tin thấp (`mean_prob < 0.55`) bị chặn ngay tại gốc Segmenter và hoàn trả các frame cuối vào `pre_ring` (160ms) để ghép vào câu kế tiếp.
   - **Smart Soft-Cut (Lớp 3)**: Nếu người nói đọc liên tục chạm trần `8.0s`, Segmenter tự động quét cửa sổ `2.0s` cuối để tìm ranh giới âm tiết có xác suất VAD thấp nhất (`prob < 0.65`) và cắt mềm tại đó thay vì bổ ngang thân từ.

---

## 4. Phòng ồn — đòn bẩy khác hẳn

Khi ở môi trường cực ồn (quán ăn đông người, tiếng nói chuyện sát bên cạnh vượt quá `_NOISE_FLOOR = 0.005`), VAD có thể nhận diện nhầm tiếng người xung quanh là tiếng nói của bạn (`prob` lúc im lặng dâng cao `> 0.60`).

Khi đó, nâng ngưỡng VAD bằng cờ dòng lệnh:

```bash
python run_pipeline.py --vad-threshold 0.75
```

Mặc định giữ ở **0.35** để bắt nhạy cả giọng nói nhỏ trong phòng bình thường.

---

## 5. Bảng tra nhanh

| Triệu chứng | Nguyên nhân thường gặp | Xử lý |
|---|---|---|
| Đo `diag_vad.py` thấy Giai đoạn 2 `is_speech: 0.0%` | Đang đeo tai nghe hoặc dùng mic từ xa nên mic không thu được tiếng loa mẫu | Chạy `python scripts/diag_vad.py --mic-only` và tự nói vào mic |
| Toàn ra khối 8s hoặc tự dịch khi không nói | Phòng quá ồn hoặc có tiếng TV/người nói sát mic | Nâng `--vad-threshold 0.70` (hoặc `0.85` nếu cực ồn) |
| Muốn thử nghiệm độ nhạy cắt câu khác nhau | Chuyển đổi giữa 3 giai đoạn của Segmenter | Đặt biến môi trường `ONEVOICE_VAD_STAGE=1`, `2`, hoặc `3` (mặc định `3`) |
| Mất câu trả lời đơn âm tiết cực ngắn ("Ừ", "是") | `min_utterance_s` lọc các mẩu quá ngắn | Hạ `--min-utterance 0.25` |
| Phụ đề trễ khi đọc chậm | `silence_timeout` cơ sở cao | Hạ `--silence-timeout 0.5` |
| `📊 ... bỏ do quá tải` | STT+MT chậm hơn tốc độ nói | Đặt `ONEVOICE_MT_BEAM=1` |

Tham số Segmenter đặt ở [`onevoice/config.py`](../onevoice/config.py), kèm lý do
từng con số. Mọi thứ đều override được bằng cờ dòng lệnh —
`python run_pipeline.py --help`, nhóm **"Cắt câu (Segmenter)"**.
