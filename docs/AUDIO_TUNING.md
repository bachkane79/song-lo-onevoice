# Micro, VAD và cắt câu

Mọi con số trong file này là **số đo trên máy thật**, không phải số đoán. Đo lại
bằng `python scripts/diag_vad.py` trước khi chỉnh bất cứ tham số nào.

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
🎙️  ▶ Lượt 3 — có tiếng nói
🔇  ■ Lượt 3 — im lặng, hết lượt (2.1s nói)
🇨🇳 STT    开饭时间早上9点至下午5点。
🇻🇳 DỊCH   Bữa sáng 9h đến 5h.   #003 · 2.4s nói · 0.31s xử lý (STT 162ms + dịch 150ms)
```

`VAD-Segmenter/` in visualizer **31 dòng/giây** cộng 5 dòng mỗi lần chốt câu —
phụ đề bị đẩy trôi khỏi màn hình trước khi kịp đọc. Quy tắc repo là không sửa
thư mục đó, nên [`onevoice/vadlog.py`](../onevoice/vadlog.py) xử lý từ phía ngoài:
tạm thay `sys.stdout` trong lúc VAD chạy và nuốt **đúng những dòng ồn đã biết**.
Lỗi luồng âm thanh và cảnh báo overflow vẫn cho qua — mất cảnh báo mic là mất
manh mối gỡ lỗi quan trọng nhất của thiết bị này.

`--verbose` trả lại log gốc. `--layout full` dùng bố cục cũ (có đường kẻ).

> **Số lượt ≠ số câu.** Một lượt nói có thể bị Segmenter cắt thành nhiều câu, nên
> hai con số đánh riêng. Thấy "Lượt 3" ra hai câu là đúng.
>
> **Dòng "hết lượt" đôi khi in sau dòng DỊCH.** Worker dịch xong trước khi hết
> `silence_timeout`. Đúng bản chất chạy song song — ép tuần tự sẽ chặn luồng VAD.

---

## 3. Cắt câu — vì sao Energy Valley bị tắt mặc định

Đây là phần quan trọng nhất của file này.

### Số đo

`scripts/diag_vad.py` trên mic Realtek của máy dev, phòng yên:

| | phòng yên | đang nói |
|---|---|---|
| VAD prob (median) | 0.031 | 0.891 |
| **Năng lượng (median)** | **0.101** | **0.124** |
| Chuỗi im lặng dài nhất | 5.15s | **0.10s** |

### Kết luận 1 — Silence Timeout không phải thủ phạm

Khoảng lặng dài nhất **bên trong** một câu liền mạch chỉ **0.10s**. Mọi giá trị
`silence_timeout >= 0.4s` đều không thể cắt giữa câu. Nâng nó lên chỉ thêm độ
trễ, vì số này cộng **thẳng** vào độ trễ của mọi câu. Hiện đặt **0.7s** = 0.10s
đo được + biên cho người thật ngập ngừng (máy đọc thì đều, người thì không).

### Kết luận 2 — Energy Valley không dùng được trên phần cứng này

Tầng đó cắt khi `energy < valley_ratio * peak`, nên **chỉ có nghĩa khi năng lượng
lúc im thấp hơn hẳn lúc nói**. Ở đây hai số gần bằng nhau (0.101 vs 0.124) vì AGC
trong `VAD_process` kéo mọi thứ về biên độ mục tiêu 0.3 — đúng cái dải động mà
valley cần đã bị san phẳng.

Với `peak ≈ 0.40`:

| valley_ratio | ngưỡng cắt | hậu quả |
|---|---|---|
| 0.25 (mặc định gốc) | 0.100 | Nằm **ngay dưới** median lúc nói 0.124 → bắn liên tục giữa câu. **Đây là lỗi "cắt hết giữa câu".** |
| 0.12 | 0.056 | Dưới cả sàn ồn → không bao giờ bắn, mọi câu rơi xuống Max Duration 8s |

Không có giá trị nào ở giữa chạy được, vì **khoảng nghỉ thật và tiếng ồn nền cùng
nằm ở mức 0.10**. Nên tầng 2 tắt mặc định, dựa vào VAD silence — chỗ có độ tách
bạch rất tốt (0.031 vs 0.891).

Bật lại bằng `--energy-valley` nếu chạy trên mic khác hoặc sau khi AGC được
chỉnh — nhưng **đo lại trước**: `diag_vad.py` tự tính tỉ số năng lượng và nói
thẳng có nên bật hay không.

### Kết luận 3 — mẩu vụn còn lọt qua nhánh Silence Timeout

Nhánh Silence Timeout của Segmenter **không kiểm tra `min_segment_duration`** (số
đó chỉ chặn tầng Energy Valley). Khi valley cắt, nó giữ lại phần đuôi ~150ms làm
đầu câu kế; người nói im ngay sau đó thì phần đuôi ấy ra thành "câu" 0.2s. STT
nghe ra một thán từ vô nghĩa (`嗯。` → "Này."), máy dịch vẫn tốn 100ms dịch nó, và
nó chiếm một chỗ trong hàng đợi 4 slot của câu thật.

`OneVoicePipeline.on_segment_ready` bỏ mẩu ngắn hơn `min_utterance_s` (0.4s) và
đếm lại, in ra lúc dừng.

---

## 4. Phòng ồn — đòn bẩy khác hẳn

Hai lần đo cách nhau vài phút trên cùng máy:

| | phòng yên | phòng có người nói chuyện |
|---|---|---|
| VAD prob lúc "im" (median) | 0.031 | **0.649** |
| % frame bị coi là có tiếng | 19.4% | **79.8%** |

Khi phòng ồn, VAD **không bao giờ im** → `silence_timeout` không bắn → mọi câu
chạm trần `max_segment_duration` 8s. Triệu chứng ngược hẳn với cắt vụn, và
**chỉnh tham số Segmenter không giải quyết được** — phải nâng ngưỡng VAD:

```bash
python run_pipeline.py --vad-threshold 0.9
```

Đo thật trong phòng ồn: câu ra 1.1–4.8s, đúng dáng câu, hết cả khối 8s lẫn mẩu
0.2s.

Mặc định vẫn để **0.35** vì hai lần đo lệch nhau 20 lần — chọn 0.9 làm mặc định
sẽ nuốt mất người nói nhỏ trong phòng yên. Đây là số phụ thuộc môi trường, nên để
ở cờ chứ không đóng vào config.

---

## 5. Bảng tra nhanh

| Triệu chứng | Nguyên nhân thường gặp | Xử lý |
|---|---|---|
| Câu đứt giữa chừng | Energy Valley bật trên mic có AGC | Bỏ `--energy-valley` (mặc định đã tắt) |
| Toàn ra khối 8s, nội dung lộn xộn | Phòng ồn, VAD không bao giờ im | `--vad-threshold 0.9` |
| Nhiều câu 0.2–0.4s vô nghĩa | Mảnh vụn Segmenter nhả ra | Tăng `--min-utterance` |
| Mất câu trả lời ngắn thật ("Vâng", "是") | `min_utterance_s` quá cao | Hạ `--min-utterance 0.25` |
| Phụ đề trễ đều mọi câu | `silence_timeout` cao | Hạ `--silence-timeout 0.5` |
| `📊 ... bỏ do quá tải` | STT+MT chậm hơn tốc độ nói | `ONEVOICE_MT_BEAM=1` |

Tham số Segmenter đặt ở [`onevoice/config.py`](../onevoice/config.py), kèm lý do
từng con số. Mọi thứ đều override được bằng cờ dòng lệnh —
`python run_pipeline.py --help`, nhóm **"Cắt câu (Segmenter)"**.
