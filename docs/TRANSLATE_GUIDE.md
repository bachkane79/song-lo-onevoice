# 📖 Hướng Dẫn Kỹ Thuật: Máy Dịch zh ⇄ vi (Module 4)

Tài liệu về tầng **Translate** trong pipeline thiết bị phiên dịch du lịch Việt -
Hoa. Xem [VAD_SEGMENTER_GUIDE.md](../VAD-Segmenter/VAD_SEGMENTER_GUIDE.md) cho
Module 1 & 2, [JETSON_SETUP.md](JETSON_SETUP.md) cho phần nhúng thiết bị.

---

## 🏗️ 1. Vị trí trong pipeline

```
🎙️ Mic → [1] VAD → [2] Segmenter ──callback RAM──► [3] STT ──text──► [4] TRANSLATE ──► [5] Display
                                                    ▲                      ▲
                                            onevoice/stt.py        onevoice/translate.py
                                          (chỗ cắm cho STT)          (tài liệu này)
```

Điều phối nằm ở `onevoice/pipeline.py`. Điểm cần nhớ về luồng: Segmenter gọi
`on_segment_ready()` **ngay trong luồng xử lý VAD**, nên pipeline chỉ đẩy câu vào
hàng đợi rồi trả về ngay; STT + dịch chạy ở worker thread riêng. Chạy dịch trực
tiếp trong callback sẽ làm nghẽn luồng đọc micro và **mất câu nói tiếp theo**.

---

## 🤖 2. Model đang dùng

**NLLB-200-distilled-600M, đã cắt vocab riêng cho zh+vi, export CTranslate2 int8.**

| | giá trị |
|---|---|
| Disk | **394 MB** (`model.bin` 393.6 MB + vocab 0.8 MB) + tokenizer 2.5 MB |
| Peak RSS | ~970 MB – 1.07 GB |
| Tốc độ (i9-9900K, beam 4) | **247 ms/câu** trung bình, 458 ms câu dài nhất |
| Chiều dịch | **một model làm cả hai chiều** zh→vi và vi→zh |
| Phụ thuộc runtime | `ctranslate2` + `tokenizers` — **không cần torch/transformers** |

### Vì sao chọn model này

| model | chrF++ zh→vi | disk | ms/câu |
|---|---|---|---|
| **NLLB-600M vocab-trim** | **52.5** | **394 M** | 364 (beam1) |
| NLLB-600M gốc | 52.7 | 600 M | 561 (beam1) |
| Helsinki opus-mt-zh-vi | 37.2 | 77 M | 66 |

Opus-MT nhẹ hơn 5 lần nhưng **bịa số liệu và bỏ nội dung ở 27% câu** (dịch
`菜太咸了` "mặn quá" thành "ngon quá", `胡志明市` thành một tên người Hàn Quốc) —
không dùng được cho thiết bị mà người dùng không có cách nào kiểm chứng.

Việc cắt vocab lấy được **−34% disk, −35% latency** so với NLLB gốc mà chất lượng
gần như không đổi: 26/30 câu zh→vi và 28/30 câu vi→zh cho kết quả **giống hệt
từng ký tự**.

---

## 🔌 3. Cách dùng

### Dùng nhanh

```python
from onevoice import NllbTranslator

tr = NllbTranslator()
tr.warmup()                       # nạp sẵn lúc khởi động, tránh câu đầu bị chậm

print(tr.translate("这个多少钱？", "zh").target)
# → Đây là bao nhiêu tiền?

print(tr.translate("Cho tôi hai vé đi Hà Nội.", "vi").target)
# → 给我两张去河内的门票.
```

### Tự nhận diện chiều dịch

```python
res = tr.translate_auto(text, force_src_lang=None, stt_lang=stt.lang)
if res is None:
    ...   # câu chỉ có số/dấu câu — bỏ qua, đừng dịch bừa
```

Thứ tự ưu tiên: `force_src_lang` → `stt_lang` (STT trả về) → đoán theo hệ chữ
viết. Bước cuối gần như không thể sai vì tiếng Việt dùng Latin còn tiếng Hoa dùng
Hán tự (`onevoice/langid.py`).

### Kiểm tra nhanh từ dòng lệnh

```bash
python -m onevoice.translate "这个地方怎么走?"
echo "Nhà vệ sinh ở đâu?" | python -m onevoice.translate --json
```

### `TranslationResult` trả về gì

| trường | ý nghĩa |
|---|---|
| `target` | bản dịch |
| `src_lang` / `tgt_lang` | `"zh"` / `"vi"` |
| `latency_ms` | thời gian dịch |
| `unk_chars` | **các ký tự model không đọc được** — xem mục 5 |
| `zh_norm_hits` | số lần bảng chuẩn hóa `<unk>` can thiệp |
| `glossary_hits` | số lần bảng địa danh can thiệp |
| `truncated` | input dài quá đã bị cắt |

---

## ⚙️ 4. Tham số & cách chỉnh

Tất cả nằm ở `onevoice/config.py`, override được bằng biến môi trường:

| biến | mặc định | ghi chú |
|---|---|---|
| `ONEVOICE_MT_BEAM` | `4` | 1 = nhanh hơn 25%, **kém rõ với câu ngắn** |
| `ONEVOICE_MT_THREADS` | `4` | đặt bằng số core vật lý |
| `ONEVOICE_MT_COMPUTE` | `int8` | `int8_float32` nếu thấy lỗi số học |
| `ONEVOICE_MT_MODEL` | `models/mt/nllb-zhvi-int8` | thư mục model CT2 |
| `ONEVOICE_MT_TOKENIZER` | `models/mt/tokenizer.json` | |

### Vì sao mặc định beam 4 chứ không phải beam 1

Bộ benchmark gốc toàn câu dài (tin tức, kỹ thuật) nên beam 1 chỉ kém 0.2 chrF++.
Nhưng thiết bị này gặp **câu hội thoại ngắn**, và ở đó beam 1 sai rõ rệt:

| câu | beam 1 | beam 4 |
|---|---|---|
| `Nhà vệ sinh ở đâu ạ?` | `所在哪里?` ❌ mất chữ | `卫生间在哪里?` ✅ |
| `太贵了。` | "- Đủ đắt tiền." ❌ | "Rất đắt tiền." ✅ |
| `请问洗手间在哪里？` | "Tôi hỏi, nhà vệ sinh ở đâu?" | "Làm ơn hỏi, nhà vệ sinh ở đâu?" ✅ |

Với câu ngắn, beam 4 chỉ tốn thêm ~11% thời gian. Đánh đổi rẻ. Chỉ hạ về 1 khi đã
đo trên Jetson thật và thấy không kịp.

---

## 🚨 5. Hạn chế đã biết — ĐỌC KỸ PHẦN NÀY

Đây là phần quan trọng nhất tài liệu. Các lỗi dưới đây **không có cảnh báo tự
nhiên**: model vẫn sinh ra câu tiếng Việt/tiếng Hoa trôi chảy, người dùng không
thể biết là nó đã dịch sai.

### 5.1. Vocab NLLB thiếu nhiều ký tự Hán thường dùng

NLLB không có `byte_fallback`. Ký tự nào không có piece trong vocab sẽ thành
`<unk>` và model **mất hoàn toàn** thông tin đó.

Đo trên 41 MB Wikipedia tiếng Hoa (`scripts/audit_vocab.py`):

| nhóm ký tự thường dùng nhất | số ký tự thiếu |
|---|---|
| Top 500 | **0** ✅ |
| Top 1 000 | 4 (宋 秦 魏 昌) |
| Top 2 000 | 54 |
| Top 3 000 | **331 (11%)** |

Phần thiếu **không rải đều** — nó tập trung đúng ba nhóm mà thiết bị du lịch dùng
nhiều nhất:

- **Ăn uống**: 厕 咸 啤 汤 烫 虾 蟹 辣 酱 醋 鸡 炒 煎 鸭 锅 筷 勺 葱 姜 蒜 椒
- **Họ người**: 刘 吴 杨 赵 郑 邓 冯 蒋 姚 崔 薛 彭 蔡 袁 …
- **Tên tỉnh**: 浙 湘 陕 闽 粤 沪 皖 赣 鄂 桂 黔 滇 …

Hậu quả thật, chưa xử lý:

| câu nguồn | model dịch ra | ký tự bị `<unk>` |
|---|---|---|
| `你好，请问厕所在哪里？` | "Xin chào, hãy hỏi **bà mẹ** ở đâu?" | 厕 |
| `汤太咸了。` | "**Cô ấy đã chết rồi.**" | 汤 咸 |
| `我对虾过敏。` | "Tôi bị dị ứng với **bơ**." | 虾 |
| `这个鸡肉太辣了。` | "Con gà này quá **ngốc**." | 鸡 辣 |
| `我们去吃火锅吧。` | "Chúng ta hãy đi ăn **cá sấu**." | 锅 |

**Đã kiểm chứng đây KHÔNG phải lỗi do cắt vocab** — chạy song song model NLLB gốc
600 MB cho ra kết quả giống hệt từng ký tự. Đây là hạn chế của chính NLLB-200.

#### Cách xử lý hiện tại (chiều zh→vi)

`onevoice/data/zh_unk_map.json` thay cụm chứa ký tự `<unk>` bằng cụm tương đương
**trước khi** tokenize. Mỗi luật đã chạy thử thật và chọn phương án tốt nhất:

| câu nguồn | không có bảng | có bảng |
|---|---|---|
| `你好，请问厕所在哪里？` | "hỏi bà mẹ ở đâu" | "xin hỏi **nhà vệ sinh** ở đâu" ✅ |
| `汤太咸了。` | "Cô ấy đã chết rồi." | "**Canh quá mặn.**" ✅ |
| `我对虾过敏。` | "dị ứng với bơ" | "dị ứng với **tôm**" ✅ |
| `我们去吃火锅吧，请给我一双筷子。` | "đi ăn cá sấu... một cặp cá sấu" | "đi ăn **lẩu**, cho tôi một **đôi đũa**" ✅ |

13/14 câu thử cải thiện rõ rệt.

> ⚠️ Từ tiếng Việt/Anh chèn vào **phải có khoảng trắng hai bên**. Đã đo:
> `菜太salty了` ra "quá ngọt" (sai), `菜太 salty 了` ra "quá mặn" (đúng) — thiếu
> khoảng trắng thì tokenizer ghép nó vào ký tự Hán liền kề.

Mở rộng bảng:

```bash
python scripts/audit_vocab.py --chars           # ký tự nào đang thiếu
python scripts/audit_vocab.py --scan phrases.txt # quét bộ câu của mình
python scripts/audit_vocab.py --check-rules      # dịch thử trước/sau khi sửa bảng
```

#### ❌ Chiều vi→zh KHÔNG chữa được bằng cách này

Model không có token để **sinh ra** các ký tự đó, nên nó bỏ luôn từ:

| câu tiếng Việt | model dịch ra | vấn đề |
|---|---|---|
| `Tôi bị dị ứng với tôm.` | `我对有过敏.` | **mất "tôm"** — nguy hiểm với người dị ứng |
| `Món này cay quá.` | `这个很苦.` | "đắng", sai nghĩa |
| `Món này mặn quá.` | `这太了.` | mất "mặn" |
| `Tôi muốn ăn thịt gà.` | `我想要吃肉.` | chỉ còn "thịt" |
| `Cho tôi một chai bia.` | `给我一个酒.` | thành "rượu" chung chung |

Đây là **trần cứng của vocab NLLB**, không có hậu xử lý nào vá được. Xem mục 6.

### 5.2. Địa danh Việt Nam bị dịch sai (chiều vi→zh)

NLLB là model 200 thứ tiếng, **không có kiến thức Hán-Việt**. Đo trên 30 địa danh
× 5 ngữ cảnh (`scripts/probe_glossary.py`): **26/30 sai ít nhất một lần**.

Hai loại lỗi, loại thứ hai nguy hiểm hơn nhiều:

1. Phiên âm theo âm Latin: `Hà Nội → 哈诺伊` (đúng: 河内), `Sa Pa → 萨帕`
2. **Dịch nghĩa bừa thành một từ tiếng Hoa thông dụng khác** — nghe hoàn toàn tự nhiên:

| địa danh | model dịch ra | nghĩa thật của chuỗi đó |
|---|---|---|
| Sài Gòn | 七个子 | "bảy đứa con" |
| Nha Trang | 时尚 | "thời trang" |
| Cần Thơ | 需要诗歌 | "cần thơ ca" |
| Ninh Bình | 和平 / 宁波 | "hòa bình" / Ninh Ba (thành phố TQ khác) |
| Bình Dương | 太平洋 | "Thái Bình Dương" |
| Đà Nẵng | 达南 / 丹阳 | phiên âm bừa |

#### Cách xử lý: thay ở ĐẦU VÀO, hai bước

Hậu xử lý chuỗi đầu ra **không dùng được** với loại lỗi thứ hai: luật
`和平 → 宁平` sẽ phá mọi câu nói về "hòa bình". Nên `onevoice/glossary.py` thay
tên tiếng Việt bằng Hán tự **ngay trong câu nguồn**, model chỉ việc chép sang
output:

```
"Tôi ở Ninh Bình ba ngày"  →  "Tôi ở 宁平 ba ngày"  →  我已经在宁平三天了 ✅
```

Với 5 địa danh mà chính Hán tự đúng lại chứa ký tự ngoài vocab (岘港, 大叻, 芹苴,
邦美蜀, 芒街 — model không chép được token nó không có), dùng thêm chuỗi "vật
mang" (`carrier`) rồi đổi lại ở đầu ra:

```
"đi Đà Nẵng" → "đi 达港" → 我想去达港 → hậu xử lý → 我想去岘港 ✅
```

Kết quả:

| câu | không glossary | có glossary |
|---|---|---|
| `Tôi muốn đi Đà Nẵng.` | 我想去**丹阳**. | 我想去**岘港**. ✅ |
| `Cần Thơ cách Sài Gòn bao xa?` | 距离**贡**有多远? | **芹苴**距离**西贡**有多远? ✅ |
| `Từ Bình Dương đi Vũng Tàu…` | 从**太平洋**到**海湾**… | 从**平阳**到**头顿**… ✅ |
| `Chúng ta cần hòa bình.` | 我们需要和平. | 我们需要和平. ✅ *(không bị đụng)* |

Khớp tên không phân biệt hoa/thường ("đà lạt" → 大叻) và ưu tiên tên dài trước
("Vịnh Hạ Long" thắng "Hạ Long").

Mở rộng: thêm mục vào `onevoice/data/glossary_vi_zh.json`, rồi
`python scripts/probe_glossary.py` để đo lại.

### 5.3. Model cắt câu / bỏ mệnh đề

NLLB đôi khi dừng giữa chừng ở câu ghép dài:

```
Tôi ở Ninh Bình ba ngày rồi đi Vịnh Hạ Long.  →  我已经在宁平三天了.   (mất vế sau)
```

Segmenter cắt câu ở mốc 8 s nên phần lớn input đã ngắn, nhưng vẫn nên biết. Nếu
gặp nhiều, hạ `max_segment_duration` xuống ~5 s để câu vào model ngắn hơn.

### 5.4. Thành ngữ và tiếng lóng

Fail ở **cả hai chiều**, đo từ trước. Không có cách vá ở tầng này.

---

## 🛣️ 6. Hướng khắc phục thật cho mục 5.1

Bảng chuẩn hóa chỉ vá được một chiều. Muốn xử lý gốc thì phải đổi vocab, tức đổi
model. Đã đo độ phủ vocab trên cùng bộ 112 ký tự du lịch:

| model | vocab | thiếu ký tự du lịch | thiếu 11 ký tự tối quan trọng |
|---|---|---|---|
| NLLB-200 (mọi kích cỡ) | 256 204 | **11/112** | **11/11** ❌ |
| **M2M-100-418M** | 128 104 | **1/112** (啡) | **0/11** ✅ |
| Helsinki opus-mt-zh-vi | 65 001 | 0/112 ✅ | 0/11 ✅ |

Ba điểm cần biết:

1. **Tăng kích cỡ NLLB không giải quyết được gì.** NLLB-1.3B và 3.3B dùng chung
   đúng bộ vocab 256 204 token đó — vẫn thiếu y hệt 11 ký tự.
2. **M2M-100-418M là ứng viên đáng đo tiếp**: nhỏ hơn NLLB-600M, phủ đủ vocab.
   Nhưng chất lượng dịch zh↔vi của nó **chưa đo** — chỉ mới đo độ phủ vocab. Cần
   chạy qua đúng bộ test đã dùng cho NLLB trước khi kết luận.
3. **Opus-MT phủ đủ vocab nhưng chất lượng không dùng được** (37 chrF++, bịa số
   liệu). Nó chỉ hữu ích như kiến trúc đích cho distillation, không phải để ship.

Thứ tự đề xuất:

1. Đo tốc độ NLLB hiện tại trên Jetson thật (`scripts/bench_translate.py`).
2. Ship bản hiện tại cho chiều zh→vi (đã vá tốt) và **cảnh báo rõ hạn chế chiều
   vi→zh** với người dùng cuối.
3. Song song: chạy M2M-100-418M qua bộ test 30 câu + bộ câu du lịch. Nếu chất
   lượng đạt thì đổi model là xong cả hai vấn đề vocab.
4. Nếu cần nhỏ hơn nữa: distill NLLB/M2M sang kiến trúc Marian 77 MB với vocab
   SentencePiece hợp nhất zh+vi tự train — lúc đó tự chọn được ký tự nào vào vocab.

---

## 🧪 7. Công cụ

| script | dùng để |
|---|---|
| `scripts/prepare_model.py` | copy + **verify** model vào `models/mt/` |
| `scripts/bench_translate.py` | đo ms/câu, peak RAM, quét beam/thread trên thiết bị thật |
| `scripts/audit_vocab.py` | tìm ký tự bị `<unk>`, kiểm chứng bảng chuẩn hóa |
| `scripts/probe_glossary.py` | đo model dịch sai địa danh thành chữ gì |
| `pytest tests/` | 53 test; tự bỏ qua phần cần model nếu chưa copy |

---

## 🧰 8. Bảo trì

Khi thêm bộ câu / lĩnh vực mới (vd y tế, mua sắm), làm đúng thứ tự này:

1. Viết bộ câu thật vào một file text.
2. `python scripts/audit_vocab.py --scan file.txt` — xem có ký tự nào bị `<unk>`.
3. Thêm luật vào `onevoice/data/zh_unk_map.json`, **kèm trường `why`**.
4. `python scripts/audit_vocab.py --check-rules` — đọc từng dòng trước/sau.
5. Chạy `pytest tests/` trước khi commit.

Khi chạy thật, để ý cảnh báo trên màn hình:

```
⚠️  Model không đọc được ký tự: 蜀 — thêm luật vào onevoice/data/zh_unk_map.json
```

Đây là cách hệ thống tự báo lỗ hổng vocab ngay tại hiện trường. Đừng bỏ qua.
