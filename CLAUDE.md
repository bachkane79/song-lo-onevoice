# CLAUDE.md

Hướng dẫn cho Claude Code khi làm việc trong repo này.

## Dự án là gì

Thiết bị phiên dịch song ngữ Việt ⇄ Hoa, chạy **hoàn toàn offline** trên Jetson.
Không gọi API, không cần mạng lúc chạy.

```
🎙️ Mic → [1] VAD → [2] Segmenter → [3] STT → [4] Translate → [5] Display
```

## ⚡ Việc đầu tiên trên máy mới clone về

**Chạy lệnh này trước khi làm bất cứ việc gì khác:**

```bash
pip install -r requirements.txt
python scripts/setup.py
```

`setup.py` tự tải toàn bộ model (~655 MB), kiểm tra sha256, rồi dịch thử một câu
mỗi chiều để chứng minh pipeline chạy được. Chạy lại nhiều lần vô hại.

Sau đó chạy thiết bị:

```bash
python run_pipeline.py                              # với micro
python run_pipeline.py --wav a.wav --show-route     # không cần micro
python run_pipeline.py --offline                    # không cần cả model STT
```

Nếu `setup.py` báo lỗi 404 khi tải model dịch → xem mục "Phát hành model" cuối file.

## Model không nằm trong git

GitHub giới hạn 100 MB/file nên **không có model nào trong repo**. `/models/` đã
bị `.gitignore` — **đừng bao giờ commit nội dung thư mục này**.

| model | dung lượng | lấy từ đâu | script |
|---|---|---|---|
| SenseVoice int8 (STT zh) | 229 MB | Hugging Face | `scripts/download_stt.py` |
| Zipformer-30M (STT vi) | 29 MB | Hugging Face | `scripts/download_stt.py` |
| NLLB-600M vocab-trim (dịch) | 397 MB | GitHub Release của chính repo này | `scripts/prepare_model.py` |

Cả hai script đều **verify sau khi tải** (sha256 với model dịch, nạp thử
recognizer với STT). Lý do: file tải thiếu vẫn nạp được và vẫn sinh ra chữ — chỉ
là chữ rác. Không kiểm tra thì lỗi chỉ lộ ra ở người dùng cuối.

## Bố cục

```
VAD-Segmenter/     Module 1 & 2 — KHÔNG SỬA, do thành viên khác bàn giao
onevoice/          Module 3, 4, 5 + điều phối
  stt.py           interface STT + engine giả lập + registry
  stt_sherpa.py    SenseVoice + Zipformer + BỘ ĐỊNH TUYẾN NGÔN NGỮ
  translate.py     máy dịch NLLB (ctranslate2 + tokenizers, KHÔNG dùng torch)
  glossary.py      địa danh Hán-Việt, thay ở ĐẦU VÀO (vi→zh)
  zh_norm.py       vá ký tự ngoài vocab NLLB (zh→vi)
  langid.py        nhận diện zh/vi theo hệ chữ viết (dự phòng)
  pipeline.py      worker thread, chính sách quá tải
  data/            bảng dữ liệu + manifest release
  mic.py           tự chọn micro — mở thử từng thiết bị trước khi nạp model
  vadlog.py        lọc log ồn của VAD-Segmenter (không sửa file họ) + gộp lượt
scripts/           setup, tải model, benchmark, audit, diag_vad
tests/             78 test — chạy `pytest -q` trước mỗi lần commit
docs/              STT_GUIDE, TRANSLATE_GUIDE, AUDIO_TUNING, JETSON_SETUP
```

## Quy tắc khi sửa code

1. **Không sửa `VAD-Segmenter/`.** Đó là bàn giao của thành viên khác. Cần gì thì
   bọc thêm ở `onevoice/pipeline.py`.
2. **Không thêm torch/transformers vào đường chạy.** Máy dịch cố tình chỉ dùng
   `ctranslate2` + `tokenizers` vì Jetson cài torch rất nặng. `transformers` chỉ
   được xuất hiện trong test (có `importorskip`).
3. **Các hằng số dưới đây là số đo, không phải số đoán.** Sửa xong phải đo lại,
   đừng chỉnh theo cảm giác:
   - `BilingualSttEngine.ALPHA/MARGIN/FLOOR` — hiệu chỉnh trên 249 mẫu audio.
     Đo lại: `python scripts/bench_stt.py --wav-dir DIR --routing`
   - `onevoice/data/zh_unk_map.json` — mỗi luật đã thử nhiều phương án và chọn
     bản tốt nhất. Đo lại: `python scripts/audit_vocab.py --check-rules`
   - `onevoice/data/glossary_vi_zh.json` — đo lại: `python scripts/probe_glossary.py`
   - Tham số VAD/Segmenter trong `PipelineConfig` — hiệu chỉnh theo số đo trên
     mic thật, xem `docs/AUDIO_TUNING.md`. Đo lại: `python scripts/diag_vad.py`
4. **Tiếng Việt trong comment/docstring/tài liệu**, theo đúng phong cách repo.
5. **Chạy `pytest -q` trước khi commit.** 78 test, phần cần model tự bỏ qua nếu
   chưa tải.

## Hai cái bẫy đã gặp, đừng lặp lại

**`tokenizer.json` của NLLB có post_processor đóng băng ở `eng_Latn`.** Nếu ai đó
"đơn giản hoá" `translate.py` thành `tokenizer.encode(text)` thì MỌI câu sẽ bị
gắn nhầm mã ngôn ngữ và chất lượng dịch tụt mà không có lỗi nào báo ra. Phải tự
dựng chuỗi `[src_lang] + pieces + </s>`. Có test khoá lại
(`test_default_special_tokens_are_wrong_lang`).

**LID có sẵn của SenseVoice không dùng được để định tuyến.** Nó không biết tiếng
Việt nên báo `<|zh|>` cho 8/8 câu tiếng Việt. Bộ định tuyến hiện tại dùng sản
lượng của Zipformer làm tín hiệu — xem docstring `BilingualSttEngine`.

**Energy Valley (tầng 2 của Segmenter) tắt mặc định — đừng bật lại mà không đo.**
AGC trong `VAD_process` kéo mọi thứ về biên độ 0.3 nên năng lượng lúc im (0.101)
gần bằng lúc nói (0.124). Tầng đó cắt khi `energy < ratio * peak`, nên `ratio`
0.25 rơi ngay dưới median lúc nói → **cắt giữa câu liên tục**; hạ xuống 0.12 thì
dưới cả sàn ồn → mọi câu rơi xuống Max Duration 8s. Không có giá trị nào ở giữa
chạy được. Chi tiết + cách đo: `docs/AUDIO_TUNING.md`.

## Hạn chế đã biết (đã ghi trong docs, không phải bug cần sửa)

- **Chiều vi→zh**: vocab NLLB thiếu token nên không sinh được 鸡/辣/虾… →
  `"Tôi bị dị ứng với tôm"` ra `我对有过敏` (mất "tôm"). Không vá được bằng hậu
  xử lý. Hướng xử lý thật: đo M2M-100-418M — xem `docs/TRANSLATE_GUIDE.md` mục 6.
- **Câu < 2 giây** định tuyến ngôn ngữ kém chắc hơn (97.8% tổng, 100% với câu đầy
  đủ). Dùng `--force-lang zh|vi` khi biết trước ai đang nói.
- **Zipformer tiếng Việt là CC BY-NC-ND 4.0** — phi thương mại. Chủ dự án đã xác
  nhận dùng phi thương mại nên không cần đổi model.

## Lệnh hay dùng

```bash
python scripts/setup.py                              # cài mọi thứ
python scripts/setup.py --check                      # chỉ kiểm tra
pytest -q                                            # 78 test
pytest -q -m "not needs_model"                       # test logic, không cần model

python scripts/diag_vad.py                           # đo VAD trước khi chỉnh Segmenter
python run_pipeline.py --list-devices                # ✓/✗ mở thử từng micro
python scripts/bench_stt.py --wav-dir DIR            # đo STT + chấm định tuyến
python scripts/bench_translate.py --sweep-beam       # đo máy dịch
python scripts/audit_vocab.py --chars                # ký tự nào model không đọc được
python -m onevoice.translate "这个地方怎么走?"        # dịch nhanh một câu
```

## Phát hành model dịch (chỉ chủ repo cần làm, một lần mỗi khi model đổi)

Model dịch nằm ở GitHub Release `mt-model-v1` của chính repo này. Manifest (tên
file + kích thước + sha256) ở `onevoice/data/mt_release.json`.

Nếu release chưa tồn tại hoặc cần phát hành bản mới:

```bash
# Cần 4 file này (lấy từ máy đã build model)
#   ct2/nllb-trim-int8/model.bin, config.json, shared_vocabulary.json
#   nllb-zhvi-trimmed/tokenizer.json

gh release create mt-model-v1 \
  model.bin config.json shared_vocabulary.json tokenizer.json \
  --title "Model dịch NLLB-600M vocab-trim (CT2 int8)" \
  --notes "394 MB, zh<->vi hai chiều. Tải tự động bằng scripts/prepare_model.py"
```

Không có `gh` thì vào **Releases → Draft a new release** trên web, đặt tag đúng
`mt-model-v1`, kéo thả 4 file vào.

Đổi model → cập nhật `size` và `sha256` trong `onevoice/data/mt_release.json`,
nếu không bước verify sẽ báo sai và từ chối dùng file mới.
