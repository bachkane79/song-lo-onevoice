"""Máy dịch Việt ⇄ Hoa chạy hoàn toàn offline.

Model: **NLLB-200-distilled-600M đã cắt vocab riêng cho zh+vi**, export
CTranslate2 int8 — 394 MB trên disk, ~1.07 GB peak RSS, một model làm cả hai
chiều zh→vi và vi→zh.

Vì sao model này (số đo trên bộ test 30 câu, 8 domain):

    | model                    | chrF++ zh→vi | disk  | ms/câu (i9, beam1) |
    |--------------------------|--------------|-------|--------------------|
    | NLLB-600M vocab-trim     | 52.5         | 394 M | 364                |
    | NLLB-600M gốc            | 52.7         | 600 M | 561                |
    | Helsinki opus-mt-zh-vi   | 37.2         |  77 M | 66                 |

Opus-MT nhẹ hơn 5× nhưng **bịa số liệu và bỏ nội dung ở 27% câu** (dịch 菜太咸了
"mặn quá" thành "ngon quá", 胡志明市 thành tên người Hàn) — không dùng được cho
thiết bị phiên dịch. Vocab-trim lấy được -34% disk / -35% latency so với NLLB
gốc mà chất lượng không đổi (26/30 câu zh→vi giống hệt từng ký tự).

Runtime chỉ cần hai thư viện: `ctranslate2` + `tokenizers`.
**Không cần torch, không cần transformers** — quan trọng trên Jetson, nơi cài
torch aarch64 vừa nặng vài GB vừa phải build từ nguồn.
"""

import json
import os
import re
import threading
import time
import unicodedata
from dataclasses import dataclass

from onevoice.config import (
    LANG_VI,
    LANG_ZH,
    NLLB_TO_SHORT,
    SHORT_TO_NLLB,
    TranslateConfig,
)
from onevoice.glossary import Glossary
from onevoice.vi_norm import ViSourceNormalizer
from onevoice.zh_norm import ZhSourceNormalizer

__all__ = ["NllbTranslator", "TranslationResult", "normalize_source"]

_EOS = "</s>"
_UNK = "<unk>"

# NLLB không có byte_fallback: ký tự nào thiếu piece 1 ký tự sẽ thành <unk>.
# Đây là các ký tự đã đo được là <unk> ở CẢ model gốc và model đã cắt vocab
# (tức hạn chế vốn có của NLLB, không do việc cắt vocab), nên phải normalize ở
# tầng tiền xử lý. `₫` là quan trọng nhất: thiết bị du lịch nói về giá liên tục.
UNK_SAFE_REPLACEMENTS = {
    "₫": " VND",
    "￥": " CNY",
    "㎡": " m2",
    "℃": " C",
}

_WS_RE = re.compile(r"[ \t ]+")


def normalize_source(text):
    """Chuẩn hóa text trước khi đưa vào tokenizer.

    - NFKC: gộp ký tự full-width / dấu tổ hợp về dạng chuẩn NLLB đã thấy khi train
    - thay các ký tự bị `<unk>` bằng chữ tương đương (₫ → VND)
    - gom khoảng trắng
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    for bad, good in UNK_SAFE_REPLACEMENTS.items():
        if bad in text:
            text = text.replace(bad, good)
    return _WS_RE.sub(" ", text).strip()


# Ký tự "có nội dung": chữ số, Latin (kèm dấu tiếng Việt), Hán, Kana.
# Bản dịch không chứa ký tự nào trong nhóm này = model chỉ nhả dấu câu.
#
# Vì sao phải kiểm: beam search của NLLB đôi khi rơi vào chuỗi dấu câu thuần khi
# câu vào có cụm lặp và bị cắt lửng. Đo thật trên mic DGM20, câu
# "Dạ đúng rồi đúng rồi nghĩa là cái đấy là là nó sẽ chạy song song với nhau cái
# chuyện gì" ra đúng ",,,,,,,,,,,,." — không exception, không <unk>, nên tầng
# hiển thị in nguyên chuỗi đó ra như một bản dịch bình thường.
_CONTENT_RE = re.compile("[0-9A-Za-zÀ-ɏḀ-ỿ"
                         "぀-ヿ一-鿿]")


@dataclass
class TranslationResult:
    """Kết quả một lần dịch."""

    source: str
    target: str
    src_lang: str          # mã ngắn: "zh" | "vi"
    tgt_lang: str          # mã ngắn: "zh" | "vi"
    latency_ms: float
    in_tokens: int = 0
    out_tokens: int = 0
    truncated: bool = False    # input dài quá max_input_tokens, đã bị cắt
    glossary_hits: int = 0     # số lần glossary Hán-Việt sửa output (vi→zh)
    zh_norm_hits: int = 0      # số lần chuẩn hóa <unk> đầu vào tiếng Hoa (zh→vi)
    # Các ký tự nguồn bị tokenizer biến thành <unk> — phần thông tin model KHÔNG
    # nhìn thấy. Danh sách không rỗng = câu dịch có thể sai/thiếu nghĩa; nên log
    # lại để bổ sung vào onevoice/data/zh_unk_map.json.
    unk_chars: tuple = ()
    # True = `target` chỉ có dấu câu, không một chữ nào (xem _CONTENT_RE).
    # `target` vẫn giữ nguyên chuỗi model nhả ra để còn gỡ lỗi được; tầng gọi
    # phải tự quyết hiển thị gì thay thế.
    degenerate: bool = False

    def __str__(self):
        return self.target


class NllbTranslator:
    """Wrapper CTranslate2 + tokenizers cho NLLB zh ⇄ vi.

    Thread-safe: `translate()` có lock nên gọi được từ nhiều thread, nhưng trong
    pipeline OneVoice chỉ có một worker gọi (dịch là tầng nặng nhất, chạy song
    song trên cùng 4 core chỉ làm chậm cả hai).

    Ví dụ::

        tr = NllbTranslator()
        print(tr.translate("这个模型量化之后推理速度提升了三倍。", "zh", "vi").target)
        print(tr.translate("Cho tôi hai vé đi Hà Nội.", "vi", "zh").target)
    """

    def __init__(self, config=None, lazy=False):
        """
        :param config: `TranslateConfig`; None = lấy mặc định (đọc biến môi trường)
        :param lazy: True = chỉ nạp model khi dịch câu đầu tiên. Mặc định False vì
            trên thiết bị ta muốn nạp sẵn lúc khởi động (nạp mất ~1-3s), không
            muốn câu nói đầu tiên của khách phải chờ.
        """
        self.config = config or TranslateConfig()
        self._lock = threading.Lock()
        self._translator = None
        self._tokenizer = None
        self._lang_token_ids = {}
        self.load_seconds = 0.0

        self.glossary = Glossary.load(self.config.glossary_path) \
            if self.config.use_glossary else Glossary.empty()
        self.zh_norm = ZhSourceNormalizer.load(self.config.zh_unk_map_path) \
            if self.config.use_zh_unk_map else ZhSourceNormalizer.empty()
        self.vi_norm = ViSourceNormalizer(enabled=True)

        if not lazy:
            self.load()

    # ── Nạp model ────────────────────────────────────────────────────────────
    def load(self):
        """Nạp tokenizer + model CTranslate2 vào RAM (idempotent)."""
        if self._translator is not None:
            return
        with self._lock:
            if self._translator is not None:
                return
            self._load_unlocked()

    def _load_unlocked(self):
        import ctranslate2
        from tokenizers import Tokenizer

        cfg = self.config
        if not os.path.isdir(cfg.model_dir):
            raise FileNotFoundError(
                "Không tìm thấy model CTranslate2 tại: %s\n"
                "→ Chạy: python scripts/prepare_model.py --source <thư mục test-mt>\n"
                "  hoặc đặt biến môi trường ONEVOICE_MT_MODEL." % cfg.model_dir)
        if not os.path.isfile(cfg.tokenizer_path):
            raise FileNotFoundError(
                "Không tìm thấy tokenizer.json tại: %s" % cfg.tokenizer_path)

        t0 = time.perf_counter()
        self._tokenizer = Tokenizer.from_file(cfg.tokenizer_path)

        # tokenizer.json của NLLB có post_processor bị "đóng băng" ở src_lang lúc
        # save (thường là eng_Latn). Nếu dùng add_special_tokens=True thì MỌI câu
        # đều bị gắn nhầm eng_Latn. Nên ta encode không special token rồi tự dựng
        # chuỗi [src_lang] ... </s> — cách này khớp từng token với
        # transformers.NllbTokenizer (đã đối chiếu, xem tests/test_translate.py).
        for nllb_code in (LANG_ZH, LANG_VI):
            tid = self._tokenizer.token_to_id(nllb_code)
            if tid is None:
                raise ValueError(
                    "tokenizer.json thiếu mã ngôn ngữ %r — sai file tokenizer?"
                    % nllb_code)
            self._lang_token_ids[nllb_code] = tid

        self._translator = ctranslate2.Translator(
            cfg.model_dir,
            device=cfg.device,
            device_index=0,
            inter_threads=cfg.inter_threads,
            intra_threads=cfg.intra_threads,
            compute_type=cfg.compute_type,
        )
        self.load_seconds = time.perf_counter() - t0

    def warmup(self):
        """Chạy một câu ngắn mỗi chiều để CTranslate2 cấp phát xong buffer.

        Câu đầu tiên luôn chậm hơn 2-3× vì phải allocate. Gọi hàm này lúc khởi
        động thiết bị để khách nói câu đầu không phải chờ.
        """
        self.load()
        self.translate("你好。", "zh", "vi")
        self.translate("Xin chào.", "vi", "zh")

    # ── Dịch ─────────────────────────────────────────────────────────────────
    def translate(self, text, src_lang, tgt_lang=None, beam_size=None):
        """Dịch một câu.

        :param text: câu nguồn (thường là output của STT)
        :param src_lang: "zh" | "vi" (hoặc mã NLLB đầy đủ "zho_Hans"/"vie_Latn")
        :param tgt_lang: None = tự lấy tiếng còn lại
        :param beam_size: override beam size cho riêng lần gọi này
        :returns: `TranslationResult`
        """
        results = self.translate_batch([text], src_lang, tgt_lang, beam_size)
        return results[0]

    def translate_batch(self, texts, src_lang, tgt_lang=None, beam_size=None):
        """Dịch nhiều câu cùng lúc (cùng một chiều).

        Dùng khi dịch file/log sẵn có. Trong pipeline thời gian thực thì mỗi lần
        chỉ có một câu, batch không giúp gì.
        """
        src_nllb = self._resolve_lang(src_lang, "src_lang")
        if tgt_lang is None:
            tgt_nllb = LANG_VI if src_nllb == LANG_ZH else LANG_ZH
        else:
            tgt_nllb = self._resolve_lang(tgt_lang, "tgt_lang")

        src_short = NLLB_TO_SHORT[src_nllb]
        tgt_short = NLLB_TO_SHORT[tgt_nllb]

        self.load()
        cfg = self.config
        beam = beam_size or cfg.beam_size

        # ── Tiền xử lý + tokenize ───────────────────────────────────────────
        # `prepared[i]` = None nghĩa là câu rỗng sau khi chuẩn hóa → không gọi
        # model, nhưng vẫn phải giữ đúng vị trí trong danh sách kết quả trả về.
        prepared = []
        for raw in texts:
            clean = normalize_source(raw)
            if not clean:
                prepared.append(None)
                continue
            norm_hits = glos_hits = 0
            if src_short == "zh":
                # Vocab NLLB thiếu nhiều ký tự Hán thường dùng (厕咸啤汤虾辣鸡...).
                # Thay trước khi tokenize, nếu không model sẽ nhận <unk> và bỏ nghĩa.
                clean, norm_hits = self.zh_norm.normalize(clean)
            else:
                # 1. Khôi phục dấu câu (, . ?) cho câu STT tiếng Việt không dấu câu
                #    và chuẩn hóa cụm khẩu ngữ dễ bị NLLB dịch word-by-word.
                clean, _ = self.vi_norm.normalize(clean)
                # 2. Thay địa danh Việt-Trung & món ăn bằng Hán tự (hoặc carrier) ngay
                #    trong câu nguồn — model chép sang output rồi hậu xử lý về Hán tự chuẩn.
                clean, glos_hits = self.glossary.apply_source(clean)

            enc = self._tokenizer.encode(clean, add_special_tokens=False)
            pieces = list(enc.tokens)
            unk_chars = self._collect_unk_chars(clean, enc)

            was_truncated = False
            limit = max(1, cfg.max_input_tokens - 2)
            if len(pieces) > limit:
                pieces = pieces[:limit]
                was_truncated = True

            prepared.append({
                "tokens": [src_nllb] + pieces + [_EOS],
                "norm_hits": norm_hits,
                "glos_hits": glos_hits,
                "unk_chars": unk_chars,
                "truncated": was_truncated,
            })

        # ── Chạy model ──────────────────────────────────────────────────────
        batch_tokens = [p["tokens"] for p in prepared if p is not None]
        raw_results, per_item_ms = [], 0.0
        if batch_tokens:
            t0 = time.perf_counter()
            with self._lock:
                raw_results = self._translator.translate_batch(
                    batch_tokens,
                    target_prefix=[[tgt_nllb]] * len(batch_tokens),
                    beam_size=beam,
                    max_decoding_length=cfg.max_decoding_length,
                    no_repeat_ngram_size=cfg.no_repeat_ngram_size,
                    replace_unknowns=False,
                )
            per_item_ms = (time.perf_counter() - t0) * 1000.0 / len(batch_tokens)

        # ── Hậu xử lý + ghép về đúng thứ tự ban đầu ─────────────────────────
        results, cursor = [], 0
        for i, item in enumerate(prepared):
            if item is None:
                results.append(TranslationResult(
                    source=texts[i] or "", target="", src_lang=src_short,
                    tgt_lang=tgt_short, latency_ms=0.0))
                continue

            hyp_tokens = list(raw_results[cursor].hypotheses[0])
            cursor += 1
            # Token đầu là target_prefix (mã ngôn ngữ đích) — bỏ đi
            if hyp_tokens and hyp_tokens[0] == tgt_nllb:
                hyp_tokens = hyp_tokens[1:]
            ids = [self._tokenizer.token_to_id(tk) for tk in hyp_tokens]
            target = self._tokenizer.decode([x for x in ids if x is not None],
                                            skip_special_tokens=True).strip()

            glos_hits = item["glos_hits"]
            if tgt_short == "zh":
                # Bước 2: đổi chuỗi "vật mang" về Hán tự đúng (达港 → 岘港).
                # Chỉ chạy với 5 địa danh mà Hán tự đúng chứa ký tự ngoài vocab.
                target, post_hits = self.glossary.apply_target(target)
                glos_hits += post_hits

            results.append(TranslationResult(
                source=texts[i], target=target, src_lang=src_short,
                tgt_lang=tgt_short, latency_ms=round(per_item_ms, 1),
                in_tokens=len(item["tokens"]), out_tokens=len(hyp_tokens),
                truncated=item["truncated"], glossary_hits=glos_hits,
                zh_norm_hits=item["norm_hits"], unk_chars=item["unk_chars"],
                degenerate=bool(target) and not _CONTENT_RE.search(target)))
        return results

    @staticmethod
    def _collect_unk_chars(text, encoding):
        """Trả về tuple các đoạn text nguồn đã bị tokenizer biến thành `<unk>`.

        Chỉ dùng để chẩn đoán/log — nếu tokenizers không trả offset thì im lặng
        bỏ qua, tuyệt đối không để việc này làm chết luồng dịch.
        """
        if _UNK not in encoding.tokens:
            return ()
        found = []
        try:
            for token, (start, end) in zip(encoding.tokens, encoding.offsets):
                if token == _UNK and 0 <= start < end <= len(text):
                    piece = text[start:end].strip()
                    if piece and piece not in found:
                        found.append(piece)
        except (AttributeError, TypeError, ValueError):
            return ()
        return tuple(found)

    def translate_auto(self, text, force_src_lang=None, stt_lang=None):
        """Dịch và tự quyết chiều: force > lang của STT > đoán theo script.

        Trả về None nếu không xác định được ngôn ngữ (câu chỉ có số/dấu câu) —
        pipeline nên bỏ qua thay vì dịch bừa.
        """
        from onevoice.langid import detect_lang

        src = None
        for candidate in (force_src_lang, stt_lang):
            if candidate:
                key = str(candidate).strip().lower()
                if key in SHORT_TO_NLLB:
                    src = NLLB_TO_SHORT[SHORT_TO_NLLB[key]]
                    break
        if src is None:
            src = detect_lang(text)
        if src is None:
            return None
        return self.translate(text, src)

    # ── Nội bộ ───────────────────────────────────────────────────────────────
    @staticmethod
    def _resolve_lang(lang, argname):
        if not lang:
            raise ValueError("%s không được để trống" % argname)
        key = str(lang).strip()
        if key in NLLB_TO_SHORT:
            return key
        nllb = SHORT_TO_NLLB.get(key.lower())
        if nllb is None:
            raise ValueError(
                "%s=%r không hỗ trợ. Chỉ nhận 'zh'/'vi' (hoặc 'zho_Hans'/'vie_Latn')."
                % (argname, lang))
        return nllb

    def stats(self):
        """Thông tin để log lúc khởi động thiết bị."""
        cfg = self.config
        size_mb = 0.0
        if os.path.isdir(cfg.model_dir):
            size_mb = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(cfg.model_dir) for f in fs) / 1024.0 ** 2
        return {
            "model_dir": cfg.model_dir,
            "disk_mb": round(size_mb, 1),
            "device": cfg.device,
            "compute_type": cfg.compute_type,
            "intra_threads": cfg.intra_threads,
            "beam_size": cfg.beam_size,
            "load_seconds": round(self.load_seconds, 2),
            "glossary_entries": len(self.glossary),
            "zh_unk_rules": len(self.zh_norm),
        }

    def close(self):
        """Giải phóng model khỏi RAM."""
        with self._lock:
            self._translator = None
            self._tokenizer = None

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _cli():
    """`python -m onevoice.translate "câu cần dịch"` — kiểm tra nhanh trên thiết bị."""
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Dịch nhanh zh <-> vi (offline)")
    ap.add_argument("text", nargs="*", help="Câu cần dịch (bỏ trống = đọc stdin)")
    ap.add_argument("--src", default=None, help="zh | vi (mặc định: tự nhận diện)")
    ap.add_argument("--beam", type=int, default=None)
    ap.add_argument("--json", action="store_true", help="In ra JSON")
    args = ap.parse_args()

    cfg = TranslateConfig()
    if args.beam:
        cfg.beam_size = args.beam
    tr = NllbTranslator(cfg)
    print("[MT] %s" % json.dumps(tr.stats(), ensure_ascii=False), file=sys.stderr)

    lines = [" ".join(args.text)] if args.text else \
        [ln.strip() for ln in sys.stdin if ln.strip()]

    for line in lines:
        res = tr.translate(line, args.src) if args.src else tr.translate_auto(line)
        if res is None:
            print("(không xác định được ngôn ngữ)", file=sys.stderr)
            continue
        if args.json:
            print(json.dumps(res.__dict__, ensure_ascii=False))
        else:
            print("%s → %s  (%.0f ms)" % (res.src_lang, res.tgt_lang, res.latency_ms),
                  file=sys.stderr)
            print(res.target)


if __name__ == "__main__":
    _cli()
