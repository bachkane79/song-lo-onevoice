"""Test tầng dịch.

Các test cần model được đánh dấu `needs_model` và tự bỏ qua nếu chưa chạy
`scripts/prepare_model.py` — nhờ đó test logic (tokenizer, glossary, langid) vẫn
chạy được trên máy CI không có 394 MB model.

    pytest tests/ -v
    pytest tests/ -v -m "not needs_model"     # chỉ test logic thuần
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from onevoice.config import TranslateConfig  # noqa: E402
from onevoice.glossary import Glossary, GlossaryEntry  # noqa: E402
from onevoice.langid import count_scripts, detect_lang  # noqa: E402
from onevoice.translate import normalize_source  # noqa: E402
from onevoice.zh_norm import ZhSourceNormalizer  # noqa: E402

CFG = TranslateConfig()
HAS_MODEL = os.path.isdir(CFG.model_dir) and os.path.isfile(CFG.tokenizer_path)
_skip_no_model = pytest.mark.skipif(
    not HAS_MODEL, reason="chưa có model — chạy scripts/prepare_model.py")


def needs_model(obj):
    """Vừa gắn marker (để lọc `-m "not needs_model"`) vừa tự bỏ qua nếu thiếu model."""
    return pytest.mark.needs_model(_skip_no_model(obj))


@pytest.fixture(scope="module")
def translator():
    from onevoice.translate import NllbTranslator
    tr = NllbTranslator()
    yield tr
    tr.close()


# ── langid ──────────────────────────────────────────────────────────────────
class TestLangId:
    def test_chinese(self):
        assert detect_lang("你好，请问厕所在哪里？") == "zh"

    def test_vietnamese(self):
        assert detect_lang("Nhà vệ sinh ở đâu ạ?") == "vi"

    def test_vietnamese_without_diacritics(self):
        # STT có thể trả tiếng Việt không dấu — vẫn phải ra "vi", không ra "zh"
        assert detect_lang("Nha ve sinh o dau a?") == "vi"

    def test_chinese_with_latin_words(self):
        # Câu tiếng Hoa lẫn từ Latin vẫn phải nhận là tiếng Hoa
        assert detect_lang("这台 GPU 有 8GB 显存") == "zh"

    def test_no_letters_returns_default(self):
        # Chỉ có số/dấu câu → không xác định được, pipeline sẽ bỏ qua
        assert detect_lang("12345 !!!") is None
        assert detect_lang("") is None
        assert detect_lang(None) is None
        assert detect_lang("???", default="vi") == "vi"

    def test_count_scripts(self):
        c = count_scripts("你好 hello")
        assert c["cjk"] == 2 and c["latin"] == 5


# ── Chuẩn hóa nguồn ─────────────────────────────────────────────────────────
class TestNormalizeSource:
    def test_dong_sign_replaced(self):
        # ₫ là <unk> ở cả model gốc lẫn model đã cắt vocab
        assert "₫" not in normalize_source("một triệu ₫")
        assert "VND" in normalize_source("một triệu ₫")

    def test_collapses_whitespace(self):
        assert normalize_source("  a    b  ") == "a b"

    def test_empty(self):
        assert normalize_source(None) == ""
        assert normalize_source("   ") == ""


class TestDegenerateDetection:
    """Nhận ra bản dịch chỉ có dấu câu — xem _CONTENT_RE trong translate.py."""

    def test_punctuation_only_is_degenerate(self):
        from onevoice.translate import _CONTENT_RE
        for junk in (",,,,,,,,,,,,.", "。。。", "!!! ???", "、、、,,,", "— — —"):
            assert not _CONTENT_RE.search(junk), junk

    def test_real_translations_are_not_degenerate(self):
        from onevoice.translate import _CONTENT_RE
        for good in ("你好。", "Xin chào.", "8 倍到 10 倍.", "1990", "岘港",
                     "Đà Nẵng", "こんにちは"):
            assert _CONTENT_RE.search(good), good


class TestZhNormalizer:
    def test_longest_rule_wins(self):
        nz = ZhSourceNormalizer([("辣", " cay "), ("辣椒酱", " tương ớt ")])
        out, n = nz.normalize("辣椒酱")
        assert out == "tương ớt" and n == 1

    def test_collapses_double_space(self):
        nz = ZhSourceNormalizer([("鸡", " gà "), ("汤", " canh ")])
        out, _ = nz.normalize("鸡汤")
        assert "  " not in out

    def test_empty_normalizer_is_identity(self):
        assert ZhSourceNormalizer.empty().normalize("你好") == ("你好", 0)

    def test_missing_file_does_not_crash(self):
        assert len(ZhSourceNormalizer.load("/khong/ton/tai.json")) == 0


# ── Glossary ────────────────────────────────────────────────────────────────
class TestGlossary:
    def test_source_substitution(self):
        g = Glossary([GlossaryEntry("Hà Nội", "河内")])
        out, n = g.apply_source("Tôi đi Hà Nội.")
        assert out == "Tôi đi 河内." and n == 1

    def test_case_insensitive(self):
        g = Glossary([GlossaryEntry("Đà Lạt", "大叻")])
        assert g.apply_source("tôi đi đà lạt")[0] == "tôi đi 大叻"

    def test_longer_name_wins(self):
        g = Glossary([GlossaryEntry("Hạ Long", "下龙"),
                      GlossaryEntry("Vịnh Hạ Long", "下龙湾")])
        assert g.apply_source("đi Vịnh Hạ Long")[0] == "đi 下龙湾"

    def test_does_not_match_inside_word(self):
        # "Huế" không được khớp bên trong "Huệ" hay "Huếch"
        g = Glossary([GlossaryEntry("Huế", "顺化")])
        assert g.apply_source("Huếch")[0] == "Huếch"

    def test_carrier_round_trip(self):
        # 岘 không có trong vocab → chèn carrier rồi đổi lại ở đầu ra
        g = Glossary([GlossaryEntry("Đà Nẵng", "岘港", carrier="达港")])
        src, _ = g.apply_source("đi Đà Nẵng")
        assert src == "đi 达港"
        assert g.apply_target("我想去达港。")[0] == "我想去岘港。"

    def test_no_post_edit_when_carrier_equals_zh(self):
        g = Glossary([GlossaryEntry("Hà Nội", "河内")])
        assert g.apply_target("我去河内。") == ("我去河内。", 0)

    def test_missing_file_does_not_crash(self):
        assert len(Glossary.load("/khong/ton/tai.json")) == 0

    def test_shipped_glossary_is_valid(self):
        g = Glossary.load(CFG.glossary_path)
        assert len(g) >= 20, "glossary địa danh bị rỗng?"
        assert g.lookup_vi("hà nội") == "河内"
        assert g.lookup_vi("Đà Nẵng") == "岘港"

    def test_chinese_places_and_food_in_shipped_glossary(self):
        g = Glossary.load(CFG.glossary_path)
        assert g.lookup_vi("Thượng Hải") == "上海"
        assert g.lookup_vi("Bắc Kinh") == "北京"
        assert g.lookup_vi("vịt quay") == "烤鸭"
        assert g.lookup_vi("bún bò") == "牛肉粉"


class TestViNormalizer:
    def test_restores_question_mark_for_unpunctuated_stt(self):
        from onevoice.vi_norm import restore_vi_punctuation
        out, n = restore_vi_punctuation("Bạn có biết nhà hàng nào nổi tiếng với món đó không")
        assert out.endswith("?") and n >= 1

    def test_restores_period_and_locative_comma(self):
        from onevoice.vi_norm import restore_vi_punctuation
        out, _ = restore_vi_punctuation("Ở việt nam chúng tôi có một món đặc sản đó chính là bún bò")
        assert out == "Ở việt nam, chúng tôi có một món đặc sản, đó chính là bún bò."

    def test_colloquial_chi_cho_toi_normalized(self):
        from onevoice.vi_norm import restore_vi_punctuation
        out, _ = restore_vi_punctuation("Bạn có thể chỉ cho tôi quán ăn ngon được không")
        assert "giới thiệu cho tôi" in out
        assert out.endswith("?")

    def test_preserves_existing_punctuation(self):
        from onevoice.vi_norm import restore_vi_punctuation
        out, n = restore_vi_punctuation("Nhà vệ sinh ở đâu ạ?")
        assert out == "Nhà vệ sinh ở đâu ạ?" and n == 0


# ── Máy dịch (cần model) ────────────────────────────────────────────────────
@needs_model
class TestTranslator:
    def test_zh_to_vi(self, translator):
        res = translator.translate("这个多少钱？", "zh")
        assert res.src_lang == "zh" and res.tgt_lang == "vi"
        assert res.target and detect_lang(res.target) == "vi"

    def test_vi_to_zh(self, translator):
        res = translator.translate("Cái này bao nhiêu tiền?", "vi")
        assert res.src_lang == "vi" and res.tgt_lang == "zh"
        assert res.target and detect_lang(res.target) == "zh"

    def test_accepts_full_nllb_codes(self, translator):
        a = translator.translate("你好。", "zh").target
        b = translator.translate("你好。", "zho_Hans").target
        assert a == b

    def test_rejects_unsupported_lang(self, translator):
        with pytest.raises(ValueError):
            translator.translate("hello", "en")

    def test_empty_input(self, translator):
        assert translator.translate("", "zh").target == ""
        assert translator.translate("   ", "vi").target == ""

    def test_batch_keeps_order_with_empty_slots(self, translator):
        out = translator.translate_batch(["你好。", "", "谢谢。"], "zh")
        assert len(out) == 3
        assert out[1].target == ""
        assert out[0].target and out[2].target
        assert out[0].target != out[2].target

    def test_auto_detect(self, translator):
        assert translator.translate_auto("你好。").tgt_lang == "vi"
        assert translator.translate_auto("Xin chào.").tgt_lang == "zh"
        assert translator.translate_auto("12345") is None

    def test_stt_lang_overrides_script_guess(self, translator):
        # Tiếng Việt không dấu: script detector đoán "vi", STT nói "vi" → khớp
        res = translator.translate_auto("Nha ve sinh o dau", stt_lang="vi")
        assert res.src_lang == "vi"

    def test_force_lang_beats_stt(self, translator):
        res = translator.translate_auto("你好。", force_src_lang="zh", stt_lang="vi")
        assert res.src_lang == "zh"

    def test_zh_unk_map_fixes_toilet(self, translator):
        # Không có bảng chuẩn hóa, 厕 thành <unk> và câu dịch ra "bà mẹ"
        res = translator.translate("你好，请问厕所在哪里？", "zh")
        assert "vệ sinh" in res.target.lower()
        assert res.zh_norm_hits > 0

    def test_unk_chars_reported(self, translator):
        # 蜀 không có trong vocab và không có luật thay → phải báo lên
        res = translator.translate("这是蜀国的历史。", "zh")
        assert res.unk_chars, "phải báo được ký tự bị <unk>"

    def test_glossary_fixes_place_name(self, translator):
        res = translator.translate("Tôi muốn đi Đà Nẵng.", "vi")
        assert "岘港" in res.target
        assert res.glossary_hits > 0

    def test_glossary_does_not_touch_unrelated(self, translator):
        # "hòa bình" phải giữ nguyên 和平, không bị biến thành 宁平 (Ninh Bình)
        res = translator.translate("Chúng ta cần hòa bình.", "vi")
        assert res.glossary_hits == 0
        assert "宁平" not in res.target

    def test_long_input_truncated_not_crashed(self, translator):
        res = translator.translate("你好。" * 500, "zh")
        assert res.truncated and res.target

    def test_default_special_tokens_are_wrong_lang(self, translator):
        """Lý do phải tự dựng chuỗi token thay vì dùng add_special_tokens=True.

        `tokenizer.json` của NLLB có post_processor bị đóng băng ở mã ngôn ngữ
        lúc save (`eng_Latn`). Nếu ai đó "đơn giản hóa" code thành
        `encode(text)` thì MỌI câu sẽ bị gắn nhầm `eng_Latn` và chất lượng dịch
        tụt mà không có lỗi nào báo ra. Test này khóa cái bẫy đó lại.
        """
        enc = translator._tokenizer.encode("这个模型很好。")
        assert enc.tokens[0] == "eng_Latn", (
            "post_processor đổi rồi — kiểm tra lại cách dựng token trong "
            "translate_batch()")
        assert translator._tokenizer.encode(
            "这个模型很好。", add_special_tokens=False).tokens[0] != "eng_Latn"

    def test_tokenizer_matches_transformers(self, translator):
        """Đối chiếu đường tokenize thuần `tokenizers` với `transformers`.

        Chỉ chạy khi có sẵn thư mục tokenizer đầy đủ của HF (có
        `tokenizer_config.json`) — bản deploy chỉ mang theo `tokenizer.json` nên
        test này tự bỏ qua trên thiết bị.
        """
        transformers = pytest.importorskip("transformers")
        hf_dir = os.environ.get("ONEVOICE_HF_TOKENIZER_DIR",
                                os.path.dirname(CFG.tokenizer_path))
        if not os.path.isfile(os.path.join(hf_dir, "tokenizer_config.json")):
            pytest.skip("không có thư mục tokenizer đầy đủ của HF để đối chiếu")

        tok_hf = transformers.AutoTokenizer.from_pretrained(hf_dir, src_lang="zho_Hans")
        text = "这个模型量化之后推理速度提升了三倍。"
        expected = tok_hf.convert_ids_to_tokens(tok_hf.encode(text))
        ours = ["zho_Hans"] + list(
            translator._tokenizer.encode(text, add_special_tokens=False).tokens) + ["</s>"]
        assert ours == expected
