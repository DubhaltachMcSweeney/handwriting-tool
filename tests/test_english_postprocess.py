import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from english_postprocess import correct_english_word, postprocess_text_lines, restore_sentence_case


def test_correct_english_word_fixes_close_match_with_custom_dictionary(monkeypatch):
    dictionary_lookup = {
        4: ["BANK", "VERY", "WAS"],
        5: ["ALICE", "TIRED"],
    }

    assert correct_english_word("BAWK", dictionary_lookup=dictionary_lookup) == "BANK"
    assert correct_english_word("VORY", dictionary_lookup=dictionary_lookup) == "VERY"


def test_postprocess_text_lines_corrects_words_and_keeps_period(monkeypatch):
    dictionary_lookup = {
        2: ["AN", "ON"],
        3: ["THE"],
        4: ["BANK", "VERY"],
        5: ["TIRED"],
    }

    import english_postprocess

    monkeypatch.setattr(english_postprocess, "dictionary_by_length", lambda: dictionary_lookup)

    corrected = postprocess_text_lines(["VORY TINED AN THE BAWK."])

    assert corrected == ["VERY TIRED ON THE BANK."]


def test_postprocess_text_lines_uses_context_for_common_phrase_repairs(monkeypatch):
    dictionary_lookup = {
        1: ["W"],
        2: ["AN", "BY", "OF", "ON", "TO"],
        3: ["GET", "HER"],
        4: ["VARY", "VERY"],
        5: ["TIRED"],
        7: ["SITTING"],
        9: ["BEGINNING"],
    }

    import english_postprocess

    monkeypatch.setattr(english_postprocess, "dictionary_by_length", lambda: dictionary_lookup)

    corrected = postprocess_text_lines(["BEGINNING W GET VORY TIRED OI SITTING NY HER"])

    assert corrected == ["BEGINNING TO GET VERY TIRED OF SITTING BY HER"]


def test_restore_sentence_case_makes_text_read_like_a_sentence():
    restored = restore_sentence_case(
        "ALICE WAS BEGINNING TO GET VERY TIRED OF SITTING BY HER\nSISTER ON THE BANK. I WAS HERE!"
    )

    assert restored == (
        "Alice was beginning to get very tired of sitting by her\n"
        "sister on the bank. I was here!"
    )
