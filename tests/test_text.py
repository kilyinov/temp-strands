from mp_profile.text import fold_quote_text, is_verbatim, name_tokens, normalize_name, split_sentences


def test_quote_matching_folds_typography_and_whitespace() -> None:
    source = "We said \u201chomes\u201d \u2014 not  slogans.\nAnd we meant it."
    assert is_verbatim('We said "homes" - not slogans. And we meant it.', source)
    assert not is_verbatim("We said houses", source)
    assert fold_quote_text("A\u2019s") == fold_quote_text("A's")


def test_name_tokens_drop_honorifics_and_accents() -> None:
    assert name_tokens("PLIBERSEK, the Hon. Tanya") == ["plibersek", "tanya"]
    assert name_tokens("Dr Zoë O\u2019Brien MP") == ["zoe", "obrien"]
    assert normalize_name("Mr  Danny  O'BRIEN") == normalize_name("danny obrien")


def test_split_sentences() -> None:
    assert split_sentences("First one here. Second one? Third!") == ["First one here.", "Second one?", "Third!"]
