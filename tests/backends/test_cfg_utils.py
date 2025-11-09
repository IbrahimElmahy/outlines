from outlines.backends.cfg_utils import preprocess_cfg_grammar


def test_preprocess_cfg_grammar_quotes_special_tokens_ebnf():
    grammar = "wrapped ::= <s> answer </s>"
    processed = preprocess_cfg_grammar(
        grammar,
        special_tokens={"<s>", "</s>"},
        grammar_format="ebnf",
    )
    assert processed == 'wrapped ::= "<s>" answer "</s>"'


def test_preprocess_cfg_grammar_leaves_quoted_segments():
    grammar = 'wrapped ::= "<s>" answer "</s>"'
    processed = preprocess_cfg_grammar(
        grammar,
        special_tokens={"<s>", "</s>"},
        grammar_format="ebnf",
    )
    assert processed == grammar


def test_preprocess_cfg_grammar_handles_lark():
    grammar = "<s> answer </s>"
    processed = preprocess_cfg_grammar(
        grammar,
        special_tokens={"<s>", "</s>"},
        grammar_format="lark",
    )
    assert processed == '"<s>" answer "</s>"'


def test_preprocess_cfg_grammar_ignores_non_special_tokens():
    grammar = "<custom> answer </custom>"
    processed = preprocess_cfg_grammar(
        grammar,
        special_tokens={"<s>", "</s>"},
        grammar_format="ebnf",
    )
    assert processed == grammar
