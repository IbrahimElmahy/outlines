"""Helpers for working with CFG grammars in backend implementations."""

from __future__ import annotations

from typing import Iterable, Literal, Set


GrammarFormat = Literal["ebnf", "lark"]


def preprocess_cfg_grammar(
    grammar: str,
    *,
    special_tokens: Iterable[str],
    grammar_format: GrammarFormat,
) -> str:
    """Escape literal angle-bracket tokens before handing grammars to compilers.

    Both the `xgrammar` and `llguidance` compilers treat bare ``<token>`` segments
    as grammar syntax. When users rely on special tokens such as ``<think>``,
    compilation may fail unless those segments are quoted. This helper scans the
    grammar source, finds bare references to special tokens wrapped in angle
    brackets (and not already quoted) and replaces them with quoted literals.
    """
    angle_tokens = _collect_angle_bracket_tokens(special_tokens)
    if not angle_tokens:
        return grammar

    return _quote_angle_bracket_tokens(grammar, angle_tokens, grammar_format)


def _collect_angle_bracket_tokens(tokens: Iterable[str]) -> Set[str]:
    result: Set[str] = set()
    for token in tokens:
        if isinstance(token, str) and token.startswith("<") and token.endswith(">"):
            result.add(token)
    return result


def _quote_angle_bracket_tokens(
    grammar: str,
    tokens: Set[str],
    grammar_format: GrammarFormat,
) -> str:
    result: list[str] = []
    i = 0
    length = len(grammar)
    in_single_quote = False
    in_double_quote = False
    escape_next = False

    while i < length:
        char = grammar[i]

        if char == "\\" and not escape_next:
            escape_next = True
            result.append(char)
            i += 1
            continue

        if char == "'" and not escape_next and not in_double_quote:
            in_single_quote = not in_single_quote
            result.append(char)
            i += 1
            escape_next = False
            continue

        if char == '"' and not escape_next and not in_single_quote:
            in_double_quote = not in_double_quote
            result.append(char)
            i += 1
            escape_next = False
            continue

        if (
            char == "<"
            and not in_single_quote
            and not in_double_quote
        ):
            closing_index = grammar.find(">", i + 1)
            if closing_index != -1:
                candidate = grammar[i : closing_index + 1]
                if candidate in tokens:
                    result.append(_quote_literal(candidate, grammar_format))
                    i = closing_index + 1
                    escape_next = False
                    continue

        result.append(char)
        i += 1
        escape_next = False

    return "".join(result)


def _quote_literal(token: str, grammar_format: GrammarFormat) -> str:
    escaped = token.replace("\\", "\\\\").replace('"', '\\"')
    if grammar_format not in ("ebnf", "lark"):  # pragma: no cover
        raise ValueError(f"Unsupported grammar format: {grammar_format}")
    return f'"{escaped}"'
