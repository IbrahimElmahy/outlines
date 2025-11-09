import os
import sys

import pytest
import torch
import transformers

import outlines
from outlines.backends.llguidance import LLGuidanceBackend
from outlines.backends.xgrammar import XGrammarBackend

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
pytest.importorskip("xgrammar")

MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"
WINDOWS_PLATFORM = sys.platform.startswith("win")
LLGUIDANCE_UNSUPPORTED = WINDOWS_PLATFORM

EBNF_GRAMMAR = """
root ::= wrapped
wrapped ::= <s> answer </s>
answer ::= "yes" | "no"
"""

LARK_GRAMMAR = r"""
?start: wrapped
wrapped: <s> answer </s>
answer: "yes" | "no"

%import common.WS
%ignore WS
"""


@pytest.fixture(scope="module")
def tiny_transformer_with_tags():
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.config.pad_token_id = tokenizer.pad_token_id
    torch.manual_seed(0)
    return outlines.from_transformers(model, tokenizer)


def _generate_response(model, backend_name, processor):
    torch.manual_seed(0)
    generator = outlines.Generator(
        model,
        backend=backend_name,
        processor=processor,
    )
    return generator(
        "Respond with 'yes' or 'no'.",
        max_new_tokens=32,
        do_sample=False,
    )


def _assert_wrapped_response(text: str):
    stripped = text.strip()
    assert stripped.startswith("<s>")
    assert stripped.endswith("</s>")
    inner = stripped[len("<s>") : -len("</s>")].strip()
    assert inner in {"yes", "no"}


@pytest.mark.skipif(
    WINDOWS_PLATFORM,
    reason="xgrammar CFG integration tests are only reliable on POSIX platforms",
)
def test_xgrammar_cfg_special_tokens(tiny_transformer_with_tags):
    backend = XGrammarBackend(tiny_transformer_with_tags)
    processor = backend.get_cfg_logits_processor(EBNF_GRAMMAR)
    assert processor is not None
    response = _generate_response(
        tiny_transformer_with_tags,
        "xgrammar",
        processor,
    )
    _assert_wrapped_response(response)


@pytest.mark.skipif(
    LLGUIDANCE_UNSUPPORTED,
    reason="llguidance torch backend requires a C++ compiler on Windows",
)
def test_llguidance_cfg_special_tokens_ebnf(tiny_transformer_with_tags):
    pytest.importorskip("llguidance")
    backend = LLGuidanceBackend(tiny_transformer_with_tags)
    processor = backend.get_cfg_logits_processor(EBNF_GRAMMAR)
    assert processor is not None
    response = _generate_response(
        tiny_transformer_with_tags,
        "llguidance",
        processor,
    )
    _assert_wrapped_response(response)


@pytest.mark.skipif(
    LLGUIDANCE_UNSUPPORTED,
    reason="llguidance torch backend requires a C++ compiler on Windows",
)
def test_llguidance_cfg_special_tokens_lark(tiny_transformer_with_tags):
    pytest.importorskip("llguidance")
    backend = LLGuidanceBackend(tiny_transformer_with_tags)
    processor = backend.get_cfg_logits_processor(LARK_GRAMMAR)
    assert processor is not None
    response = _generate_response(
        tiny_transformer_with_tags,
        "llguidance",
        processor,
    )
    _assert_wrapped_response(response)
