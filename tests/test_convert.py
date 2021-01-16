from typing import Tuple

from numpy import allclose
from pytest import fixture

from bert.tokenization.bert_tokenization import FullTokenizer
from torch import no_grad
from transformers import BertModel, BertTokenizerFast

from convert_labse_tf_pt import (
    convert_tf2_hub_model_to_pytorch,
    get_labse_tokenizer,
    load_tf_model,
)

SIMILAR_SENTENCES = ["Hi, how are you?", "Hello, how are you doing?"]
TOKENIZER_ATTRIBUTES = {"padding": "max_length", "max_length": 64}
TOLERANCE = 0.1


@fixture(scope="session")
def hub_model():
    return load_tf_model()


@fixture(scope="session")
def bert_tokenizer(hub_model) -> FullTokenizer:
    return FullTokenizer(
        hub_model.vocab_file.asset_path.numpy(), hub_model.do_lower_case.numpy()
    )


@fixture(scope="session")
def hf_tokenizer(hub_model) -> BertTokenizerFast:
    return get_labse_tokenizer(hub_model)


@fixture(scope="session")
def model_tokenizer() -> Tuple[BertModel, BertTokenizerFast]:
    return convert_tf2_hub_model_to_pytorch()


def test_convert_tokenizer(hub_model):
    tokenizer = get_labse_tokenizer(hub_model)
    assert isinstance(tokenizer, BertTokenizerFast)


def test_tokenized(bert_tokenizer, hf_tokenizer):
    tf_tokenized = bert_tokenizer.convert_tokens_to_ids(
        bert_tokenizer.tokenize(SIMILAR_SENTENCES[0])
    )

    hf_tokenized = hf_tokenizer(SIMILAR_SENTENCES[0], add_special_tokens=False)
    assert tf_tokenized == hf_tokenized.input_ids


def test_convert_model():
    hub_model = load_tf_model()
    (model, labse_tokenizer) = convert_tf2_hub_model_to_pytorch()
    model = model.eval()

    pt_tokenized = labse_tokenizer(
        SIMILAR_SENTENCES[0], return_tensors="pt", **TOKENIZER_ATTRIBUTES
    )
    with no_grad():
        pt_labse_output = model(**pt_tokenized)
    pt_output = pt_labse_output.pooler_output

    tf_tokenized = labse_tokenizer(
        SIMILAR_SENTENCES[0], return_tensors="tf", **TOKENIZER_ATTRIBUTES
    )
    tf_labse_output = hub_model(
        [
            tf_tokenized.input_ids,
            tf_tokenized.attention_mask,
            tf_tokenized.token_type_ids,
        ]
    )
    tf_output = tf_labse_output[0]
    assert allclose(
        pt_output.detach().numpy(), tf_output.numpy(), rtol=TOLERANCE, atol=TOLERANCE
    )
