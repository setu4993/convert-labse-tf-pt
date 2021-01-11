from numpy import allclose
from pytest import fixture

from bert.tokenization.bert_tokenization import FullTokenizer
from torch import no_grad

from convert_labse_tf_pt import (
    convert_tf2_hub_model_to_pytorch,
    get_labse_tokenizer,
    load_tf_model,
)

SIMILAR_SENTENCES = ["Hi, how are you?", "Hello, how are you doing?"]
TOKENIZER_ATTRIBUTES = {"padding": "max_length", "max_length": 64}
TOLERANCE = 0.1


def test_convert_tokenizer():
    hub_model = load_tf_model()

    tf_tokenizer = FullTokenizer(
        hub_model.vocab_file.asset_path.numpy(), hub_model.do_lower_case.numpy()
    )
    tf_tokenized = tf_tokenizer.convert_tokens_to_ids(
        tf_tokenizer.tokenize(SIMILAR_SENTENCES[0])
    )

    hf_tokenizer = get_labse_tokenizer(hub_model)
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
