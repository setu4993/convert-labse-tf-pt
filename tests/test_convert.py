from pathlib import Path
from typing import Tuple

from numpy import allclose
from pytest import fixture, mark

from bert.tokenization.bert_tokenization import FullTokenizer
from torch import Tensor, ones, rand
from transformers import BertModel, BertTokenizerFast
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from convert_labse_tf_pt import (
    MODEL_TOKENIZER,
    convert_tf2_hub_model_to_pytorch,
    get_embedding,
    get_labse_tokenizer,
    load_tf_model,
    save_labse_models,
    similarity,
)

SIMILAR_SENTENCES = ["Hi, how are you?", "Hello, how are you doing?"]
ENGLISH_SENTENCES = [
    "dog",
    "Puppies are nice.",
    "I enjoy taking long walks along the beach with my dog.",
]
ITALIAN_SENTENCES = [
    "cane",
    "I cuccioli sono carini.",
    "Mi piace fare lunghe passeggiate lungo la spiaggia con il mio cane.",
]
JAPANESE_SENTENCES = ["犬", "子犬はいいです", "私は犬と一緒にビーチを散歩するのが好きです"]
TOLERANCE = 0.01


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


@fixture(
    params=[
        SIMILAR_SENTENCES[0],
        SIMILAR_SENTENCES,
        ENGLISH_SENTENCES,
        ITALIAN_SENTENCES,
        JAPANESE_SENTENCES,
    ]
)
def sentences(request):
    return request.param


def tf_model_output(sentences, hub_model, hf_tokenizer):
    sentences = [sentences] if isinstance(sentences, str) else sentences
    tf_tokenized = hf_tokenizer(sentences, return_tensors="tf", padding="max_length")
    return hub_model(
        [
            tf_tokenized.input_ids,
            tf_tokenized.attention_mask,
            tf_tokenized.token_type_ids,
        ]
    )


def test_convert_tokenizer(hub_model):
    tokenizer = get_labse_tokenizer(hub_model)
    assert isinstance(tokenizer, BertTokenizerFast)


def test_convert_model():
    (model, tokenizer) = convert_tf2_hub_model_to_pytorch()
    assert isinstance(model, BertModel)
    assert isinstance(tokenizer, BertTokenizerFast)


def test_tokenized(bert_tokenizer, hf_tokenizer):
    tf_tokenized = bert_tokenizer.convert_tokens_to_ids(
        bert_tokenizer.tokenize(SIMILAR_SENTENCES[0])
    )

    hf_tokenized = hf_tokenizer(SIMILAR_SENTENCES[0], add_special_tokens=False)
    assert tf_tokenized == hf_tokenized.input_ids


def test_save_labse_models(tmp_path: Path, model_tokenizer: MODEL_TOKENIZER):
    save_labse_models(*model_tokenizer, tmp_path, save_tokenizer=True, save_tf=True)

    # PyTorch Model
    assert tmp_path.joinpath("pt").joinpath("config.json").exists()
    assert tmp_path.joinpath("pt").joinpath("pytorch_model.bin").exists()

    # T5 model
    assert tmp_path.joinpath("tf").joinpath("config.json").exists()
    assert tmp_path.joinpath("tf").joinpath("tf_model.h5").exists()

    # Tokenizer.
    assert tmp_path.joinpath("tokenizer").joinpath("tokenizer_config.json").exists()
    assert tmp_path.joinpath("tokenizer").joinpath("vocab.txt").exists()
    assert tmp_path.joinpath("tokenizer").joinpath("special_tokens_map.json").exists()


def test_get_embedding(sentences, model_tokenizer: MODEL_TOKENIZER):
    output = get_embedding(sentences, *model_tokenizer)
    assert isinstance(output, BaseModelOutputWithPoolingAndCrossAttentions)
    ATTRS = ["last_hidden_state", "pooler_output"]
    for attr in ATTRS:
        assert hasattr(output, attr)
        assert isinstance(getattr(output, attr), Tensor)


def test_similarity():
    X_DIM = 2
    vector = rand(X_DIM, 128)
    scores = similarity(vector, vector)
    assert allclose(scores.diag(), ones(X_DIM))
    assert allclose(scores[0][0], scores[1][1])
    assert allclose(scores[0][1], scores[1][0])


def test_embeddings_converted_model(
    hub_model, hf_tokenizer: BertTokenizerFast, model_tokenizer: MODEL_TOKENIZER
):
    pt_output = get_embedding(SIMILAR_SENTENCES[0], *model_tokenizer).pooler_output

    tf_output = tf_model_output(hub_model, hf_tokenizer)[0]

    assert allclose(
        pt_output.detach().numpy(), tf_output.numpy(), rtol=TOLERANCE, atol=TOLERANCE
    )


def test_embeddings_similarity():
    pass
