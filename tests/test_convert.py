from pathlib import Path

from numpy import allclose

from torch import Tensor, ones, rand
from transformers import BertModel, BertTokenizerFast
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from convert_labse_tf_pt import (
    MODEL_TOKENIZER,
    convert_tf2_hub_model_to_pytorch,
    get_embedding,
    get_labse_tokenizer,
    save_labse_models,
    similarity,
)

from .helpers import Sentences


def test_convert_tokenizer(hub_model):
    tokenizer = get_labse_tokenizer(hub_model)
    assert isinstance(tokenizer, BertTokenizerFast)


def test_convert_model():
    (model, tokenizer) = convert_tf2_hub_model_to_pytorch()
    assert isinstance(model, BertModel)
    assert isinstance(tokenizer, BertTokenizerFast)


def test_tokenized(bert_tokenizer, hf_tokenizer):
    tf_tokenized = bert_tokenizer.convert_tokens_to_ids(
        bert_tokenizer.tokenize(Sentences.similar[0])
    )

    hf_tokenized = hf_tokenizer(Sentences.similar[0], add_special_tokens=False)
    assert tf_tokenized == hf_tokenized.input_ids


def test_save_labse_models(tmp_path: Path, model_tokenizer: MODEL_TOKENIZER):
    save_labse_models(*model_tokenizer, tmp_path, save_tokenizer=True, save_tf=True)

    # PyTorch Model
    assert tmp_path.joinpath("pt").joinpath("config.json").exists()
    assert tmp_path.joinpath("pt").joinpath("pytorch_model.bin").exists()

    # TF model
    assert tmp_path.joinpath("tf").joinpath("config.json").exists()
    assert tmp_path.joinpath("tf").joinpath("tf_model.h5").exists()

    # Flax model
    assert tmp_path.joinpath("flax").joinpath("config.json").exists()
    assert tmp_path.joinpath("flax").joinpath("flax_model.msgpack").exists()

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
