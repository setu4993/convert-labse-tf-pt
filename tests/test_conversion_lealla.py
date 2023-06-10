from numpy import allclose, min as np_min, abs as np_abs
from pytest import mark
from torch import from_numpy
from transformers import FlaxBertModel, TFBertModel


from convert_labse_tf_pt import (
    MODEL_TOKENIZER,
    get_embedding,
    save_labse_models,
    similarity,
)

from convert_labse_tf_pt.convert import l2_normalize_numpy_array

from tests.helpers import Sentences, tf_model_output_from_encoder

TOLERANCE = 0.02


def test_embeddings_converted_model(
    sentences,
    hub_encoder,
    lealla_atol,
    lealla_model_tokenizer: MODEL_TOKENIZER,
):
    (model, hf_tokenizer) = lealla_model_tokenizer
    pt_output = get_embedding(sentences, model, hf_tokenizer).pooler_output
    tf_output = tf_model_output_from_encoder(sentences, hub_encoder)

    assert allclose(l2_normalize_numpy_array(pt_output.detach().numpy()), tf_output.numpy(), atol=lealla_atol)


def test_embeddings_all_converted_models(
    sentences,
    hub_encoder,
    lealla_model_tokenizer,
    lealla_atol,
    tmp_path,
):
    # Create models.
    (pt_model, hf_tokenizer) = lealla_model_tokenizer
    save_labse_models(pt_model, hf_tokenizer, output_path=tmp_path, huggingface_path=True)

    # TF Hub output.
    hub_output = tf_model_output_from_encoder(sentences, hub_encoder)

    pt_output = get_embedding(sentences, pt_model, hf_tokenizer).pooler_output

    tf_output = TFBertModel.from_pretrained(tmp_path)(
        hf_tokenizer(sentences, return_tensors="tf", padding="max_length")
    ).pooler_output

    flax_output = FlaxBertModel.from_pretrained(tmp_path)(
        **hf_tokenizer(sentences, return_tensors="jax", padding="max_length")
    ).pooler_output

    # Verify all combinations produce equivalent output embeddings.
    numpy_arrays = [
        hub_output.numpy(),
        l2_normalize_numpy_array(pt_output.detach().numpy()),
        l2_normalize_numpy_array(tf_output.numpy()),
        l2_normalize_numpy_array(flax_output),
    ]
    for array1 in numpy_arrays:
        for array2 in numpy_arrays:
            assert allclose(array1, array2, atol=lealla_atol)


@mark.parametrize(
    "sentences1,sentences2,atol",
    [
        (Sentences.english, Sentences.italian, TOLERANCE),
        (Sentences.english, Sentences.japanese, 0.11),
        (Sentences.italian, Sentences.japanese, 0.15),
    ],
)
def test_similarity_converted_model(
    sentences1,
    sentences2,
    atol,
    hub_encoder,
    lealla_model_tokenizer: MODEL_TOKENIZER,
):
    (model, hf_tokenizer) = lealla_model_tokenizer
    pt_output1 = get_embedding(sentences1, model, hf_tokenizer).pooler_output
    pt_output2 = get_embedding(sentences2, model, hf_tokenizer).pooler_output
    tf_output1 = tf_model_output_from_encoder(sentences1, hub_encoder)
    tf_output2 = tf_model_output_from_encoder(sentences2, hub_encoder)

    pt_similarity = similarity(pt_output1, pt_output2)
    tf_similarity = similarity(from_numpy(tf_output1.numpy()), from_numpy(tf_output2.numpy()))
    assert allclose(pt_similarity, tf_similarity, atol=atol)
