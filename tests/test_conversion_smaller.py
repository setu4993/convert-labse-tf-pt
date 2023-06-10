from numpy import allclose
from pytest import fixture, mark
from torch import from_numpy
from transformers import FlaxBertModel, TFBertModel

from convert_labse_tf_pt import (
    MODEL_TOKENIZER,
    convert_tf2_hub_model_to_pytorch,
    get_embedding,
    load_tf_model,
    save_labse_models,
    similarity,
)
from convert_labse_tf_pt.configurations import SmallerLaBSE
from tests.helpers import Sentences, tf_model_output_from_hub_model

TOLERANCE = 0.01


@fixture(scope="session")
def smaller_hub_model():
    return load_tf_model(tf_saved_model=SmallerLaBSE().tf_hub_link)


@fixture(scope="session")
def smaller_model_tokenizer() -> MODEL_TOKENIZER:
    return convert_tf2_hub_model_to_pytorch(tf_saved_model=None, smaller=True)


def test_embeddings_converted_model(
    sentences,
    smaller_hub_model,
    smaller_model_tokenizer: MODEL_TOKENIZER,
):
    (model, hf_tokenizer) = smaller_model_tokenizer
    pt_output = get_embedding(sentences, model, hf_tokenizer).pooler_output
    tf_output = tf_model_output_from_hub_model(sentences, smaller_hub_model, hf_tokenizer, smaller=True)

    assert allclose(pt_output.detach().numpy(), tf_output.numpy(), atol=TOLERANCE)


def test_embeddings_all_converted_models(
    sentences,
    smaller_hub_model,
    smaller_model_tokenizer,
    tmp_path,
):
    # Create models.
    (pt_model, hf_tokenizer) = smaller_model_tokenizer
    save_labse_models(pt_model, hf_tokenizer, output_path=tmp_path, huggingface_path=True)

    # TF Hub output.
    hub_output = tf_model_output_from_hub_model(sentences, smaller_hub_model, hf_tokenizer, smaller=True)

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
        pt_output.detach().numpy(),
        tf_output.numpy(),
        flax_output,
    ]
    for array1 in numpy_arrays:
        for array2 in numpy_arrays:
            assert allclose(array1, array2, atol=TOLERANCE)


@mark.parametrize(
    "sentences1,sentences2",
    [
        (Sentences.english, Sentences.italian),
        (Sentences.english, Sentences.japanese),
        (Sentences.italian, Sentences.japanese),
    ],
)
def test_similarity_converted_model(
    sentences1,
    sentences2,
    smaller_hub_model,
    smaller_model_tokenizer: MODEL_TOKENIZER,
):
    (model, hf_tokenizer) = smaller_model_tokenizer
    pt_output1 = get_embedding(sentences1, model, hf_tokenizer).pooler_output
    pt_output2 = get_embedding(sentences2, model, hf_tokenizer).pooler_output
    tf_output1 = tf_model_output_from_hub_model(sentences1, smaller_hub_model, hf_tokenizer, smaller=True)
    tf_output2 = tf_model_output_from_hub_model(sentences2, smaller_hub_model, hf_tokenizer, smaller=True)

    pt_similarity = similarity(pt_output1, pt_output2)
    tf_similarity = similarity(from_numpy(tf_output1.numpy()), from_numpy(tf_output2.numpy()))
    assert allclose(pt_similarity, tf_similarity, atol=TOLERANCE)
