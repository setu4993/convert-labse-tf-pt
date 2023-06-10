from numpy import allclose
from pytest import mark
from torch import from_numpy
from transformers import FlaxBertModel, TFBertModel

from convert_labse_tf_pt import (
    MODEL_TOKENIZER,
    get_embedding,
    save_labse_models,
    similarity,
)
from tests.helpers import Sentences, tf_model_output_from_hub_model

TOLERANCE = 0.01


def test_embeddings_converted_model(
    sentences,
    hub_model,
    model_tokenizer: MODEL_TOKENIZER,
):
    (model, hf_tokenizer) = model_tokenizer
    pt_output = get_embedding(sentences, model, hf_tokenizer).pooler_output
    tf_output = tf_model_output_from_hub_model(sentences, hub_model, hf_tokenizer)

    assert allclose(pt_output.detach().numpy(), tf_output.numpy(), atol=TOLERANCE)


def test_embeddings_all_converted_models(
    sentences,
    hub_model,
    model_tokenizer,
    v1_labse_model,
    tmp_path,
):
    # Create models.
    (pt_model, hf_tokenizer) = model_tokenizer
    save_labse_models(pt_model, hf_tokenizer, output_path=tmp_path, huggingface_path=True)

    # TF Hub output.
    hub_output = tf_model_output_from_hub_model(sentences, hub_model, hf_tokenizer)

    # v1 model output.
    v1_output = get_embedding(sentences, v1_labse_model, hf_tokenizer).pooler_output

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
        v1_output.detach().numpy(),
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
    hub_model,
    model_tokenizer: MODEL_TOKENIZER,
):
    (model, hf_tokenizer) = model_tokenizer
    pt_output1 = get_embedding(sentences1, model, hf_tokenizer).pooler_output
    pt_output2 = get_embedding(sentences2, model, hf_tokenizer).pooler_output
    tf_output1 = tf_model_output_from_hub_model(sentences1, hub_model, hf_tokenizer)
    tf_output2 = tf_model_output_from_hub_model(sentences2, hub_model, hf_tokenizer)

    pt_similarity = similarity(pt_output1, pt_output2)
    tf_similarity = similarity(from_numpy(tf_output1.numpy()), from_numpy(tf_output2.numpy()))
    assert allclose(pt_similarity, tf_similarity, atol=TOLERANCE)
