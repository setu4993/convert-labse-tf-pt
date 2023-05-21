"""
This script can be used to convert a head-less TF2.x LaBSE model to PyTorch, as
published on TensorFlow Hub: https://tfhub.dev/google/LaBSE/2.

The script re-maps the TF2.x Bert weight names to the original names, so the model can
be imported with Huggingface/transformer.

This script is adapted from HuggingFace's BERT conversion script: https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/convert_bert_original_tf2_checkpoint_to_pytorch.py
"""
from argparse import ArgumentParser
from pathlib import Path
from re import match
from typing import List, Tuple, Union

import torch.nn.functional as F
import tensorflow_text  # noqa: F401
from loguru import logger
from tensorflow_hub import load
from torch import from_numpy, matmul, no_grad
from transformers import (
    BertConfig,
    BertModel,
    BertTokenizerFast,
    PretrainedConfig,
    FlaxBertModel,
    TFBertModel,
)
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from convert_labse_tf_pt.configurations import LEALLABase, LEALLALarge, LEALLASmall, LaBSE, SmallerLaBSE

PATH = Union[str, Path]
MODEL_TOKENIZER = Tuple[BertModel, BertTokenizerFast]


def load_tf_model(tf_saved_model: PATH = LaBSE().tf_hub_link):
    tf_saved_model = tf_saved_model
    logger.info(f"Loading TF LaBSE model from {tf_saved_model}...")
    return load(tf_saved_model)


def get_pretrained_config(conversion_config: LaBSE = LaBSE(), labse_config: PATH = None) -> PretrainedConfig:
    if labse_config:
        labse_config = Path(labse_config) if isinstance(labse_config, str) else labse_config
        logger.info(f"Loading model based on config from {labse_config}...")
        config = BertConfig.from_json_file(labse_config)
    else:
        config = BertConfig.from_pretrained("bert-base-uncased")
        config.update(conversion_config.dict())
    return config


def get_labse_tokenizer(conversion_config: LaBSE = LaBSE()) -> BertTokenizerFast:
    logger.info(f"Using vocab file {conversion_config.vocab_file} for HF LaBSE tokenizer...")
    return BertTokenizerFast(
        conversion_config.vocab_file,
        # Preserve case.
        do_lower_case=False,
        # Use the length from the positional_embeddings size.
        model_max_length=512,
    )


def save_labse_models(
    model: BertModel,
    tokenizer: BertTokenizerFast,
    output_path: PATH,
    save_safe_tensor: bool = True,
    save_tokenizer: bool = True,
    save_tf: bool = True,
    save_flax: bool = True,
    huggingface_path: bool = False,
):
    output_path = Path(output_path) if isinstance(output_path, str) else output_path

    pt_output_path = output_path.joinpath("pt") if not huggingface_path else output_path
    pt_output_path.mkdir(exist_ok=True, parents=True)
    model.save_pretrained(pt_output_path)
    logger.info(f"Saved PyTorch model to {pt_output_path}.")

    if save_tokenizer:
        tokenizer_output_path = output_path.joinpath("tokenizer") if not huggingface_path else output_path
        tokenizer_output_path.mkdir(exist_ok=True, parents=True)
        tokenizer.save_pretrained(tokenizer_output_path)
        logger.info(f"Saved tokenizer to {tokenizer_output_path}.")

    if save_safe_tensor:
        logger.info(f"Loading HuggingFace compatible PyTorch LaBSE model from {pt_output_path}.")
        pt_output_path = output_path.joinpath("pt_safe") if not huggingface_path else output_path
        pt_model = BertModel.from_pretrained(pt_output_path)
        for parameter in pt_model.parameters():
            parameter = parameter.contiguous()
        pt_model.save_pretrained(pt_output_path, safe_serialization=True)
        del pt_model
        logger.info(f"Saved PyTorch `safetensors` model to {pt_output_path}.")

    if save_tf:
        tf_output_path = output_path.joinpath("tf") if not huggingface_path else output_path
        tf_output_path.mkdir(exist_ok=True, parents=True)
        logger.info(f"Loading HuggingFace compatible TF LaBSE model from {pt_output_path}.")
        tf_model = TFBertModel.from_pretrained(pt_output_path, from_pt=True)
        tf_model.save_pretrained(tf_output_path)
        del tf_model
        logger.info(f"Saved TF model to {tf_output_path}.")

    if save_flax:
        flax_output_path = output_path.joinpath("flax") if not huggingface_path else output_path
        flax_output_path.mkdir(exist_ok=True, parents=True)
        logger.info(f"Loading HuggingFace compatible Flax LaBSE model from {pt_output_path}.")
        flax_model = FlaxBertModel.from_pretrained(pt_output_path, from_pt=True)
        flax_model.save_pretrained(flax_output_path)
        del flax_model
        logger.info(f"Saved Flax model to {flax_output_path}.")


def load_labse_weights(tf_model, model: BertModel) -> BertModel:  # noqa: C901
    logger.info("Converting LaBSE weights...")
    for var in tf_model.variables:
        full_name, array = var.name, var.numpy()
        name = full_name.replace(":0", "").split("/")

        # Corresponds to `do_lower_case` attribute of the model.
        if full_name.startswith("Variable"):
            continue
        pointer = model
        trace = []
        for i, m_name in enumerate(name):
            # Encoder layers
            if m_name == "transformer":
                trace.extend(["encoder"])
                pointer = getattr(pointer, "encoder")
            elif match(r"layer_\d+", m_name):
                layer_num = int(m_name.split("_")[-1])
                trace.extend(["layer", str(layer_num)])
                pointer = getattr(pointer, "layer")
                pointer = pointer[layer_num]
            # Embeddings.
            elif i == 0 and "embedding" in m_name:
                trace.append("embeddings")
                pointer = getattr(pointer, "embeddings")
                if m_name == "word_embeddings":
                    trace.append("word_embeddings")
                    pointer = getattr(pointer, "word_embeddings")
                elif m_name == "position_embedding":
                    trace.append("position_embeddings")
                    pointer = getattr(pointer, "position_embeddings")
                elif m_name == "type_embeddings":
                    trace.append("token_type_embeddings")
                    pointer = getattr(pointer, "token_type_embeddings")
                # LayerNorm for embeddings.
                elif m_name == "embeddings":
                    continue
                else:
                    raise ValueError(f"Unknown embedding layer with name {full_name}")
                trace.append("weight")
                pointer = getattr(pointer, "weight")
            elif m_name == "layer_norm":
                trace.append("LayerNorm")
                pointer = getattr(pointer, "LayerNorm")
            # Self-attention layer.
            elif m_name == "self_attention":
                trace.extend(["attention"])
                pointer = getattr(pointer, "attention")
            # Attention key.
            elif m_name == "key":
                trace.append("self")
                trace.append("key")
                pointer = getattr(pointer, "self")
                pointer = getattr(pointer, "key")
            # Attention query.
            elif m_name == "query":
                trace.append("self")
                trace.append("query")
                pointer = getattr(pointer, "self")
                pointer = getattr(pointer, "query")
            # Attention value.
            elif m_name == "value":
                trace.append("self")
                trace.append("value")
                pointer = getattr(pointer, "self")
                pointer = getattr(pointer, "value")
            # Attention output.
            elif m_name == "attention_output":
                trace.extend(["output", "dense"])
                pointer = getattr(pointer, "output")
                pointer = getattr(pointer, "dense")
            # Attention output LayerNorm.
            elif m_name == "self_attention_layer_norm":
                trace.extend(["attention", "output", "LayerNorm"])
                pointer = getattr(pointer, "attention")
                pointer = getattr(pointer, "output")
                pointer = getattr(pointer, "LayerNorm")
            # Attention intermediate.
            elif m_name == "intermediate":
                trace.extend(["intermediate", "dense"])
                pointer = getattr(pointer, "intermediate")
                pointer = getattr(pointer, "dense")
            # Output.
            elif m_name == "output":
                trace.extend(["output", "dense"])
                pointer = getattr(pointer, "output")
                pointer = getattr(pointer, "dense")
            # Output LayerNorm.
            elif m_name == "output_layer_norm":
                trace.extend(["output", "LayerNorm"])
                pointer = getattr(pointer, "output")
                pointer = getattr(pointer, "LayerNorm")
            # Pooler.
            elif m_name == "pooler_transform":
                trace.extend(["pooler", "dense"])
                pointer = getattr(pointer, "pooler")
                pointer = getattr(pointer, "dense")
            # Weights, biases.
            elif m_name in ["bias", "beta"]:
                trace.append("bias")
                pointer = getattr(pointer, "bias")
            elif m_name in ["kernel", "gamma"]:
                trace.append("weight")
                pointer = getattr(pointer, "weight")
            else:
                logger.warning(f"Ignored {m_name}")

        # For certain layers reshape is necessary.
        trace = ".".join(trace)
        if match(r"(\S+)\.attention\.self\.(key|value|query)\.(bias|weight)", trace) or match(
            r"(\S+)\.attention\.output\.dense\.weight", trace
        ):
            array = array.reshape(pointer.data.shape)
        if "kernel" in full_name:
            array = array.transpose()
        if pointer.shape == array.shape:
            pointer.data = from_numpy(array)
        else:
            raise ValueError(
                f"Shape mismatch in layer {full_name}: Model expects shape "
                f"{pointer.shape} but layer contains shape: {array.shape}"
            )
        logger.info(f"Successfully set variable {full_name} to PyTorch layer {trace}")
    return model


def load_lealla_weights(tf_model, model: BertModel) -> BertModel:
    logger.info("Converting LEALLA weights...")
    for var in tf_model.variables:
        full_name, array = var.name, var.numpy()
        name = full_name.replace(":0", "").split("/")

        # Corresponds to `do_lower_case` attribute of the model.
        if full_name.startswith("Variable"):
            continue
        pointer = model
        trace = []
        for i, m_name in enumerate(name):
            # All `name`s start with `student_bert` which is just a container.
            if m_name == "student_bert":
                continue
            elif match(r"layer_\d+", m_name):
                layer_num = int(m_name.split("_")[-1])
                trace.extend(["layer", str(layer_num)])
                pointer = getattr(pointer, "layer")
                pointer = pointer[layer_num]
            # Embeddings.
            elif "embedding" in m_name:
                if m_name == "embeddings":
                    trace.append("embeddings")
                    pointer = getattr(pointer, "embeddings")
                    continue
                elif m_name == "word_embeddings":
                    trace.append("word_embeddings")
                    pointer = getattr(pointer, "word_embeddings")
                elif m_name == "position_embeddings":
                    trace.append("position_embeddings")
                    pointer = getattr(pointer, "position_embeddings")
                # `in` because we need to match both `token_type_embeddings` and `token_type_embeddings_real`.
                elif "token_type_embeddings" in m_name:
                    trace.append("token_type_embeddings")
                    pointer = getattr(pointer, "token_type_embeddings")
                else:
                    raise ValueError(f"Unknown embedding layer with name {full_name}")
                trace.append("weight")
                pointer = getattr(pointer, "weight")
            # Layers with equivalent names.
            elif m_name in [
                "encoder",
                "LayerNorm",
                "attention",
                "self",
                "key",
                "query",
                "value",
                "intermediate",
                "output",
                "dense",
                "pooler",
            ]:
                trace.extend([m_name])
                pointer = getattr(pointer, m_name)
            # Weights, biases.
            elif m_name in ["bias", "beta"]:
                trace.append("bias")
                pointer = getattr(pointer, "bias")
            elif m_name in ["kernel", "gamma"]:
                trace.append("weight")
                pointer = getattr(pointer, "weight")
            else:
                logger.warning(f"Ignored {m_name}")

        # For certain layers reshape is necessary.
        trace = ".".join(trace)
        if match(r"(\S+)\.attention\.self\.(key|value|query)\.(bias|weight)", trace) or match(
            r"(\S+)\.attention\.output\.dense\.weight", trace
        ):
            array = array.reshape(pointer.data.shape)
        if "kernel" in full_name:
            array = array.transpose()
        if pointer.shape == array.shape:
            pointer.data = from_numpy(array)
        else:
            raise ValueError(
                f"Shape mismatch in layer {full_name}: Model expects shape "
                f"{pointer.shape} but layer contains shape: {array.shape}"
            )
        logger.info(f"Successfully set variable {full_name} to PyTorch layer {trace}")
    return model


def load_weights(tf_model, pretrained_config, conversion_config):
    logger.info("Creating empty model from pretrained config...")
    model = BertModel(pretrained_config)
    if "labse" in conversion_config.repo.lower():
        return load_labse_weights(tf_model, model)
    elif "lealla" in conversion_config.repo.lower():
        return load_lealla_weights(tf_model, model)
    else:
        raise ValueError(f"Unknown repo {conversion_config.repo}.")


def convert_tf2_hub_model_to_pytorch(
    conversion_config: LaBSE = LaBSE(),
    tf_saved_model: PATH = None,
    labse_config: PATH = None,
    output_path: PATH = None,
    huggingface_path: bool = False,
) -> MODEL_TOKENIZER:
    logger.info("Creating base configuration.")
    pretrained_config = get_pretrained_config(conversion_config, labse_config)

    logger.info("Loading pre-trained LaBSE TensorFlow SavedModel from TF Hub or disk.")
    tf_model = load_tf_model(tf_saved_model or conversion_config.tf_hub_link)

    logger.info("Loading weights from TF SavedModel.")
    model = load_weights(tf_model, pretrained_config, conversion_config)

    logger.info("Initializing LaBSE tokenizer.")
    tokenizer = get_labse_tokenizer(conversion_config)

    if output_path:
        logger.info(f"Saving model and tokenizer to {output_path}.")
        save_labse_models(model, tokenizer, output_path, huggingface_path=huggingface_path)
        logger.info(f"Saved model and tokenizer to {output_path}.")
    else:
        logger.warning("output_path not set, skipping saving model and tokenizer to disk.")

    return (model, tokenizer)


def get_embedding(
    sentences: Union[str, List[str]],
    model: BertModel = None,
    tokenizer: BertTokenizerFast = None,
) -> BaseModelOutputWithPoolingAndCrossAttentions:
    if not model and not tokenizer:
        (model, tokenizer) = convert_tf2_hub_model_to_pytorch()
    elif not model:
        (model, _) = convert_tf2_hub_model_to_pytorch()
    elif not tokenizer:
        tokenizer = get_labse_tokenizer()

    if isinstance(sentences, str):
        sentences = [sentences]

    model = model.eval()

    tokenized = tokenizer(sentences, return_tensors="pt", padding="max_length", truncation=True)
    with no_grad():
        output = model(**tokenized)
    return output


def similarity(embeddings_1, embeddings_2):
    normalized_embeddings_1 = F.normalize(embeddings_1, p=2)
    normalized_embeddings_2 = F.normalize(embeddings_2, p=2)
    return matmul(normalized_embeddings_1, normalized_embeddings_2.transpose(0, 1))


def get_base_arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--tf_saved_model",
        type=str,
        help="Path or URL to the TensorFlow 2.x SavedModel.",
        default=None,
    )
    parser.add_argument(
        "--config",
        type=str,
        help="JSON config file corresponding to the LaBSE model config. This file specifies the model architecture.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path where the model and tokenizer should be output.",
        default=None,
    )
    parser.add_argument(
        "--huggingface_path",
        help="Should models be exported in HuggingFace default folder structure?",
        default=False,
        action="store_true",
    )
    return parser


def convert_labse():
    parser = get_base_arg_parser()
    parser.add_argument(
        "--smaller",
        help="Convert smaller-LaBSE model?",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()
    conversion_config = LaBSE() if not args.smaller else SmallerLaBSE()

    convert_tf2_hub_model_to_pytorch(
        conversion_config,
        args.tf_saved_model,
        args.config,
        args.output_path,
        args.huggingface_path,
    )


def convert_lealla():
    parser = get_base_arg_parser()
    parser.add_argument(
        "--size",
        help="Size of the model to convert. One of 'small', 'base', 'large'.",
        default="small",
        type=str,
    )
    args = parser.parse_args()
    if args.size == "small":
        conversion_config = LEALLASmall()
    elif args.size == "base":
        conversion_config = LEALLABase()
    elif args.size == "large":
        conversion_config = LEALLALarge()
    else:
        raise ValueError(f"Unknown model size {args.size}.")
    convert_tf2_hub_model_to_pytorch(
        conversion_config,
        args.tf_saved_model,
        args.config,
        args.output_path,
        args.huggingface_path,
    )
