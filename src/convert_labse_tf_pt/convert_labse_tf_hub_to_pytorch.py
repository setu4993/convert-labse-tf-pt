"""
This script can be used to convert a head-less TF2.x LaBSE model to PyTorch, as
published on TensorFlow Hub: https://tfhub.dev/google/LaBSE/1.

The script re-maps the TF2.x Bert weight names to the original names, so the model can
be imported with Huggingface/transformer.
"""
import argparse
import re
from logging import INFO, getLogger

import torch
from tensorflow_hub import load
from transformers import BertConfig, BertModel, BertTokenizerFast

logger = getLogger(__name__)
logger.setLevel(INFO)


def load_tf_model():
    return load("https://tfhub.dev/google/LaBSE/1")


def get_labse_tokenizer(tf_model) -> BertTokenizerFast:
    return BertTokenizerFast(
        tf_model.vocab_file.asset_path.numpy(),
        do_lower_case=tf_model.do_lower_case.numpy().item(),
    )


def load_tf2_weights_in_bert(model, tf_model):
    # Convert layers.
    logger.info("Converting weights...")
    for var in tf_model.variables:
        full_name, array = var.name, var.numpy()
        name = full_name.replace(":0", "").split("/")

        # corresponds to `do_lower_case` attribute of the model.
        if full_name.startswith("Variable"):
            continue
        pointer = model
        trace = []
        for i, m_name in enumerate(name):
            if m_name != "layer_norm" and m_name.startswith("layer"):
                layer_num = int(m_name.split("_")[-1])
                # encoder layers
                trace.extend(["layer", str(layer_num)])
                pointer = getattr(pointer, "layer")
                pointer = pointer[layer_num]
            #             elif layer_num == config.num_hidden_layers + 4:
            #                 # pooler layer
            #                 trace.extend(["pooler", "dense"])
            #                 pointer = getattr(pointer, "pooler")
            #                 pointer = getattr(pointer, "dense")
            elif m_name == "transformer":
                trace.extend(["encoder"])
                pointer = getattr(pointer, "encoder")
            elif m_name == "embeddings" and name[1] == "layer_norm":
                trace.extend(["embeddings", "LayerNorm"])
                pointer = getattr(pointer, "embeddings")
                pointer = getattr(pointer, "LayerNorm")
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
                else:
                    raise ValueError("Unknown embedding layer with name {full_name}")
                trace.append("weight")
                pointer = getattr(pointer, "weight")
            elif m_name.startswith("self_attention"):
                # self-attention layer
                trace.extend(["attention"])
                pointer = getattr(pointer, "attention")
            elif m_name == "attention_output":
                # output attention dense
                trace.extend(["output", "dense"])
                pointer = getattr(pointer, "output")
                pointer = getattr(pointer, "dense")
            elif m_name == "output":
                # output dense
                trace.extend(["output", "dense"])
                pointer = getattr(pointer, "output")
                pointer = getattr(pointer, "dense")
            #         elif m_name == "output_layer_norm":
            #             # output dense
            #             trace.extend(["output", "LayerNorm"])
            #             pointer = getattr(pointer, "output")
            #             pointer = getattr(pointer, "LayerNorm")
            elif m_name == "key":
                # attention key
                trace.append("self")
                trace.append("key")
                pointer = getattr(pointer, "self")
                pointer = getattr(pointer, "key")
            elif m_name == "query":
                # attention query
                trace.append("self")
                trace.append("query")
                pointer = getattr(pointer, "self")
                pointer = getattr(pointer, "query")
            elif m_name == "value":
                # attention value
                trace.append("self")
                trace.append("value")
                pointer = getattr(pointer, "self")
                pointer = getattr(pointer, "value")
            elif m_name == "intermediate":
                # attention intermediate dense
                trace.extend(["intermediate", "dense"])
                pointer = getattr(pointer, "intermediate")
                pointer = getattr(pointer, "dense")
            elif m_name == "pooler_transform":
                trace.extend(["pooler", "dense"])
                pointer = getattr(pointer, "pooler")
                pointer = getattr(pointer, "dense")
            #         elif m_name == "_output_layer_norm":
            #             # output layer norm
            #             trace.append("output")
            #             pointer = getattr(pointer, "output")
            # weights & biases
            elif m_name in ["bias", "beta"]:
                trace.append("bias")
                pointer = getattr(pointer, "bias")
            elif m_name in ["kernel", "gamma"]:
                trace.append("weight")
                pointer = getattr(pointer, "weight")
            else:
                logger.warning(f"Ignored {m_name}")
            if "_layer_norm" in m_name:
                # output attention norm
                trace.extend(["output", "LayerNorm"])
                #             pointer = getattr(pointer, "attention")
                pointer = getattr(pointer, "output")
                pointer = getattr(pointer, "LayerNorm")

        # for certain layers reshape is necessary
        logger.info(f"{full_name} -> {trace}")
        trace = ".".join(trace)
        traces.append(trace)
        if re.match(
            r"(\S+)\.attention\.self\.(key|value|query)\.(bias|weight)", trace
        ) or re.match(r"(\S+)\.attention\.output\.dense\.weight", trace):
            array = array.reshape(pointer.data.shape)
        if "kernel" in full_name:
            array = array.transpose()
        if pointer.shape == array.shape:
            pointer.data = torch.from_numpy(array)
        else:
            raise ValueError(
                f"Shape mismatch in layer {full_name}: Model expects shape {pointer.shape} but layer contains shape: {array.shape}"
            )
        logger.info(f"Successfully set variable {full_name} to PyTorch layer {trace}")
    #         if trace == "encoder.layer.0.attention.self.query.weight":
    #             break
    return model


def convert_tf2_checkpoint_to_pytorch(
    # tf_checkpoint_path,
    # config_path,
    # pytorch_dump_path
):
    # Instantiate model
    config_path = "config/labse_config.json"
    logger.info(f"Loading model based on config from {config_path}...")
    config = BertConfig.from_json_file(config_path)
    model = BertModel(config)

    tf_model = load_tf_model()

    # Load weights from checkpoint
    # logger.info(f"Loading weights from checkpoint {tf_checkpoint_path}...")
    load_tf2_weights_in_bert(model, tf_model)

    # Save pytorch-model
    # logger.info(f"Saving PyTorch model to {pytorch_dump_path}...")
    # torch.save(model.state_dict(), pytorch_dump_path)
    return model


def convert_tf2_hub_model_to_pytorch():
    return convert_tf2_checkpoint_to_pytorch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tf_checkpoint_path",
        type=str,
        required=True,
        help="Path to the TensorFlow 2.x checkpoint path.",
    )
    parser.add_argument(
        "--bert_config_file",
        type=str,
        required=True,
        help="The config json file corresponding to the BERT model. This specifies the model architecture.",
    )
    parser.add_argument(
        "--pytorch_dump_path",
        type=str,
        required=True,
        help="Path to the output PyTorch model (must include filename).",
    )
    args = parser.parse_args()
    convert_tf2_checkpoint_to_pytorch(
        args.tf_checkpoint_path, args.bert_config_file, args.pytorch_dump_path
    )
