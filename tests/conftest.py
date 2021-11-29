from pytest import fixture

from bert.tokenization.bert_tokenization import FullTokenizer
from transformers import BertTokenizerFast

from convert_labse_tf_pt import (
    MODEL_TOKENIZER,
    convert_tf2_hub_model_to_pytorch,
    get_labse_tokenizer,
    load_tf_model,
)

from .helpers import Sentences

TF_SAVED_MODEL = None


@fixture(scope="session")
def hub_model():
    return load_tf_model(tf_saved_model=TF_SAVED_MODEL)


@fixture(scope="session")
def bert_tokenizer(hub_model) -> FullTokenizer:
    return FullTokenizer(
        hub_model.vocab_file.asset_path.numpy(), hub_model.do_lower_case.numpy()
    )


@fixture(scope="session")
def hf_tokenizer(hub_model) -> BertTokenizerFast:
    return get_labse_tokenizer(hub_model)


@fixture(scope="session")
def model_tokenizer() -> MODEL_TOKENIZER:
    return convert_tf2_hub_model_to_pytorch(tf_saved_model=TF_SAVED_MODEL)


@fixture(
    params=[
        Sentences.similar[0],
        Sentences.similar,
        Sentences.english,
        Sentences.italian,
        Sentences.japanese,
    ]
)
def sentences(request):
    return request.param
