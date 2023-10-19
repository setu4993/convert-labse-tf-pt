from bert.tokenization.bert_tokenization import FullTokenizer
from pytest import fixture
from tensorflow_hub import KerasLayer
from transformers import BertModel, BertTokenizerFast

from convert_labse_tf_pt import (
    MODEL_TOKENIZER,
    convert_tf2_hub_model_to_pytorch,
    get_labse_tokenizer,
    load_tf_model,
)
from convert_labse_tf_pt.configurations import (
    LaBSE,
    LEALLABase,
    LEALLALarge,
    LEALLASmall,
)
from tests.helpers import Sentences

TF_SAVED_MODEL = None


@fixture(scope="session")
def hub_model():
    return load_tf_model()


@fixture(scope="session")
def bert_tokenizer(hub_model) -> FullTokenizer:
    return FullTokenizer(hub_model.vocab_file.asset_path.numpy(), hub_model.do_lower_case.numpy())


@fixture(scope="session")
def hf_tokenizer() -> BertTokenizerFast:
    return get_labse_tokenizer(LaBSE())


@fixture(scope="session")
def model_tokenizer() -> MODEL_TOKENIZER:
    return convert_tf2_hub_model_to_pytorch()


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


@fixture(scope="session")
def v1_labse_model() -> BertModel:
    return BertModel.from_pretrained("setu4993/LaBSE", revision="v1")


@fixture(
    scope="module",
    params=[LEALLASmall(), LEALLABase(), LEALLALarge()],
)
def lealla_config(request):
    return request.param


@fixture
def lealla_atol(sentences):
    if sentences == Sentences.japanese:
        return 0.13
    else:
        return 0.02


@fixture(scope="module")
def hub_encoder(lealla_config):
    return KerasLayer(lealla_config.tf_hub_link)


@fixture(scope="module")
def lealla_model_tokenizer(lealla_config) -> MODEL_TOKENIZER:
    return convert_tf2_hub_model_to_pytorch(lealla_config)
