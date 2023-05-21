from bert.tokenization.bert_tokenization import FullTokenizer
from huggingface_hub import cached_download, hf_hub_url
from pytest import fixture
from transformers import BertModel, BertTokenizerFast

from convert_labse_tf_pt import (
    MODEL_TOKENIZER,
    convert_tf2_hub_model_to_pytorch,
    get_labse_tokenizer,
    load_tf_model,
)
from convert_labse_tf_pt.configurations import LaBSE
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
    config_location = cached_download(hf_hub_url(repo_id="setu4993/LaBSE", revision="v1", filename="config.json"))
    model_location = cached_download(hf_hub_url(repo_id="setu4993/LaBSE", revision="v1", filename="pytorch_model.bin"))
    return BertModel.from_pretrained(model_location, config=config_location)
