from pytest import fixture

import tensorflow_hub as hub
from bert.tokenization.bert_tokenization import FullTokenizer


@fixture
def tf_model():
    return hub.load("https://tfhub.dev/google/LaBSE/1")


def test_convert_model(tf_model):
    pass


def test_convert_tokenizer(tf_model):
    # tf_tokenizer = FullTokenizer()
    pass
