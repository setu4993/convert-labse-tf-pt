from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict


@dataclass
class LaBSE:
    repo: str = "LaBSE"

    # Attributes from `BertConfig`.
    vocab_size: int = 501153
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    num_hidden_layers: int = 12

    # Extra fields.
    vocab_file: Path = Path(__file__).parent.joinpath(
        "data",
        "labse_vocab",
        "cased_vocab.txt",
    )
    size: str = "base"
    tf_hub_link: str = "https://tfhub.dev/google/LaBSE/2"

    def dict(self) -> Dict[str, int]:
        dct = asdict(self)
        # Drop fields that are not supported by `BertConfig`.
        dct.pop("repo")
        dct.pop("vocab_file")
        dct.pop("size")
        dct.pop("tf_hub_link")
        return dct


@dataclass
class SmallerLaBSE(LaBSE):
    repo: str = "smaller-LaBSE"
    vocab_size: int = 173347

    vocab_file: Path = Path(__file__).parent.joinpath(
        "data",
        "smaller_vocab",
        "vocab-en-fr-es-de-zh-ar-zh_classical-it-ja-ko-nl-pl-pt-th-tr-ru.txt",
    )
    size: str = "small"
    tf_hub_link: str = "https://tfhub.dev/jeongukjae/smaller_LaBSE_15lang/1"


@dataclass
class LEALLALarge(LaBSE):
    repo: str = "LEALLA-large"

    hidden_size: int = 256
    intermediate_size: int = 1024
    num_attention_heads: int = 8
    num_hidden_layers: int = 24

    size: str = "large"
    tf_hub_link: str = "https://tfhub.dev/google/LEALLA/LEALLA-large/1"


@dataclass
class LEALLABase(LaBSE):
    repo: str = "LEALLA-base"

    hidden_size: int = 192
    intermediate_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 24

    size: str = "base"
    tf_hub_link: str = "https://tfhub.dev/google/LEALLA/LEALLA-base/1"


@dataclass
class LEALLASmall(LaBSE):
    repo: str = "LEALLA-small"

    hidden_size: int = 128
    intermediate_size: int = 512
    num_attention_heads: int = 8
    num_hidden_layers: int = 24

    size: str = "small"
    tf_hub_link: str = "https://tfhub.dev/google/LEALLA/LEALLA-small/1"
