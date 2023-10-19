---
language:
  - af
  - am
  - ar
  - as
  - az
  - be
  - bg
  - bn
  - bo
  - bs
  - ca
  - ceb
  - co
  - cs
  - cy
  - da
  - de
  - el
  - en
  - eo
  - es
  - et
  - eu
  - fa
  - fi
  - fr
  - fy
  - ga
  - gd
  - gl
  - gu
  - ha
  - haw
  - he
  - hi
  - hmn
  - hr
  - ht
  - hu
  - hy
  - id
  - ig
  - is
  - it
  - ja
  - jv
  - ka
  - kk
  - km
  - kn
  - ko
  - ku
  - ky
  - la
  - lb
  - lo
  - lt
  - lv
  - mg
  - mi
  - mk
  - ml
  - mn
  - mr
  - ms
  - mt
  - my
  - ne
  - nl
  - no
  - ny
  - or
  - pa
  - pl
  - pt
  - ro
  - ru
  - rw
  - si
  - sk
  - sl
  - sm
  - sn
  - so
  - sq
  - sr
  - st
  - su
  - sv
  - sw
  - ta
  - te
  - tg
  - th
  - tk
  - tl
  - tr
  - tt
  - ug
  - uk
  - ur
  - uz
  - vi
  - wo
  - xh
  - yi
  - yo
  - zh
  - zu
tags:
  - bert
  - sentence_embedding
  - multilingual
  - google
  - sentence-similarity
  - lealla
  - labse
license: apache-2.0
datasets:
  - CommonCrawl
  - Wikipedia
---

# LEALLA-large

## Model description

LEALLA is a collection of lightweight language-agnostic sentence embedding models supporting 109 languages, distilled from [LaBSE](https://ai.googleblog.com/2020/08/language-agnostic-bert-sentence.html). The model is useful for getting multilingual sentence embeddings and for bi-text retrieval.

- Model: [HuggingFace's model hub](https://huggingface.co/setu4993/LEALLA-large).
- Paper: [arXiv](https://arxiv.org/abs/2302.08387).
- Original model: [TensorFlow Hub](https://tfhub.dev/google/LEALLA/LEALLA-large/1).
- Conversion from TensorFlow to PyTorch: [GitHub](https://github.com/setu4993/convert-labse-tf-pt).

This is migrated from the v1 model on the TF Hub. The embeddings produced by both the versions of the model are [equivalent](https://github.com/setu4993/convert-labse-tf-pt/blob/c0d4fbce789b0709a9664464f032d2e9f5368a86/tests/test_conversion_lealla.py#L31). Though, for some of the languages (like Japanese), the LEALLA models appear to require higher tolerances when comparing embeddings and similarities.

## Usage

Using the model:

```python
import torch
from transformers import BertModel, BertTokenizerFast


tokenizer = BertTokenizerFast.from_pretrained("setu4993/LEALLA-large")
model = BertModel.from_pretrained("setu4993/LEALLA-large")
model = model.eval()

english_sentences = [
    "dog",
    "Puppies are nice.",
    "I enjoy taking long walks along the beach with my dog.",
]
english_inputs = tokenizer(english_sentences, return_tensors="pt", padding=True)

with torch.no_grad():
    english_outputs = model(**english_inputs)
```

To get the sentence embeddings, use the pooler output:

```python
english_embeddings = english_outputs.pooler_output
```

Output for other languages:

```python
italian_sentences = [
    "cane",
    "I cuccioli sono carini.",
    "Mi piace fare lunghe passeggiate lungo la spiaggia con il mio cane.",
]
japanese_sentences = ["犬", "子犬はいいです", "私は犬と一緒にビーチを散歩するのが好きです"]
italian_inputs = tokenizer(italian_sentences, return_tensors="pt", padding=True)
japanese_inputs = tokenizer(japanese_sentences, return_tensors="pt", padding=True)

with torch.no_grad():
    italian_outputs = model(**italian_inputs)
    japanese_outputs = model(**japanese_inputs)

italian_embeddings = italian_outputs.pooler_output
japanese_embeddings = japanese_outputs.pooler_output
```

For similarity between sentences, an L2-norm is recommended before calculating the similarity:

```python
import torch.nn.functional as F


def similarity(embeddings_1, embeddings_2):
    normalized_embeddings_1 = F.normalize(embeddings_1, p=2)
    normalized_embeddings_2 = F.normalize(embeddings_2, p=2)
    return torch.matmul(
        normalized_embeddings_1, normalized_embeddings_2.transpose(0, 1)
    )


print(similarity(english_embeddings, italian_embeddings))
print(similarity(english_embeddings, japanese_embeddings))
print(similarity(italian_embeddings, japanese_embeddings))
```

## Details

Details about data, training, evaluation and performance metrics are available in the [original paper](https://arxiv.org/abs/2302.08387).

### BibTeX entry and citation info

```bibtex
@inproceedings{mao-nakagawa-2023-lealla,
    title = "{LEALLA}: Learning Lightweight Language-agnostic Sentence Embeddings with Knowledge Distillation",
    author = "Mao, Zhuoyuan  and
      Nakagawa, Tetsuji",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.eacl-main.138",
    doi = "10.18653/v1/2023.eacl-main.138",
    pages = "1886--1894",
    abstract = "Large-scale language-agnostic sentence embedding models such as LaBSE (Feng et al., 2022) obtain state-of-the-art performance for parallel sentence alignment. However, these large-scale models can suffer from inference speed and computation overhead. This study systematically explores learning language-agnostic sentence embeddings with lightweight models. We demonstrate that a thin-deep encoder can construct robust low-dimensional sentence embeddings for 109 languages. With our proposed distillation methods, we achieve further improvements by incorporating knowledge from a teacher model. Empirical results on Tatoeba, United Nations, and BUCC show the effectiveness of our lightweight models. We release our lightweight language-agnostic sentence embedding models LEALLA on TensorFlow Hub.",
}
```
