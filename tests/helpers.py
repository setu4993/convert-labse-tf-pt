from types import SimpleNamespace

Sentences = SimpleNamespace(
    similar=["Hi, how are you?", "Hello, how are you doing?"],
    english=[
        "dog",
        "Puppies are nice.",
        "I enjoy taking long walks along the beach with my dog.",
    ],
    italian=[
        "cane",
        "I cuccioli sono carini.",
        "Mi piace fare lunghe passeggiate lungo la spiaggia con il mio cane.",
    ],
    japanese=["犬", "子犬はいいです", "私は犬と一緒にビーチを散歩するのが好きです"],
)


def tf_model_output(sentences, hub_model, hf_tokenizer, smaller: bool = False):
    sentences = [sentences] if isinstance(sentences, str) else sentences
    tf_tokenized = hf_tokenizer(sentences, return_tensors="tf", padding="max_length")
    key = "default" if not smaller else "pooled_output"
    return hub_model(
        {
            "input_type_ids": tf_tokenized.token_type_ids,
            "input_mask": tf_tokenized.attention_mask,
            "input_word_ids": tf_tokenized.input_ids,
        }
    )[key]
