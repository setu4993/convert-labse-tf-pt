# LaBSE

## Project

This project is an implementation to convert Google's [LaBSE](https://tfhub.dev/google/LaBSE/2) model from TensorFlow to PyTorch. It also offers extensions to convert the [smaller-LaBSE model](https://tfhub.dev/jeongukjae/smaller_LaBSE_15lang/1) from TensorFlow to PyTorch, and the [LEALLA family](https://tfhub.dev/google/collections/LEALLA/1) of models.

The models are uploaded to the [HuggingFace Model Hub](https://huggingface.co/setu4993/) in the PyTorch HF-compatible (original and `safetensors`), TensorFlow and Flax formats, alongwith a compatible tokenizer.

- [LaBSE](https://huggingface.co/setu4993/LaBSE)
- [smaller-LaBSE](https://huggingface.co/setu4993/smaller-LaBSE)
- [LEALLA-base](https://huggingface.co/setu4993/LEALLA-base)
- [LEALLA-small](https://huggingface.co/setu4993/LEALLA-small)
- [LEALLA-large](https://huggingface.co/setu4993/LEALLA-large)

## Export

To convert and export the models:

```shell
poetry install
poetry run convert_labse --output_path /path/to/models
```

To update the models on the [HuggingFace Model Hub](https://huggingface.co/setu4993/LaBSE):

```shell
# Clone the already uploaded models.
cd /path/to/model
git clone https://huggingface.co/setu4993/LaBSE.git

# Export models anew and update.
cd /path/to/repo
poetry install
poetry run convert_labse --output_path /path/to/models/LaBSE --huggingface_path
```

### Export Commands by Model

1. [LaBSE](https://huggingface.co/setu4993/LaBSE): `poetry run convert_labse --output_path /path/to/models/setu4993/LaBSE --huggingface_path`
2. [smaller-LaBSE](https://huggingface.co/setu4993/smaller-LaBSE): `poetry run convert_labse --output_path /path/to/models/setu4993/smaller-LaBSE --smaller --huggingface_path`
3. [LEALLA-base](https://huggingface.co/setu4993/LEALLA-base): `poetry run convert_lealla --size base --output_path /path/to/models/setu4993/LEALLA-base --huggingface_path`
4. [LEALLA-small](https://huggingface.co/setu4993/LEALLA-small): `poetry run convert_lealla --size small --output_path /path/to/models/setu4993/LEALLA-small --huggingface_path`
5. [LEALLA-large](https://huggingface.co/setu4993/LEALLA-large): `poetry run convert_lealla --size large --output_path /path/to/models/setu4993/LEALLA-large --huggingface_path`

## Model Cards

See the [`model-cards` directory](https://github.com/setu4993/convert-labse-tf-pt/tree/main/model-cards) for a copy of the model cards.

## License

This repository and the conversion code is licensed under the MIT license, but the **model** is distributed with an Apache-2.0 license.
