# LaBSE

## Project

This project is an implementation to convert Google's [LaBSE](https://tfhub.dev/google/LaBSE/2) model from TensorFlow to PyTorch. It also offers extensions to convert the [smaller-LaBSE model](https://tfhub.dev/jeongukjae/smaller_LaBSE_15lang/1) from TensorFlow to PyTorch.

The models are uploaded to the [HuggingFace Model Hub](https://huggingface.co/setu4993/) in the PyTorch, HF-compatible TensorFlow and Flax formats, alongwith a compatible tokenizer.

- [LaBSE](https://huggingface.co/setu4993/LaBSE)
- [smaller-LaBSE](https://huggingface.co/setu4993/smaller-LaBSE)

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
poetry run convert_labse --output_path /path/to/model/LaBSE --huggingface_path
```

## Model Cards

See the [`model-cards` directory](https://github.com/setu4993/convert-labse-tf-pt/tree/main/model-cards) for a copy of the model cards.

## License

This repository and the conversion code is licensed under the MIT license, but the **model** is distributed with an Apache-2.0 license.
