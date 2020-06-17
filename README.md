**Status:** Archive (code is provided as-is, no updates expected)

# image-gpt

Code and models from the paper ["Generative Pretraining from Pixels"](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf).

Supported Platforms:

- Ubuntu 16.04

## Install

You can get miniconda from https://docs.conda.io/en/latest/miniconda.html, or install the dependencies shown below manually.

```
conda create --name image-gpt python=3.7.3
conda activate image-gpt

conda install numpy=1.16.3
conda install tensorflow-gpu=1.13.1

conda install imageio=2.8.0
conda install requests=2.21.0
conda install tqdm=4.46.0
```

## Usage

This repository is meant to be a starting point for researchers and engineers to experiment with image GPT (iGPT). Our code forks GPT-2 to highlight that it can be easily applied across domains. The diff from `gpt-2/src/model.py` to `image-gpt/src/model.py` includes a new activation function, renaming of several variables, and the introduction of a start-of-sequence token, none of which change the model architecture.

### Downloading Pre-trained Models

To download a model checkpoint, run `download.py`. The `--model` argument should be one of "s", "m", or "l", and the `--ckpt` argument should be one of "131000", "262000", "524000", or "1000000".

```
python download.py --model s --ckpt 1000000
```

This command downloads the iGPT-S checkpoint at 1M training iterations. The default download directory is set to `/root/downloads/`, and can be changed using the `--download_dir` argument.

### Downloading Datasets

To download datasets, run `download.py` with the `--dataset` argument set to "imagenet" or "cifar10".

```
python download.py --model s --ckpt 1000000 --dataset imagenet
```

This command additionally downloads 32x32 ImageNet encoded with the 9-bit color palette described in the paper. The datasets we provide are center-cropped images intended for evaluation; random cropped images are required to faithfully replicate training.

### Downloading Color Clusters

To download the color cluster file defining our 9-bit color palette, run `download.py` with the `--clusters` flag set.

```
python download.py --model s --ckpt 1000000 --dataset imagenet --clusters
```

This command additionally downloads the color cluster file. `src/run.py:sample` shows how to decode from 9-bit color to RGB and `src/utils.py:color_quantize` shows how to go the other way around.

### Sampling

Once the desired checkpoint and color cluster file are downloaded, we can run the script in sampling mode. The following commands sample from iGPT-S, iGPT-M, and iGPT-L respectively:

```
python src/run.py --sample --n_embd 512  --n_head 8  --n_layer 24
python src/run.py --sample --n_embd 1024 --n_head 8  --n_layer 36
python src/run.py --sample --n_embd 1536 --n_head 16 --n_layer 48
```

If your data is not in `/root/downloads/`, set `--ckpt_path` and `--color_cluster_path` manually. To run on fewer than 8 GPUs, use a command of the following form:

```
CUDA_VISIBLE_DEVICES=0,1 python src/run.py --sample --n_embd 512  --n_head 8  --n_layer 24 --n_gpu 2
```

### Evaluating

Once the desired checkpoint and evaluation dataset are downloaded, we can run the script in evaluation mode. The following commands evaluate iGPT-S, iGPT-M, and iGPT-L on ImageNet respectively:

```
python src/run.py --eval --n_embd 512  --n_head 8  --n_layer 24
python src/run.py --eval --n_embd 1024 --n_head 8  --n_layer 36
python src/run.py --eval --n_embd 1536 --n_head 16 --n_layer 48
```

If your data is not in `/root/downloads/`, set `--ckpt_path` and `--data_path` manually. You should see that the test generative losses are 2.0895, 2.0614, and 2.0466, matching Figure 3 in the paper.

### Citation

Please use the following bibtex entry:
```
@article{chen2020generative,
  title={Generative Pretraining from Pixels},
  author={Chen, Mark and Radford, Alec and Child, Rewon and Wu, Jeff and Jun, Heewoo and Dhariwal, Prafulla and Luan, David and Sutskever, Ilya},
  year={2020}
}
```

## License

[Modified MIT](./LICENSE)
