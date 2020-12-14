FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04
RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip3 install

# set directory
WORKDIR  /image-gpt/

# requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# copy executive files
COPY ["src/", "download.py",  "/src/"]

# download pre-trained model
RUN python3 /src/download.py --model s --ckpt 1000000

# download dataset
RUN python3 /src/download.py --model s --ckpt 1000000 --dataset imagenet

# download color clusters
RUN python3 /src/download.py --model s --ckpt 1000000 --dataset imagenet --clusters

# run iGPT-S
CMD python3 /src/run.py --eval --n_embd 512  --n_head 8  --n_layer 24
