FROM python:3.6-stretch

RUN apt-get update && \
	apt-get install -y gcc make apt-transport-https ca-certificates build-essential

# check pyhon environment
RUN python3 --version
RUN pip3 --version

# set directory
WORKDIR  /image-gpt/

# requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ["src/", "download.py",  "/src/"]
RUN ls -la /src/*

# download pre-trained model
RUN python /src/download.py --model s --ckpt 1000000

# download dataset
RUN python /src/download.py --model s --ckpt 1000000 --dataset imagenet

# download color clusters
RUN python /src/download.py --model s --ckpt 1000000 --dataset imagenet --clusters

# run iGPT-S
CMD python /src/run.py --eval --n_embd 512  --n_head 8  --n_layer 24
