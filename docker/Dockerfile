FROM python:3.6-stretch

RUN apt-get update && \
	apt-get install -y gcc make apt-transport-https ca-certificates build-essential

# check pyhon environment
RUN python3 --version
RUN pip3 --version

# set directory
WORKDIR  /image-gpt/src/model.py

# requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# download pre-trained model
RUN python download.py --model s --ckpt 1000000

# download dataset
RUN python download.py --model s --ckpt 1000000 --dataset imagenet

# run iGPT-S
CMD python src/run.py --eval --n_embd 512  --n_head 8  --n_layer 24
