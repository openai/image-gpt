import argparse
import json
import os
import sys
import requests
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--download_dir", type=str, default="/root/downloads/")

    parser.add_argument("--bert", action="store_true", help="download a bert model (default: ar)")
    parser.add_argument("--model", type=str, choices=["s", "m", "l"], help="parameter counts are s:76M, m:455M, l:1362M")
    parser.add_argument("--ckpt", type=str, choices=["131000", "262000", "524000", "1000000"])
    parser.add_argument("--clusters", action="store_true", help="download the color clusters file")
    parser.add_argument("--dataset", type=str, choices=["imagenet", "cifar10"])

    args = parser.parse_args()
    print("input args:\n", json.dumps(vars(args), indent=4, separators=(",", ":")))
    return args


def main(args):
    if not os.path.exists(args.download_dir):
        os.makedirs(args.download_dir)

    urls = []

    # download the checkpoint
    if args.model and args.ckpt:
        base_url = f"https://openaipublic.blob.core.windows.net/image-gpt/checkpoints/igpt-{args.model}{'-bert' if args.bert else ''}/{args.ckpt}"

        size_to_shards = {"s": 32, "m": 32, "l": 64}
        shards = size_to_shards[args.model]

        for filename in [f"model.ckpt-{args.ckpt}.data-{i:05d}-of-{shards:05d}" for i in range(shards)]:
            urls.append(f"{base_url}/{filename}")
        urls.append(f"{base_url}/model.ckpt-{args.ckpt}.index")
        urls.append(f"{base_url}/model.ckpt-{args.ckpt}.meta")

    # download the color clusters file
    if args.clusters:
        urls.append("https://openaipublic.blob.core.windows.net/image-gpt/color-clusters/kmeans_centers.npy")

    # download color clustered dataset
    if args.dataset:
        for split in ["trX", "trY", "vaX", "vaY", "teX", "teY"]:
            urls.append(f"https://openaipublic.blob.core.windows.net/image-gpt/datasets/{args.dataset}_{split}.npy")

    # run the download
    for url in urls:
        filename = url.split("/")[-1]
        r = requests.get(url, stream=True)
        with open(f"{args.download_dir}/{filename}", "wb") as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(ncols=80, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
