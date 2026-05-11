"""Download HatefulMemes images from the Cauldron and save as {id}.png.

Cauldron row N maps 1:1 to data.json id N (verified by byte-identical hashes
on existing local images at ids 3, 100, 1018, 8497).
"""
import argparse
import os

from datasets import load_dataset
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_IMAGE_DIR = os.path.join(REPO_ROOT, "data", "images")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config", default="hateful_memes")
    parser.add_argument("--repo", default="HuggingFaceM4/the_cauldron")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ds = load_dataset(args.repo, args.config, split="train")
    n = len(ds)

    saved = skipped = 0
    for i in tqdm(range(n)):
        out_path = os.path.join(args.out_dir, f"{i}.png")
        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            continue
        img = ds[i]["images"][0]
        img.save(out_path, "PNG")
        saved += 1

    print(f"Saved {saved}, skipped {skipped}, total {n}")


if __name__ == "__main__":
    main()
