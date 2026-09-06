"""
make_splits.py

Creates train/dev/test splits for the deepfake detector dataset.

CRITICAL DESIGN DECISION: We split by VIDEO ID, not by individual image.
If we split by image, frames from the same video (same person, same identity)
could appear in both train and test sets. The model could then learn to
recognize the PERSON rather than learning general forgery artifacts -- this
is a classic data leakage bug in face-based ML tasks, and it would make our
reported accuracy meaningless (it wouldn't reflect real-world performance on
unseen people).

Usage:
    python make_splits.py --real_dir data/faces_extracted/real --fake_dir data/faces_extracted/fake --output_dir data/splits
"""

import argparse
import random
from pathlib import Path
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Create train/dev/test splits by video ID.")
    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--fake_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_frac", type=float, default=0.70)
    parser.add_argument("--dev_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_video_id(filename):
    stem = Path(filename).stem
    parts = stem.split("_")
    video_id = "_".join(parts[:-1])
    return video_id


def collect_by_video(folder, label):
    folder = Path(folder)
    files = sorted(folder.glob("*.jpg"))
    records = []
    video_ids = set()
    for f in files:
        vid = get_video_id(f.name)
        video_ids.add(vid)
        records.append({"filepath": str(f), "video_id": vid, "label": label})
    return records, sorted(video_ids)


def split_video_ids(video_ids, train_frac, dev_frac, seed):
    ids = list(video_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(n * train_frac)
    n_dev = int(n * dev_frac)
    train_ids = set(ids[:n_train])
    dev_ids = set(ids[n_train:n_train + n_dev])
    test_ids = set(ids[n_train + n_dev:])
    return train_ids, dev_ids, test_ids


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    real_records, real_video_ids = collect_by_video(args.real_dir, "real")
    fake_records, fake_video_ids = collect_by_video(args.fake_dir, "fake")

    print(f"Real: {len(real_records)} images across {len(real_video_ids)} videos")
    print(f"Fake: {len(fake_records)} images across {len(fake_video_ids)} videos")

    real_train_ids, real_dev_ids, real_test_ids = split_video_ids(
        real_video_ids, args.train_frac, args.dev_frac, args.seed)
    fake_train_ids, fake_dev_ids, fake_test_ids = split_video_ids(
        fake_video_ids, args.train_frac, args.dev_frac, args.seed)

    all_records = real_records + fake_records

    def assign_split(record):
        vid = record["video_id"]
        if record["label"] == "real":
            if vid in real_train_ids:
                return "train"
            elif vid in real_dev_ids:
                return "dev"
            else:
                return "test"
        else:
            if vid in fake_train_ids:
                return "train"
            elif vid in fake_dev_ids:
                return "dev"
            else:
                return "test"

    for r in all_records:
        r["split"] = assign_split(r)

    df = pd.DataFrame(all_records)

    leak_check = df.groupby("video_id")["split"].nunique()
    leaked = leak_check[leak_check > 1]
    if len(leaked) > 0:
        raise RuntimeError(f"Data leakage detected! Video IDs in multiple splits: {list(leaked.index)}")
    print("Leakage check passed: no video_id appears in more than one split.")

    for split_name in ["train", "dev", "test"]:
        split_df = df[df["split"] == split_name]
        out_path = output_dir / f"{split_name}.csv"
        split_df[["filepath", "label"]].to_csv(out_path, index=False)
        n_real = (split_df["label"] == "real").sum()
        n_fake = (split_df["label"] == "fake").sum()
        n_videos = split_df["video_id"].nunique()
        print(f"{split_name}: {len(split_df)} images ({n_real} real, {n_fake} fake) "
              f"from {n_videos} videos -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()