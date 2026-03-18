#!/usr/bin/env python3
"""
Download the Alexandria-MP-20 dataset from Hugging Face.

The dataset combines the Materials Project (MP-20) and the Alexandria database,
providing 675,204 inorganic crystal structures labelled with formation energy,
energy above hull (E_hull), band gap, and crystallographic metadata.

  Training   : 540,162 structures
  Validation :  67,521 structures
  Test       :  67,521 structures
  Total      : 675,204 structures
  Elements   : 86

Dataset source: https://huggingface.co/datasets/OMatG/Alex-MP-20

Usage
-----
  python data/download_dataset.py --output_dir /data/alex_mp_20

  # To download only a small subset for testing:
  python data/download_dataset.py --output_dir /data/alex_mp_20 --subset 10000
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASET_REPO = "OMatG/Alex-MP-20"
DATASET_SPLITS = ["train", "validation", "test"]


def download(output_dir: Path, subset: int | None = None) -> None:
    """Download and save the Alexandria-MP-20 dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("The 'datasets' package is required: pip install datasets huggingface-hub")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading Alexandria-MP-20 from HuggingFace Hub → {output_dir}")
    logger.info(f"Repository : {DATASET_REPO}")
    if subset:
        logger.info(f"Subset     : first {subset} examples per split")

    for split in DATASET_SPLITS:
        logger.info(f"  Fetching split: {split} ...")
        try:
            ds = load_dataset(DATASET_REPO, split=split, trust_remote_code=True)
            if subset:
                ds = ds.select(range(min(subset, len(ds))))

            split_dir = output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(split_dir))
            logger.info(f"  Saved {len(ds):,} structures → {split_dir}")

        except Exception as e:
            logger.error(f"  Failed to download split '{split}': {e}")
            raise

    logger.info(f"\nDataset download complete → {output_dir}")
    logger.info("Pass --dataset_path to the training scripts to use this dataset.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the Alexandria-MP-20 crystal structure dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("data/alex_mp_20"),
        help="Directory to save the downloaded dataset.",
    )
    parser.add_argument(
        "--subset", type=int, default=None,
        help="If set, download only the first N examples per split (for testing).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download(output_dir=args.output_dir, subset=args.subset)
