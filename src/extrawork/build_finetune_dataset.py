from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a finetune dataset by combining a new dataset with sampled replay data from a base dataset."
    )
    parser.add_argument("--new-dataset-root", type=Path, required=True)
    parser.add_argument("--base-dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--replay-ratio",
        type=float,
        default=1.0,
        help="How many base samples to add relative to the new dataset size for each split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def npz_files(split_root: Path) -> list[Path]:
    return sorted(split_root.glob("*.npz"))


def ensure_empty_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            import shutil

            shutil.rmtree(child)


def hardlink_into(files: list[Path], target_root: Path, source_label: str) -> list[dict[str, str]]:
    manifest = []
    for file_path in files:
        target_path = target_root / file_path.name
        os.link(file_path, target_path)
        manifest.append(
            {
                "filename": file_path.name,
                "source_dataset": source_label,
                "source_path": str(file_path),
            }
        )
    return manifest


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_root = args.output_root
    train_root = output_root / "train"
    test_root = output_root / "test"
    ensure_empty_dir(train_root)
    ensure_empty_dir(test_root)

    new_train = npz_files(args.new_dataset_root / "train")
    new_test = npz_files(args.new_dataset_root / "test")
    base_train = npz_files(args.base_dataset_root / "train")
    base_test = npz_files(args.base_dataset_root / "test")

    replay_train_count = min(len(base_train), int(round(len(new_train) * args.replay_ratio)))
    replay_test_count = min(len(base_test), int(round(len(new_test) * args.replay_ratio)))
    replay_train = sorted(rng.sample(base_train, replay_train_count))
    replay_test = sorted(rng.sample(base_test, replay_test_count))

    manifest = {
        "new_dataset_root": str(args.new_dataset_root),
        "base_dataset_root": str(args.base_dataset_root),
        "output_root": str(output_root),
        "replay_ratio": args.replay_ratio,
        "seed": args.seed,
        "train": [],
        "test": [],
    }

    manifest["train"].extend(hardlink_into(new_train, train_root, args.new_dataset_root.name))
    manifest["train"].extend(hardlink_into(replay_train, train_root, args.base_dataset_root.name))
    manifest["test"].extend(hardlink_into(new_test, test_root, args.new_dataset_root.name))
    manifest["test"].extend(hardlink_into(replay_test, test_root, args.base_dataset_root.name))

    summary = {
        "new_train_count": len(new_train),
        "new_test_count": len(new_test),
        "replay_train_count": len(replay_train),
        "replay_test_count": len(replay_test),
        "combined_train_count": len(manifest["train"]),
        "combined_test_count": len(manifest["test"]),
    }
    manifest["summary"] = summary

    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
