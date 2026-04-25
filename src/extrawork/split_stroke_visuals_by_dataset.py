from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split stroke_visuals_by_char directories into train/test views using dataset sample ids."
    )
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--link-mode",
        choices=["junction", "copy"],
        default="junction",
        help="Use directory junctions by default to avoid duplicating image files.",
    )
    return parser.parse_args()


def sample_ids(split_dir: Path) -> set[str]:
    return {path.stem for path in split_dir.glob("*.npz")}


def visual_sample_id(folder_name: str) -> str | None:
    if "_" not in folder_name:
        return None
    return folder_name.split("_", 1)[1]


def ensure_link(source: Path, target: Path, link_mode: str) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if link_mode == "copy":
        import shutil

        shutil.copytree(source, target)
        return

    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(target), str(source)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> None:
    args = parse_args()
    visual_root = args.visual_root
    dataset_root = args.dataset_root
    output_root = args.output_root

    train_ids = sample_ids(dataset_root / "train")
    test_ids = sample_ids(dataset_root / "test")
    visual_dirs = [path for path in visual_root.iterdir() if path.is_dir()]

    summary: dict[str, object] = {
        "visual_root": str(visual_root),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "link_mode": args.link_mode,
        "visual_dir_count": len(visual_dirs),
        "train_dataset_count": len(train_ids),
        "test_dataset_count": len(test_ids),
        "matched_train": 0,
        "matched_test": 0,
        "unmatched_visual_dirs": [],
        "missing_train_visuals": [],
        "missing_test_visuals": [],
    }

    matched_train_ids: set[str] = set()
    matched_test_ids: set[str] = set()

    for visual_dir in visual_dirs:
        sample_id = visual_sample_id(visual_dir.name)
        if sample_id is None:
            summary["unmatched_visual_dirs"].append(visual_dir.name)
            continue

        if sample_id in train_ids:
            ensure_link(visual_dir, output_root / "train" / visual_dir.name, args.link_mode)
            summary["matched_train"] += 1
            matched_train_ids.add(sample_id)
        elif sample_id in test_ids:
            ensure_link(visual_dir, output_root / "test" / visual_dir.name, args.link_mode)
            summary["matched_test"] += 1
            matched_test_ids.add(sample_id)
        else:
            summary["unmatched_visual_dirs"].append(visual_dir.name)

    summary["missing_train_visuals"] = sorted(train_ids - matched_train_ids)[:50]
    summary["missing_test_visuals"] = sorted(test_ids - matched_test_ids)[:50]
    summary["missing_train_visual_count"] = len(train_ids - matched_train_ids)
    summary["missing_test_visual_count"] = len(test_ids - matched_test_ids)

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "split_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
