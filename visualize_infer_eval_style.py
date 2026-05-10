import argparse
import random
from pathlib import Path

import numpy as np
import torch

from utils import apply_stroke_t, random_colors, save_picture


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render test inference outputs using the same 4-row layout as ExtractNet val visuals."
    )
    parser.add_argument(
        "--result-dir",
        default="out/infer_test_npz_4_1",
        help="Directory containing *_extract_result.npz files.",
    )
    parser.add_argument(
        "--source-dir",
        default="dataset/npz_4_1/test",
        help="Directory containing source test .npz files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to <result-dir>_eval_style.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of samples to render.",
    )
    return parser.parse_args()


def shuffle_color(colors, step=3):
    shuffled = []
    for index in range(step):
        offset = 0
        while index + offset * step < len(colors):
            shuffled.append(colors[index + offset * step])
            offset += 1
    return shuffled


def build_overlay(base_image, stroke_masks):
    overlay = np.zeros(shape=(256, 256, 3), dtype=np.float32) + base_image.transpose(1, 2, 0)
    colors = shuffle_color(random_colors(len(stroke_masks)))
    random.shuffle(colors)
    for index, stroke_mask in enumerate(stroke_masks):
        overlay = apply_stroke_t(overlay, stroke_mask > 0.5, colors[index])
    return overlay


def render_sample(result_path, source_dir, output_dir):
    sample_id = result_path.name.replace("_extract_result.npz", "")
    source_path = source_dir / f"{sample_id}.npz"
    if not source_path.exists():
        print(f"skip missing source sample: {source_path}")
        return

    result_data = np.load(result_path)
    source_data = np.load(source_path)

    extract_result = result_data["extract_result"]
    target_image = source_data["target_image"].astype(np.float32)
    target_single_image = source_data["target_single_image"].astype(np.float32)
    reference_color = source_data["reference_color_image"].astype(np.float32)

    extract_result_show = build_overlay(target_image, extract_result)
    label_result_show = build_overlay(target_image, target_single_image)

    save_list = [
        torch.from_numpy(reference_color).unsqueeze(0).repeat(2, 1, 1, 1),
        torch.from_numpy(np.repeat(target_image, 3, axis=0)).unsqueeze(0).repeat(2, 1, 1, 1),
        torch.from_numpy(label_result_show.transpose(2, 0, 1)).unsqueeze(0).repeat(2, 1, 1, 1),
        torch.from_numpy(extract_result_show.transpose(2, 0, 1)).unsqueeze(0).repeat(2, 1, 1, 1),
    ]
    title_list = ["reference_color", "target_data", "stroke_label", "stroke_extraction"]
    out_path = output_dir / f"{sample_id}.bmp"
    save_picture(*save_list, title_list=title_list, path=str(out_path), nrow=2)
    print(f"saved {out_path.name}")


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir) if args.output_dir else result_dir.parent / f"{result_dir.name}_eval_style"
    output_dir.mkdir(parents=True, exist_ok=True)

    result_paths = sorted(
        result_dir.glob("*_extract_result.npz"),
        key=lambda path: path.name.replace("_extract_result.npz", ""),
    )
    if args.limit is not None:
        result_paths = result_paths[: args.limit]

    if not result_paths:
        print(f"no extract_result files found in: {result_dir}")
        return

    for result_path in result_paths:
        render_sample(result_path, source_dir, output_dir)

    print(f"all done, outputs saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
