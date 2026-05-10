import argparse
from pathlib import Path

import numpy as np

from extraction_stroke_application_for_single_character_ import ExtractStroke


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run batch inference on a test dataset split and save *_extract_result.npz files."
    )
    parser.add_argument(
        "--input-dir",
        default="dataset/npz_4_1/test",
        help="Directory containing test .npz samples.",
    )
    parser.add_argument(
        "--output-dir",
        default="out/infer_test_npz_4_1",
        help="Directory used to save *_extract_result.npz outputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of samples to process.",
    )
    return parser.parse_args()


def build_input_data(model, data):
    reference_color_image, reference_single_centroid = model.get_reference_data(
        data["reference_single_image"], data["stroke_label"]
    )
    return {
        "target_image": data["target_image"],
        "reference_single_image": data["reference_single_image"],
        "reference_color_image": reference_color_image,
        "reference_single_centroid": reference_single_centroid,
        "stroke_label": data["stroke_label"],
    }


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_paths = sorted(input_dir.glob("*.npz"))
    if args.limit is not None:
        sample_paths = sample_paths[: args.limit]

    if not sample_paths:
        raise ValueError(f"No .npz files found under {input_dir}")

    print(f"found {len(sample_paths)} test samples")
    model = ExtractStroke()

    for index, sample_path in enumerate(sample_paths, start=1):
        data = np.load(sample_path)
        input_data = build_input_data(model, data)
        extract_result = model.get_extract_strokes(input_data)
        output_path = output_dir / f"{sample_path.stem}_extract_result.npz"
        np.savez_compressed(
            output_path,
            extract_result=np.asarray(extract_result, dtype=np.uint8),
            sample_id=sample_path.stem,
        )
        print(f"[{index}/{len(sample_paths)}] saved {output_path.name}")

    print(f"done: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
