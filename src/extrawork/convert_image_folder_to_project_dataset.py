from __future__ import annotations

import argparse
import colorsys
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from project_paths import RAW_DATA_ROOT


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIR = RAW_DATA_ROOT / "image"
DEFAULT_OUTPUT_DIR = RAW_DATA_ROOT / "data_anal" / "converted_from_image"

STYLE_CHANNELS = 8
IMAGE_SIZE = 256
DEFAULT_CENTER = np.array([127.5, 127.5], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert annotated character images into the npz/npy layout used by this project."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Folder that contains whole-character PNG files and a strokes subfolder.",
    )
    parser.add_argument(
        "--sample-id",
        required=True,
        help="Sample stem, for example 0_1. The script expects 0_1.png and strokes/0_1_stroke_*.png.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used to save the converted sample.",
    )
    parser.add_argument(
        "--char-name",
        default=None,
        help="Optional character name stored in the output npz. Defaults to sample-id.",
    )
    parser.add_argument(
        "--stroke-labels",
        default=None,
        help="Comma-separated 24-class stroke labels, one integer per stroke image.",
    )
    parser.add_argument(
        "--stroke-names",
        default=None,
        help="Optional comma-separated stroke names, stored only for readability.",
    )
    parser.add_argument(
        "--use-target-as-reference",
        action="store_true",
        help="Reuse target stroke masks as reference stroke masks when no separate reference data exists.",
    )
    return parser.parse_args()


def load_rgba(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGBA")
    array = np.asarray(image)
    if array.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"{path} must be {IMAGE_SIZE}x{IMAGE_SIZE}, got {array.shape[:2]}")
    return array


def rgba_to_bool_mask(image: np.ndarray) -> np.ndarray:
    alpha_mask = image[:, :, 3] > 0
    rgb = image[:, :, :3]
    non_white = np.any(rgb < 250, axis=2)
    return (alpha_mask & non_white).astype(bool)


def detect_char_color(rgb: np.ndarray) -> np.ndarray:
    flat = rgb.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    non_white = ~(colors >= 250).all(axis=1)
    colors = colors[non_white]
    counts = counts[non_white]
    if len(colors) == 0:
        raise ValueError("stroke image does not contain any foreground color")

    is_gray = (colors[:, 0] == colors[:, 1]) & (colors[:, 1] == colors[:, 2])
    gray_colors = colors[is_gray]
    gray_counts = counts[is_gray]
    if len(gray_colors) > 0:
        return gray_colors[np.argmax(gray_counts)]
    return colors[np.argmax(counts)]


def extract_target_stroke_mask(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(Image.open(path).convert("RGB"))
    if rgb.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"{path} must be {IMAGE_SIZE}x{IMAGE_SIZE}, got {rgb.shape[:2]}")

    char_color = detect_char_color(rgb)
    is_white = (rgb >= 250).all(axis=2)
    is_char = np.all(rgb == char_color.reshape(1, 1, 3), axis=2)
    stroke_mask = (~is_white) & (~is_char)
    return stroke_mask.astype(bool), char_color


def compute_centroid(mask: np.ndarray) -> np.ndarray:
    points = np.where(mask > 0)
    if len(points[0]) == 0:
        return DEFAULT_CENTER.copy()
    center_y = float(points[0].mean())
    center_x = float(points[1].mean())
    return np.array([center_y, center_x], dtype=np.float32)


def random_colors(count: int) -> list[tuple[float, float, float]]:
    hsv = [(i / count, 1.0, 1.0) for i in range(count)]
    return [colorsys.hsv_to_rgb(*each) for each in hsv]


SEG_COLORS = random_colors(24)


def build_reference_color_image(single_masks: np.ndarray, stroke_labels: np.ndarray) -> np.ndarray:
    color = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    for index, stroke_mask in enumerate(single_masks):
        label = int(stroke_labels[index])
        rgb = SEG_COLORS[label % len(SEG_COLORS)]
        for channel in range(3):
            color[:, :, channel] = np.where(stroke_mask > 0.5, rgb[channel], color[:, :, channel])
    return color.transpose(2, 0, 1).astype(np.float32)


def build_style_array(target_mask: np.ndarray) -> np.ndarray:
    style = np.zeros((STYLE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    style[0] = target_mask
    return style


def parse_csv_items(raw_value: str | None) -> list[str]:
    if raw_value is None:
        return []
    parts = [part.strip() for part in raw_value.split(",")]
    return [part for part in parts if part]


def parse_stroke_labels(raw_value: str | None, stroke_count: int) -> tuple[np.ndarray, bool]:
    if raw_value is None:
        labels = np.arange(stroke_count, dtype=np.int64)
        return labels, True

    parts = parse_csv_items(raw_value)
    if len(parts) != stroke_count:
        raise ValueError(f"stroke label count mismatch: expected {stroke_count}, got {len(parts)}")

    labels = np.asarray([int(part) for part in parts], dtype=np.int64)
    if np.any(labels < 0) or np.any(labels > 23):
        raise ValueError("stroke labels must be in [0, 23]")
    return labels, False


def natural_stroke_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"_stroke_(\d+)", path.stem)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def save_numpy_outputs(
    output_dir: Path,
    sample_id: str,
    target_mask: np.ndarray,
    single_masks: np.ndarray,
    stroke_labels: np.ndarray,
    stroke_names: str,
    char_name: str,
    labels_are_placeholders: bool,
) -> dict[str, object]:
    reference_single_image = single_masks.copy()
    target_single_image = single_masks.copy()
    reference_single_centroid = np.stack([compute_centroid(mask) for mask in reference_single_image], axis=0)
    reference_color_image = build_reference_color_image(reference_single_image, stroke_labels)
    target_image = target_mask[np.newaxis, :, :]
    style = build_style_array(target_mask)

    sample_root = output_dir / sample_id
    sample_root.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        sample_root / f"{sample_id}.npz",
        name=np.array(char_name),
        stroke_name=np.array(stroke_names),
        stroke_label=stroke_labels.astype(np.int64),
        reference_color_image=reference_color_image,
        reference_single_image=reference_single_image.astype(bool),
        reference_single_centroid=reference_single_centroid.astype(np.float32),
        target_image=target_image.astype(bool),
        target_single_image=target_single_image.astype(bool),
    )

    np.save(sample_root / f"{sample_id}_kaiti_color.npy", reference_color_image)
    np.save(sample_root / f"{sample_id}_single.npy", reference_single_image.astype(bool))
    np.save(sample_root / f"{sample_id}_style_single.npy", target_single_image.astype(bool))
    np.save(sample_root / f"{sample_id}_style.npy", style)
    np.save(sample_root / f"{sample_id}_seg.npy", stroke_labels.astype(np.int64))

    meta = {
        "sample_id": sample_id,
        "char_name": char_name,
        "stroke_count": int(single_masks.shape[0]),
        "stroke_labels": stroke_labels.tolist(),
        "stroke_names": stroke_names,
        "labels_are_placeholders": labels_are_placeholders,
        "output_dir": str(sample_root),
        "notes": [
            "reference_single_image and target_single_image are identical in this export",
            "for real training, reference strokes should ideally come from a separate canonical decomposition",
        ],
    }
    (sample_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    args = parse_args()
    image_dir = args.image_dir
    output_dir = args.output_dir
    sample_id = args.sample_id

    whole_path = image_dir / f"{sample_id}.png"
    strokes_dir = image_dir / "strokes"
    stroke_paths = sorted(strokes_dir.glob(f"{sample_id}_stroke_*.png"), key=natural_stroke_sort_key)
    if not whole_path.exists():
        raise FileNotFoundError(f"whole-character image not found: {whole_path}")
    if not stroke_paths:
        raise FileNotFoundError(f"no stroke images found under: {strokes_dir}")

    target_mask = rgba_to_bool_mask(load_rgba(whole_path))
    single_masks = []
    char_colors = []
    for stroke_path in stroke_paths:
        stroke_mask, char_color = extract_target_stroke_mask(stroke_path)
        single_masks.append(stroke_mask)
        char_colors.append(char_color.tolist())
    single_masks_np = np.stack(single_masks, axis=0).astype(bool)

    stroke_labels, labels_are_placeholders = parse_stroke_labels(args.stroke_labels, single_masks_np.shape[0])
    stroke_name_items = parse_csv_items(args.stroke_names)
    stroke_names = ",".join(stroke_name_items)
    char_name = args.char_name or sample_id

    meta = save_numpy_outputs(
        output_dir=output_dir,
        sample_id=sample_id,
        target_mask=target_mask,
        single_masks=single_masks_np,
        stroke_labels=stroke_labels,
        stroke_names=stroke_names,
        char_name=char_name,
        labels_are_placeholders=labels_are_placeholders,
    )
    meta["char_gray_candidates"] = char_colors
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
