from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
from PIL import Image
from project_paths import RAW_DATA_ROOT


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHA_ROOT = RAW_DATA_ROOT / "cha_strokes"
INSTANCES_DIR = CHA_ROOT / "instances"
STROKES_DIR = CHA_ROOT / "strokes"
INSTANCES_INDEX = CHA_ROOT / "instances.json"
SAMPLE_LIMIT = 20


def list_stems(directory: Path, suffix: str) -> list[str]:
    return sorted(path.stem for path in directory.glob(f"*{suffix}") if path.name != ".DS_Store")


def load_instances_index() -> dict[str, str]:
    records = json.loads(INSTANCES_INDEX.read_text(encoding="utf-8"))
    return {record["id"]: record["character"] for record in records}


def load_rgba_mask(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("RGBA"))
    alpha = image[:, :, 3] > 0
    non_white = np.any(image[:, :, :3] < 250, axis=2)
    return alpha & non_white


def rasterize_strokes(json_path: Path) -> tuple[list[np.ndarray], Counter, list[int]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    stroke_masks: list[np.ndarray] = []
    type_counter: Counter = Counter()
    contour_lengths: list[int] = []

    for stroke in data:
        mask = np.zeros((256, 256), dtype=np.uint8)
        contour = np.asarray(stroke.get("contour", []), dtype=np.int32)
        if contour.size:
            contour = contour.reshape(-1, 1, 2)
            cv2.fillPoly(mask, [contour], 1)
            contour_lengths.append(int(contour.shape[0]))
        else:
            contour_lengths.append(0)
        stroke_masks.append(mask.astype(bool))

        for stroke_type in stroke.get("types", []):
            type_counter[stroke_type] += 1

    return stroke_masks, type_counter, contour_lengths


def analyze() -> dict[str, object]:
    image_stems = list_stems(INSTANCES_DIR, ".png")
    json_stems = list_stems(STROKES_DIR, ".json")
    index_map = load_instances_index()

    image_set = set(image_stems)
    json_set = set(json_stems)
    index_set = set(index_map)

    paired_stems = sorted(image_set & json_set)
    missing_json = sorted(image_set - json_set)
    missing_image = sorted(json_set - image_set)
    missing_index = sorted((image_set | json_set) - index_set)

    image_shapes: Counter = Counter()
    image_modes: Counter = Counter()
    stroke_counts: list[int] = []
    typed_stroke_counter: Counter = Counter()
    contour_lengths: list[int] = []
    exact_match = 0
    small_diff = 0
    large_diff = 0
    diff_samples: list[dict[str, object]] = []

    for stem in paired_stems:
        image_path = INSTANCES_DIR / f"{stem}.png"
        json_path = STROKES_DIR / f"{stem}.json"

        with Image.open(image_path) as image:
            image_shapes[str(np.asarray(image).shape)] += 1
            image_modes[image.mode] += 1

        target_mask = load_rgba_mask(image_path)
        stroke_masks, type_counter, stroke_contour_lengths = rasterize_strokes(json_path)
        stroke_counts.append(len(stroke_masks))
        typed_stroke_counter.update(type_counter)
        contour_lengths.extend(stroke_contour_lengths)

        union_mask = np.zeros((256, 256), dtype=bool)
        for stroke_mask in stroke_masks:
            union_mask |= stroke_mask

        diff_pixels = int(np.logical_xor(target_mask, union_mask).sum())
        if diff_pixels == 0:
            exact_match += 1
        elif diff_pixels <= 100:
            small_diff += 1
        else:
            large_diff += 1
        if diff_pixels > 0 and len(diff_samples) < SAMPLE_LIMIT:
            diff_samples.append(
                {
                    "id": stem,
                    "character": index_map.get(stem, ""),
                    "diff_pixels": diff_pixels,
                    "target_pixels": int(target_mask.sum()),
                    "union_pixels": int(union_mask.sum()),
                    "stroke_count": len(stroke_masks),
                }
            )

    strokes_with_types = sum(typed_stroke_counter.values())
    total_strokes = sum(stroke_counts)

    return {
        "counts": {
            "instance_png": len(image_stems),
            "stroke_json": len(json_stems),
            "instances_index": len(index_map),
            "paired_samples": len(paired_stems),
            "name_covered_by_index": len(paired_stems) - len(missing_index),
            "missing_json_for_instance": len(missing_json),
            "missing_instance_for_json": len(missing_image),
            "missing_index_entries": len(missing_index),
        },
        "missing_examples": {
            "missing_json": missing_json[:SAMPLE_LIMIT],
            "missing_image": missing_image[:SAMPLE_LIMIT],
            "missing_index": missing_index[:SAMPLE_LIMIT],
        },
        "image_stats": {
            "shapes": image_shapes.most_common(5),
            "modes": image_modes.most_common(5),
        },
        "stroke_stats": {
            "min_strokes": min(stroke_counts) if stroke_counts else 0,
            "max_strokes": max(stroke_counts) if stroke_counts else 0,
            "avg_strokes": mean(stroke_counts) if stroke_counts else 0,
            "min_contour_points": min(contour_lengths) if contour_lengths else 0,
            "max_contour_points": max(contour_lengths) if contour_lengths else 0,
            "avg_contour_points": mean(contour_lengths) if contour_lengths else 0,
        },
        "type_stats": {
            "total_strokes": total_strokes,
            "strokes_with_any_type": strokes_with_types,
            "strokes_without_type": total_strokes - strokes_with_types,
            "non_empty_type_ratio": (strokes_with_types / total_strokes) if total_strokes else 0,
            "unique_types": sorted(typed_stroke_counter),
            "top_types": typed_stroke_counter.most_common(20),
        },
        "rasterization_check": {
            "exact_match": exact_match,
            "small_diff_le_100": small_diff,
            "large_diff_gt_100": large_diff,
            "diff_examples": diff_samples,
        },
    }


def print_report(report: dict[str, object]) -> None:
    counts = report["counts"]
    image_stats = report["image_stats"]
    stroke_stats = report["stroke_stats"]
    type_stats = report["type_stats"]
    rasterization = report["rasterization_check"]

    print("Counts")
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    print()
    print("Image stats")
    print(json.dumps(image_stats, ensure_ascii=False, indent=2))
    print()
    print("Stroke stats")
    print(json.dumps(stroke_stats, ensure_ascii=False, indent=2))
    print()
    print("Type stats")
    print(json.dumps(type_stats, ensure_ascii=False, indent=2))
    print()
    print("Rasterization check")
    print(json.dumps(rasterization, ensure_ascii=False, indent=2))
    print()
    print("NPZ gap assessment")
    print(
        f"- name: only partially recoverable from instances.json "
        f"({counts['name_covered_by_index']}/{counts['paired_samples']} samples covered)"
    )
    print("- target_image: can be recovered from instances/*.png")
    print("- target_single_image: can be rasterized from strokes/*.json contour")
    print("- reference_single_centroid: can be computed from rasterized single-stroke masks")
    print("- stroke order: available from json list order / index")
    print("- stroke_name: not available as Chinese stroke-name sequence")
    print("- stroke_label (0..23): not available reliably from current data")
    print("- reference_single_image: missing if you need a separate reference decomposition")
    print("- reference_color_image: can only be synthesized after you have stroke_label")


if __name__ == "__main__":
    print_report(analyze())
