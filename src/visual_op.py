from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
from project_paths import INFERENCE_OUTPUT_ROOT, raw_dataset_dir


PROJECT_ROOT = Path(__file__).resolve().parent
BATCH_RESULT_DIR = INFERENCE_OUTPUT_ROOT / "batch_extract_test"
DATASET_TEST_DIR = raw_dataset_dir("RHSEDB") / "test"
OUTPUT_ROOT = BATCH_RESULT_DIR / "stroke_visuals_by_char"

BACKGROUND_COLOR = np.array([255, 255, 255], dtype=np.uint8)
CHAR_COLOR = np.array([190, 190, 190], dtype=np.uint8)
STROKE_COLOR = np.array([114, 183, 242], dtype=np.uint8)


def sanitize_folder_name(name: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*]', "_", name.strip())
    safe = safe.replace("\n", "_").replace("\r", "_")
    return safe or "unknown"


def get_char_name(source_data: np.lib.npyio.NpzFile) -> str:
    raw_name = source_data["name"]
    if hasattr(raw_name, "item"):
        raw_name = raw_name.item()
    return str(raw_name)


def render_stroke_image(target_image: np.ndarray, stroke_mask: np.ndarray) -> np.ndarray:
    canvas = np.full((256, 256, 3), BACKGROUND_COLOR, dtype=np.uint8)

    whole_char_mask = target_image > 0.5
    single_stroke_mask = stroke_mask > 0.5

    canvas[whole_char_mask] = CHAR_COLOR
    canvas[single_stroke_mask] = STROKE_COLOR
    return canvas


def save_image(image: np.ndarray, out_path: Path) -> None:
    success, encoded = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError(f'failed to encode image: {out_path}')
    encoded.tofile(str(out_path))


def save_stroke_visuals(result_path: Path) -> None:
    sample_id = result_path.name.replace("_extract_result.npz", "")
    source_path = DATASET_TEST_DIR / f"{sample_id}.npz"

    if not source_path.exists():
        print(f"skip missing source sample: {source_path}")
        return

    result_data = np.load(result_path)
    source_data = np.load(source_path)

    extract_result = result_data["extract_result"]
    target_image = source_data["target_image"][0]
    char_name = get_char_name(source_data)

    folder_name = sanitize_folder_name(f"{char_name}_{sample_id}")
    sample_out_dir = OUTPUT_ROOT / folder_name
    sample_out_dir.mkdir(parents=True, exist_ok=True)

    for idx, stroke_mask in enumerate(extract_result, start=1):
        image = render_stroke_image(target_image, stroke_mask)
        out_path = sample_out_dir / f"stroke_{idx:02d}.png"
        save_image(image, out_path)

    info_path = sample_out_dir / "meta.txt"
    info_path.write_text(
        f"sample_id={sample_id}\n"
        f"char_name={char_name}\n"
        f"stroke_count={extract_result.shape[0]}\n",
        encoding="utf-8",
    )

    print(f"saved {extract_result.shape[0]} stroke images -> {sample_out_dir}")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    result_paths = sorted(
        BATCH_RESULT_DIR.glob("*_extract_result.npz"),
        key=lambda p: int(p.name.replace("_extract_result.npz", "")),
    )

    if not result_paths:
        print(f"no extract_result files found in: {BATCH_RESULT_DIR}")
        return

    for result_path in result_paths:
        save_stroke_visuals(result_path)

    print(f"all done, outputs saved in: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
