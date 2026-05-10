# Stroke Extraction of Chinese Characters

PyTorch implementation of the AAAI 2023 oral paper
[Stroke Extraction of Chinese Character Based on Deep Structure Deformable Image Registration](https://ojs.aaai.org/index.php/AAAI/article/view/25220).

This repository focuses on a three-stage pipeline for extracting individual strokes
from handwritten Chinese characters:

1. `SDNet` estimates global and local deformation between the reference glyph and the target handwriting.
2. `SegNet` predicts stroke-category priors from the transformed reference.
3. `ExtractNet` refines each stroke into a binary stroke mask.

The code in this repo is organized around that pipeline and includes training,
batch inference, and result visualization scripts.

## Repository Layout

```text
.
├── batch_infer_test.py                         # batch inference on a test split
├── extraction_stroke_application_for_single_character_.py
├── main_train.py                              # end-to-end training entrypoint
├── train_SDNet.py
├── train_SegNet.py
├── train_ExtractNet.py
├── load_data_for_SDNet.py
├── load_data_for_SegNetExtractNet.py
├── visualize_infer_results.py                 # export per-stroke PNGs
├── visualize_infer_eval_style.py              # render overlay-style outputs
├── char_recognise/
├── content_net_model/
├── dataset/
├── model/
└── utils*.py
```

Large training artifacts, generated outputs, cached datasets, and model checkpoints are
excluded from version control through `.gitignore`.

## Environment

- Python 3.8 or newer
- PyTorch 1.9 or newer
- CUDA is recommended for training and for the default inference script

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset

The project uses the **Regular Handwriting Character Stroke Extraction Dataset (RHSEDB)**.
The original dataset description is kept in `dataset/Introduction of RHSEDB.md`.

Each `.npz` sample contains:

```text
name                         character name
stroke_name                  stroke names
stroke_label                 numeric stroke labels
reference_color_image        (3, 256, 256)
reference_single_image       (N, 256, 256)
reference_single_centroid    (N, 2)
target_image                 (1, 256, 256)
target_single_image          (N, 256, 256)
```

The original README referenced the RHSEDB download here:

- RHSEDB: https://drive.google.com/file/d/1Q8dxAgSUkLp8IDVdjb9RK4zwACFVhGvu/view?usp=drive_link
- VGG weights for `char_recognise`: https://drive.google.com/file/d/1UgE1iYv4r6sPsjMRb84ACCCLe5nYZtTb/view?usp=drive_link
- ContentNet weights: https://drive.google.com/file/d/1R2h-jDhv2pBHVEeBvFUfLH2jQ7qBCuXl/view?usp=drive_link

Expected local paths:

```text
dataset/npz_4_1/
char_recognise/out_vgg_bn/model/model.pth
content_net_model/out/model_content.pth
```

## Training

Run the full three-stage pipeline:

```bash
python main_train.py
```

Train modules separately when needed:

```bash
python train_SDNet.py
python train_SegNet.py
python train_ExtractNet.py
```

`main_train.py` currently uses `dataset='npz_4_1'` and produces:

- `out/SDNet`
- `out/SegNet_npz_4_1`
- `out/ExtractNet_npz_4_1`
- `dataset_forSegNet_ExtractNet_npz_4_1`

## Inference and Visualization

Run batch inference on the test split:

```bash
python batch_infer_test.py --input-dir dataset/npz_4_1/test --output-dir out/infer_test_npz_4_1
```

Convert predicted stroke masks into per-stroke PNG files:

```bash
python visualize_infer_results.py --result-dir out/infer_test_npz_4_1 --source-dir dataset/npz_4_1/test
```

Generate evaluation-style visualizations:

```bash
python visualize_infer_eval_style.py --result-dir out/infer_test_npz_4_1 --source-dir dataset/npz_4_1/test
```

For single-sample inference, see:

```text
extraction_stroke_application_for_single_character_.py
```

That script expects these trained checkpoints:

```text
model/sdnet_model.pth
model/model.pth
model/model_extract.pth
```

## Notes

- Public repository uploads should not include the full dataset, generated outputs, or large checkpoints.
- The default inference script sets `device = torch.device("cuda")`; use a CUDA-capable environment or adjust the script for CPU fallback if needed.

## Citation

If this repository helps your work, please cite:

```bibtex
@inproceedings{li2023stroke,
  title={Stroke Extraction of Chinese Character Based on Deep Structure Deformable Image Registration},
  author={Li, Meng and Yu, Yahan and Yang, Yi and Ren, Guanghao and Wang, Jian},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={1},
  pages={1360--1367},
  year={2023}
}
```
