from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_ROOT.parent

DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_ROOT = DATA_ROOT / "dataset"
PREPARED_DATA_ROOT = DATA_ROOT / "prepared"

ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
MODEL_ROOT = ARTIFACTS_ROOT / "models"
INFERENCE_MODEL_ROOT = MODEL_ROOT / "inference"
TRAINING_MODEL_ROOT = MODEL_ROOT / "training"

OUTPUT_ROOT = ARTIFACTS_ROOT / "outputs"
TRAINING_OUTPUT_ROOT = OUTPUT_ROOT / "training"
INFERENCE_OUTPUT_ROOT = OUTPUT_ROOT / "inference"

LOG_ROOT = PROJECT_ROOT / "logs"


def raw_dataset_dir(dataset_name: str) -> Path:
    return RAW_DATA_ROOT / dataset_name


def prepared_dataset_dir(dataset_name: str) -> Path:
    return PREPARED_DATA_ROOT / f"dataset_forSegNet_ExtractNet_{dataset_name}"


def training_model_dir(run_name: str) -> Path:
    return TRAINING_MODEL_ROOT / run_name


def training_output_dir(run_name: str) -> Path:
    return TRAINING_OUTPUT_ROOT / run_name
