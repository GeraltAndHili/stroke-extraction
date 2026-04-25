import argparse
from datetime import datetime

from project_paths import (
    INFERENCE_MODEL_ROOT,
    build_run_name,
    prepared_dataset_dir,
    training_model_dir,
    training_output_dir,
)
from train_ExtractNet import TrainExtractNet
from train_SDNet import TrainSDNet
from train_SegNet import TrainSegNet

DEFAULT_FINETUNE_DATASET = "NPZ4_1_replay_1x"
DEFAULT_SDNET_EPOCHS = 12
DEFAULT_SEGNET_EPOCHS = 6
DEFAULT_EXTRACTNET_EPOCHS = 10
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 5e-5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune from the current baseline checkpoints without overwriting the base outputs."
    )
    parser.add_argument("--dataset", default=DEFAULT_FINETUNE_DATASET)
    parser.add_argument("--run-tag", default=None, help="Optional suffix for finetune outputs. Defaults to a timestamp.")
    parser.add_argument("--sdnet-epochs", type=int, default=DEFAULT_SDNET_EPOCHS)
    parser.add_argument("--segnet-epochs", type=int, default=DEFAULT_SEGNET_EPOCHS)
    parser.add_argument("--extractnet-epochs", type=int, default=DEFAULT_EXTRACTNET_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    return parser.parse_args()


def default_run_tag():
    return datetime.now().strftime("finetune_%Y%m%d_%H%M%S")


def print_run_summary(dataset, run_tag, prepared_path, args):
    stage_names = {
        "SDNet": build_run_name("SDNet", dataset, run_tag),
        "SegNet": build_run_name("SegNet", dataset, run_tag),
        "ExtractNet": build_run_name("ExtractNet", dataset, run_tag),
    }
    print(f"Finetune dataset: {dataset}")
    print(f"Finetune tag: {run_tag}")
    print(f"Prepared dataset output: {prepared_path}")
    print(
        "Hyperparameters: "
        f"lr={args.learning_rate}, batch_size={args.batch_size}, "
        f"sdnet_epochs={args.sdnet_epochs}, segnet_epochs={args.segnet_epochs}, "
        f"extractnet_epochs={args.extractnet_epochs}"
    )
    for stage_name, run_name in stage_names.items():
        print(f"{stage_name} checkpoints: {training_model_dir(run_name)}")
        print(f"{stage_name} visuals: {training_output_dir(run_name)}")


def main():
    args = parse_args()
    run_tag = args.run_tag or default_run_tag()
    prepared_path = str(prepared_dataset_dir(args.dataset, run_tag))
    print_run_summary(args.dataset, run_tag, prepared_path, args)

    sdnet_run_name = build_run_name("SDNet", args.dataset, run_tag)
    segnet_run_name = build_run_name("SegNet", args.dataset, run_tag)
    extractnet_run_name = build_run_name("ExtractNet", args.dataset, run_tag)

    train_sdnet = TrainSDNet(dataset=args.dataset, run_name=sdnet_run_name)
    train_sdnet.load_model_parameter(INFERENCE_MODEL_ROOT / "sdnet_model.pth")
    train_sdnet.train_model(
        epochs=args.sdnet_epochs,
        init_learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )
    train_sdnet.calculate_prior_information_and_qualitative(save_path=prepared_path)

    train_segnet = TrainSegNet(dataset=args.dataset, run_name=segnet_run_name)
    train_segnet.load_model_parameter(INFERENCE_MODEL_ROOT / "model.pth")
    train_segnet.train_model(
        epochs=args.segnet_epochs,
        init_learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        dataset_path=prepared_path,
    )

    train_extractnet = TrainExtractNet(
        dataset=args.dataset,
        run_name=extractnet_run_name,
        segnet_run_name=segnet_run_name,
    )
    train_extractnet.load_model_parameter(INFERENCE_MODEL_ROOT / "model_extract.pth")
    train_extractnet.train_model(
        epochs=args.extractnet_epochs,
        init_learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        dataset=prepared_path,
    )


if __name__ == "__main__":
    main()
