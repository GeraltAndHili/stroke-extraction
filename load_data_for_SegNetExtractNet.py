import os
import os.path
import pickle
import random
import time

import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as pyimg
import torch
import torch.utils.data as data

from utils import seg_label_to7

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""
load data for SegNet and ExtractNet
"""


class SegNetExtractNetLoader(data.Dataset):
    def __init__(self, is_training, dataset_path, is_single=False, split=None, val_ratio=0.1, seed=42):
        self.is_training = is_training
        self.is_single = is_single
        split = split or ("train" if is_training else "test")
        self.path = self._build_split_paths(dataset_path, split, val_ratio=val_ratio, seed=seed)
        self.path = sorted(self.path, key=lambda x: x[-1])
        print(f"number of dataset: {len(self.path)}")

    @staticmethod
    def _sample_sort_key(file_name):
        sample_id = file_name[:-16]
        try:
            return (0, int(sample_id))
        except ValueError:
            return (1, sample_id)

    def _build_records(self, split_dir, file_names):
        return [[
            os.path.join(split_dir, each),
            os.path.join(split_dir, each[:-16] + "_style.npy"),
            os.path.join(split_dir, each[:-16] + "_seg.npy"),
            os.path.join(split_dir, each[:-16] + "_single.npy"),
            os.path.join(split_dir, each[:-16] + "_style_single.npy"),
            self._sample_sort_key(each),
        ] for each in file_names]

    def _build_split_paths(self, dataset_path, split, val_ratio=0.1, seed=42):
        if split == "test":
            split_dir = os.path.join(dataset_path, "test")
            file_names = [each for each in os.listdir(split_dir) if "color" in each]
            return self._build_records(split_dir, file_names)

        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split: {split}")

        train_dir = os.path.join(dataset_path, "train")
        file_names = sorted(each for each in os.listdir(train_dir) if "color" in each)
        if len(file_names) < 2:
            selected = file_names if split == "train" else []
            return self._build_records(train_dir, selected)

        shuffled_names = file_names[:]
        random.Random(seed).shuffle(shuffled_names)
        val_count = int(len(shuffled_names) * val_ratio)
        val_count = max(1, min(len(shuffled_names) - 1, val_count))
        val_names = set(shuffled_names[:val_count])
        if split == "train":
            selected = [each for each in file_names if each not in val_names]
        else:
            selected = [each for each in file_names if each in val_names]
        return self._build_records(train_dir, selected)

    def get_seg_image(self, reference_single, seg_label):
        reference_image = np.zeros(shape=(7, 256, 256), dtype=np.float32)
        for i in range(seg_label.shape[0]):
            id_7 = seg_label_to7(seg_label[i])
            reference_image[id_7] += reference_single[i]
        return np.clip(reference_image, 0, 1)

    def get_data(self, item):
        reference_color = np.load(self.path[item][0])
        target_style = np.load(self.path[item][1])
        seg_id = np.load(self.path[item][2])
        reference_transformed_single = np.load(self.path[item][3])
        target_single_stroke = np.load(self.path[item][4])

        target_image = target_style[:1]
        target_data = np.repeat(target_image, 3, axis=0).astype(np.float32)
        reference_segment_transformation_data = self.get_seg_image(reference_transformed_single, seg_id)
        label_seg = self.get_seg_image(target_single_stroke, seg_id)

        if self.is_single:
            return {
                "target_data": target_data,
                "reference_color": reference_color,
                "label_seg": label_seg,
                "reference_segment_transformation_data": reference_segment_transformation_data,
                "seg_id": seg_id,
                "reference_transformed_single": reference_transformed_single,
                "target_single_stroke": target_single_stroke,
            }

        return {
            "target_data": target_data,
            "reference_color": reference_color,
            "label_seg": label_seg,
        }

    def __len__(self):
        return len(self.path)

    def __getitem__(self, item):
        return self.get_data(item)
