import os
import os.path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
Load input data for SDNet
"""


class SDNetLoader(data.Dataset):
    def __init__(self, is_training, dataset_path, is_inference=False):
        self.is_inference = is_inference
        if is_training:
            split_dir = os.path.join(dataset_path, "train")
        else:
            split_dir = os.path.join(dataset_path, "test")
        self.path = [os.path.join(split_dir, each) for each in os.listdir(split_dir)]
        print(f"number of dataset: {len(self.path)}")

    def get_data(self, item):
        data_frame = np.load(self.path[item])
        reference_color_image = data_frame["reference_color_image"]
        reference_single_image = data_frame["reference_single_image"]
        reference_single_centroid = data_frame["reference_single_centroid"]
        target_image = data_frame["target_image"]
        target_single_image = data_frame["target_single_image"]
        stroke_label = data_frame["stroke_label"]

        stroke_num = reference_single_image.shape[0]
        expand_zeros = []
        expand_single_centroid = []

        for i in range(30):
            if i >= reference_single_image.shape[0]:
                expand_zeros.append(np.zeros(shape=(256, 256), dtype=float))
                expand_single_centroid.append(np.array([127.5, 127.5]))

        expand_zeros = np.array(expand_zeros)
        reference_single_image = np.concatenate([reference_single_image, expand_zeros], axis=0)
        target_single_image = np.concatenate([target_single_image, expand_zeros], axis=0)

        expand_single_centroid = np.array(expand_single_centroid)
        reference_single_centroid = np.concatenate([reference_single_centroid, expand_single_centroid], axis=0)

        result = {
            "target_single_stroke": target_single_image,
            "reference_single_stroke": reference_single_image,
            "target_data": target_image,
            "reference_color": reference_color_image,
            "stroke_num": stroke_num,
            "reference_single_stroke_centroid": reference_single_centroid,
        }
        if self.is_inference:
            result["stroke_label"] = stroke_label
        return result

    def __len__(self):
        return len(self.path)

    def __getitem__(self, item):
        return self.get_data(item)


def save_dict(obj, name):
    with open(name, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_dict(name):
    with open(name, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    c = SDNetLoader(is_training=False, dataset_path="dataset/CCSEDB")
    data_loader = data.DataLoader(c, batch_size=8, shuffle=False)

    for i in range(10):
        print(i)
        for i_batch, sample_batched in enumerate(data_loader):
            style_single_image = sample_batched["style_single_image"][0].numpy()
            kaiti_single_image = sample_batched["kaiti_single_image"][0].numpy()
            original = sample_batched["original_style"].numpy()
            kaiti_color = sample_batched["kaiti_color"].numpy()
            stroke_num = sample_batched["stroke_num"][0].numpy()
            kaiti_center = sample_batched["kaiti_center"].numpy()
            assert not np.isnan(kaiti_center).any()
            stroke_num = int(stroke_num)
            plt.figure(0)
            for j in range(8):
                plt.subplot2grid((2, 8), (0, j))
                plt.imshow(kaiti_color[j].transpose((1, 2, 0)))
                plt.subplot2grid((2, 8), (1, j))
                plt.imshow(original[j].squeeze())
            plt.show()
