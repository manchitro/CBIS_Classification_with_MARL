from os.path import join
from os import listdir
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
import tqdm
from skimage import exposure
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as tr

from transforms import Normalization


class CBISDataset(Dataset):
    def __init__(self, resource_path: str):
        super().__init__()

        self.__cbis_root_path = resource_path

        # get csv files
        self.dicom_info_csv = pd.read_csv(
            join(self.__cbis_root_path, "csv", "dicom_info.csv"), sep=","
        )
        mass_train_csv = pd.read_csv(
            join(self.__cbis_root_path, "csv", "mass_case_description_train_set.csv"), sep=","
        )
        mass_test_csv = pd.read_csv(
            join(self.__cbis_root_path, "csv", "mass_case_description_test_set.csv"), sep=","
        )
        calc_train_csv = pd.read_csv(
            join(self.__cbis_root_path, "csv", "calc_case_description_train_set.csv"), sep=","
        )
        calc_test_csv = pd.read_csv(
            join(self.__cbis_root_path, "csv", "calc_case_description_test_set.csv"), sep=","
        )

        tqdm.tqdm.pandas()

        # mass dataset
        self.dataset_mass_train = [
            (str(path), label)
            for path, label in zip(
                mass_train_csv["cropped image file path"].tolist(),
                mass_train_csv["pathology"].tolist()
            )
        ]
        self.dataset_mass_test = [
            (str(path), label)
            for path, label in zip(
                mass_test_csv["cropped image file path"].tolist(),
                mass_test_csv["pathology"].tolist()
            )
        ]

        # calc dataset
        self.dataset_calc_train = [
            (str(path), label)
            for path, label in zip(
                calc_train_csv["cropped image file path"].tolist(),
                calc_train_csv["pathology"].tolist()
            )
        ]
        self.dataset_calc_test = [
            (str(path), label)
            for path, label in zip(
                calc_test_csv["cropped image file path"].tolist(),
                calc_test_csv["pathology"].tolist()
            )
        ]

        self.n_dataset_mass_train = len(self.dataset_mass_train)
        self.n_dataset_mass_test = len(self.dataset_mass_test)
        # print('mass_train_csv length: ', self.n_dataset_mass_train)
        # print('mass_test_csv length: ', self.n_dataset_mass_test)
        print('mass_csv length: ', self.n_dataset_mass_test +
              self.n_dataset_mass_train)

        self.n_dataset_calc_train = len(self.dataset_calc_train)
        self.n_dataset_calc_test = len(self.dataset_calc_test)
        # print('calc_train_csv length: ', self.n_dataset_calc_train)
        # print('calc_test_csv length: ', self.n_dataset_calc_test)
        print('calc_csv length: ', self.n_dataset_calc_test +
              self.n_dataset_calc_train)

        self.dataset_mass = self.dataset_mass_train + self.dataset_mass_test
        self.dataset_calc = self.dataset_calc_train + self.dataset_calc_test
        self.n_mass_dataset = len(self.dataset_mass)
        self.n_calc_dataset = len(self.dataset_calc)

        self.original_dataset = self.dataset_mass
        print("original dataset length: ", len(self.original_dataset))

        # augmented datasets
        self.augments = ["original", "h_flip_90", "v_flip_90",
                         "h_flip_180", "v_flip_180", "h_flip_270", "v_flip_270"]
        self.augments_indices = []

        self.__dataset = []

        i = 0
        j = 0
        for augment in self.augments:
            self.__dataset = self.__dataset + self.original_dataset
            j = len(self.__dataset)
            self.augments_indices.append([i, j])
            i = j + 1

        self.n_dataset = len(self.__dataset)

        for i in range(len(self.augments)):
            print(self.augments[i], self.augments_indices[i])
        print("dataset length: ", self.n_dataset)

        self.class_to_idx = {
            "benign": 0,
            "malignant": 1,
        }

    def getCroppedImagePathFromCSVPath(self, path: str):
        components = patorch.split('/')
        folder_name = components[2]
        images = listdir(join(self.__cbis_root_path, "jpeg", folder_name))

        for image in images:
            index = self.dicom_info_csv.index[self.dicom_info_csv['image_path'].str.contains(
                folder_name+'/'+image)][0]
            if self.dicom_info_csv['SeriesDescription'][index] == 'cropped images':
                image_path = join('jpeg', folder_name, image)
                return image_path

    def cbis_pil_loader(self, path: str) -> Image.Image:
        f = open(path, 'rb')
        img = Image.open(f)
        resized_img = img.resize((200, 200))
        grey_img = resized_img.convert('L')
        grey_img_clahe = exposure.equalize_adapthist(
            np.array(grey_img), clip_limit=0.03)
        img_clahe_pil = Image.fromarray(
            (grey_img_clahe * 255).astype(np.uint8)).convert('L')
        f.close()
        return img_clahe_pil

    def __open_img(self, path: str, index: int) -> torch.Tensor:
        file = self.cbis_pil_loader(
            join(self.__cbis_root_path, self.getCroppedImagePathFromCSVPath(path)))

        transforms = tr.ToTensor()

        augment = "original"
        for i in range(len(self.augments_indices)):
            if self.augments_indices[i][0] <= index <= self.augments_indices[i][1]:
                augment = self.augments[i]

        if augment == "original":
            transforms = tr.Compose([
                tr.ToTensor()
            ])
        if augment == "h_flip_90":
            transforms = tr.Compose([
                tr.RandomHorizontalFlip(p=1),
                tr.RandomRotation(degrees=90),
                tr.ToTensor()
            ])
        if augment == "h_flip_180":
            transforms = tr.Compose([
                tr.RandomHorizontalFlip(p=1),
                tr.RandomRotation(degrees=180),
                tr.ToTensor()
            ])
        if augment == "h_flip_270":
            transforms = tr.Compose([
                tr.RandomHorizontalFlip(p=1),
                tr.RandomRotation(degrees=270),
                tr.ToTensor()
            ])
        if augment == "v_flip_90":
            transforms = tr.Compose([
                tr.RandomVerticalFlip(p=1),
                tr.RandomRotation(degrees=90),
                tr.ToTensor()
            ])
        if augment == "v_flip_180":
            transforms = tr.Compose([
                tr.RandomVerticalFlip(p=1),
                tr.RandomRotation(degrees=180),
                tr.ToTensor()
            ])
        if augment == "v_flip_270":
            transforms = tr.Compose([
                tr.RandomVerticalFlip(p=1),
                tr.RandomRotation(degrees=270),
                tr.ToTensor()
            ])

        augmented_image = transforms(file)

        normalization_transforms = tr.Compose([
            Normalization()
        ])

        normalized_image = normalization_transforms(augmented_image)

        return normalized_image

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path_csv = self.__dataset[index][0]
        # print('Getting file index: ', index)

        label = self.__dataset[index][1]
        label_to_index = 0
        if label.startswith('BENIGN'):
            label_to_index = 0
        elif label.startswith('MALIGNANT'):
            label_to_index = 1
        img = self.__open_img(img_path_csv, index)

        return img, torch.tensor(label_to_index)

    def __len__(self) -> int:
        return self.n_dataset
