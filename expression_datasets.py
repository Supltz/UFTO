import os

from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from data_prep import (
    AU_GetLabel,
    AU_split,
    Aff_GetLabel,
    Emotio_GetLabel,
    EMOTIONET_AU_COLUMNS,
    EXCLUDED_RACE_GROUPS,
    RAF_GetLabel,
    RAFAU_AU_COLUMNS,
    RACE_LABEL_MAP,
    aggregate_age,
    split_dataset,
)


DATASET_METADATA = {
    "rafdb": {
        "task_type": "multiclass",
        "num_outputs": 7,
        "dataset_type": "RAFDataset",
        "sensitive_indices": {"gender": 1, "age": 2, "race": 3},
    },
    "affectnet": {
        "task_type": "multiclass",
        "num_outputs": 7,
        "dataset_type": "AffDataset",
        "sensitive_indices": {"gender": 1, "age": 2, "race": 3},
    },
    "emotionet": {
        "task_type": "multilabel",
        "num_outputs": len(EMOTIONET_AU_COLUMNS),
        "dataset_type": "EmotioDataset",
        "sensitive_indices": {"gender": len(EMOTIONET_AU_COLUMNS), "age": len(EMOTIONET_AU_COLUMNS) + 1, "race": len(EMOTIONET_AU_COLUMNS) + 2},
    },
    "rafau": {
        "task_type": "multilabel",
        "num_outputs": len(RAFAU_AU_COLUMNS),
        "dataset_type": "AUDataset",
        "sensitive_indices": {"gender": len(RAFAU_AU_COLUMNS), "age": len(RAFAU_AU_COLUMNS) + 1, "race": len(RAFAU_AU_COLUMNS) + 2},
    },
}


class BaseDataset(Dataset):
    def __init__(
        self,
        data,
        data_path,
        csv_path=None,
        mode="train",
        label_function=None,
        dataset_type=None,
    ):
        self.data = data
        self.data_path = data_path
        self.mode = mode
        self.label_function = label_function
        self.dataset_type = dataset_type

        if csv_path:
            csv_file = os.path.join(csv_path, f"{self.mode}.csv")
            self.df = pd.read_csv(csv_file)
            if "gender" in self.df.columns:
                self.df = self.df[self.df["gender"].astype(str) != "Unsure"].reset_index(drop=True)
            if "race" in self.df.columns:
                self.df = self.df[~self.df["race"].astype(str).isin(EXCLUDED_RACE_GROUPS)].reset_index(drop=True)
            self.df.set_index("name", inplace=True)
            if self.dataset_type == "AUDataset":
                self.df.index = [f"{name[:-4]}_aligned.jpg" for name in self.df.index]
                self.data = self.df.index.tolist()
            else:
                self.data = self.df.index.tolist()
        else:
            self.df = None

        self.transform_train = transforms.Compose(
            [
                transforms.Resize((240, 240)),
                transforms.RandomCrop(224),
                transforms.RandomRotation((-20, 20)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.transform_val_test = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        return len(self.data)

    def load_image(self, idx):
        image_path = os.path.join(self.data_path, self.data[idx])
        return Image.open(image_path).convert("RGB")

    def __getitem__(self, idx):
        image = self.load_image(idx)
        if self.mode == "train":
            image = self.transform_train(image)
        else:
            image = self.transform_val_test(image)

        if self.df is not None:
            label = self.label_function(self.data[idx], self.df)
        else:
            label = self.label_function(self.data[idx])

        if label is None:
            raise ValueError(f"Missing label for sample: {self.data[idx]}")

        if self.dataset_type in ["AUDataset", "EmotioDataset"]:
            au_labels = torch.tensor(label["emotion"], dtype=torch.float32)
            label_tensor = torch.cat(
                [
                    au_labels,
                    torch.tensor(
                        [label["gender"], label["age"], label["race"]],
                        dtype=torch.long,
                    ),
                ]
            )
        else:
            label_tensor = torch.tensor(
                [label["emotion"], label["gender"], label["age"], label["race"]],
                dtype=torch.long,
            )

        return image, label_tensor


class RAFDataset(BaseDataset):
    def __init__(self, data, path, csv_path, mode):
        super().__init__(
            data,
            path,
            csv_path=csv_path,
            mode=mode,
            label_function=RAF_GetLabel,
            dataset_type="RAFDataset",
        )


class AffDataset(BaseDataset):
    def __init__(self, data, data_path, csv_path, mode):
        super().__init__(
            data,
            data_path,
            csv_path=csv_path,
            mode=mode,
            label_function=Aff_GetLabel,
            dataset_type="AffDataset",
        )


class EmotioDataset(BaseDataset):
    def __init__(self, data, data_path, csv_path, mode):
        super().__init__(
            data,
            data_path,
            csv_path=csv_path,
            mode=mode,
            label_function=Emotio_GetLabel,
            dataset_type="EmotioDataset",
        )


class AUDataset(BaseDataset):
    def __init__(self, data, data_path, csv_path, mode):
        super().__init__(
            data,
            data_path,
            csv_path=csv_path,
            mode=mode,
            label_function=AU_GetLabel,
            dataset_type="AUDataset",
        )


def build_dataset_splits(dataset_name, config):
    dataset_name = dataset_name.lower()
    if dataset_name == "rafdb":
        train_files, val_files, test_files = split_dataset(config["csv_path"])
        return (
            RAFDataset(train_files, config["data_path"], config["csv_path"], "train"),
            RAFDataset(val_files, config["data_path"], config["csv_path"], "valid"),
            RAFDataset(test_files, config["data_path"], config["csv_path"], "test"),
        )
    if dataset_name == "affectnet":
        train_files, val_files, test_files = split_dataset(config["csv_path"])
        return (
            AffDataset(train_files, config["data_path"], config["csv_path"], "train"),
            AffDataset(val_files, config["data_path"], config["csv_path"], "valid"),
            AffDataset(test_files, config["data_path"], config["csv_path"], "test"),
        )
    if dataset_name == "emotionet":
        train_files, val_files, test_files = split_dataset(config["csv_path"])
        return (
            EmotioDataset(train_files, config["data_path"], config["csv_path"], "train"),
            EmotioDataset(val_files, config["data_path"], config["csv_path"], "valid"),
            EmotioDataset(test_files, config["data_path"], config["csv_path"], "test"),
        )
    if dataset_name == "rafau":
        train_files, val_files, test_files = AU_split(config["csv_path"])
        return (
            AUDataset(train_files, config["data_path"], config["csv_path"], "train"),
            AUDataset(val_files, config["data_path"], config["csv_path"], "valid"),
            AUDataset(test_files, config["data_path"], config["csv_path"], "test"),
        )
    raise ValueError(f"Unknown dataset: {dataset_name}")


def is_multilabel_task(dataset_name):
    return DATASET_METADATA[dataset_name]["task_type"] == "multilabel"


def extract_task_targets(labels, dataset_name):
    num_outputs = DATASET_METADATA[dataset_name]["num_outputs"]
    if is_multilabel_task(dataset_name):
        return labels[:, :num_outputs].float()
    return labels[:, 0].long()


def extract_sensitive_targets(labels, dataset_name):
    return {
        name: labels[:, index].long()
        for name, index in DATASET_METADATA[dataset_name]["sensitive_indices"].items()
    }


def infer_sensitive_num_classes(dataset, dataset_name):
    if getattr(dataset, "df", None) is not None:
        gender_map = {"Male": 0, "Female": 1}
        race_map = {label: index for index, label in RACE_LABEL_MAP.items()}

        gender_values = dataset.df["gender"].astype(str).map(gender_map).dropna()
        age_values = dataset.df["age"].apply(aggregate_age).dropna()
        race_values = dataset.df["race"].astype(str).map(race_map).dropna()

        return {
            "gender": int(gender_values.max()) + 1,
            "age": int(age_values.max()) + 1,
            "race": int(race_values.max()) + 1,
        }

    label_tensors = [dataset[idx][1] for idx in range(len(dataset))]
    stacked = torch.stack(label_tensors)
    sensitive_targets = extract_sensitive_targets(stacked, dataset_name)
    return {
        name: int(values.max().item()) + 1
        for name, values in sensitive_targets.items()
    }
