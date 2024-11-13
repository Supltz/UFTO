from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset
import pandas as pd
from data_prep import RAF_GetLabel, Aff_GetLabel, Emotio_GetLabel, AU_GetLabel, load_emotion_data, load_attribute_data

class BaseDataset(Dataset):
    def __init__(self, data, data_path, csv_path=None, mode='train', label_function=None, dataset_type=None):
        self.data = data
        self.data_path = data_path
        self.mode = mode
        self.label_function = label_function
        self.dataset_type = dataset_type  # Add dataset_type to determine how to handle labels

        # Load the CSV file once if applicable
        if csv_path:
            self.df = pd.read_csv(f"{csv_path}{self.mode}.csv")
            self.df.set_index('name', inplace=True)
            if self.dataset_type == "AUDataset":
                self.df.index = [f"{name[:-4]}_aligned.jpg" for name in self.df.index]
        else:
            self.df = None

        # Define transformations
        self.transform_train = transforms.Compose([
            transforms.Resize((240, 240)),
            transforms.RandomCrop(224),
            transforms.RandomRotation((-20, 20)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform_val_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def load_image(self, idx):
        image_path = self.data_path + self.data[idx]
        image = Image.open(image_path).convert("RGB")
        return image

    def __getitem__(self, idx):
        image = self.load_image(idx)

        if self.mode == 'train':
            image = self.transform_train(image)
        else:
            image = self.transform_val_test(image)
        if self.df is not None:
            label = self.label_function(self.data[idx], self.df) if self.label_function else None
        else:
           label = self.label_function(self.data[idx]) if self.label_function else None 

        if label is not None:
            if self.dataset_type in ['AUDataset', 'EmotioDataset']:
    # Special handling for AU and Emotio datasets
                au_labels = torch.tensor(label['emotion'], dtype=torch.long)  # Convert AU labels to a tensor
                gender = torch.tensor(label['gender'], dtype=torch.long)
                age = torch.tensor(label['age'], dtype=torch.long)
                race = torch.tensor(label['race'], dtype=torch.long)

                # Combine all the labels into a single tensor
                label = torch.cat([au_labels, gender.unsqueeze(0), age.unsqueeze(0), race.unsqueeze(0)])
            else:
                # Default handling for RAF and AffectNet datasets
                label = torch.tensor([label["emotion"], label["gender"], label["age"], label["race"]], dtype=torch.long)

        return image, label

# RAF Dataset
class RAFDataset(BaseDataset):
    def __init__(self, data, path, mode):
        # Preload the data once
        emo_path = "/vol/lian/datasets/basic/EmoLabel/list_patition_label.txt"
        attr_path_template = "/vol/lian/datasets/basic/Annotation/manual/{}_manu_attri.txt"

        self.emotion_data = load_emotion_data(emo_path)
        self.attribute_data = load_attribute_data(attr_path_template)

        super().__init__(data, path, mode=mode, label_function=self.get_raf_label, dataset_type='RAFDataset')

    def get_raf_label(self, item):
        return RAF_GetLabel(item, self.emotion_data, self.attribute_data)

# AffectNet Dataset
class AffDataset(BaseDataset):
    def __init__(self, data, data_path, csv_path, mode):
        super().__init__(data, data_path, csv_path=csv_path, mode=mode, label_function=Aff_GetLabel, dataset_type='AffDataset')

# EmotioNet Dataset
class EmotioDataset(BaseDataset):
    def __init__(self, data, data_path, csv_path, mode):
        super().__init__(data, data_path, csv_path=csv_path, mode=mode, label_function=Emotio_GetLabel, dataset_type='EmotioDataset')

# AU Dataset
class AUDataset(BaseDataset):
    def __init__(self, data, data_path, csv_path, mode):
        super().__init__(data, data_path, csv_path=csv_path, mode=mode, label_function=AU_GetLabel, dataset_type='AUDataset')