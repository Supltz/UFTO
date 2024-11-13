import os
import numpy as np
from tqdm import tqdm
import pandas as pd

# Preload the emotion data into a dictionary
def load_emotion_data(emo_path):
    emo_dict = {}
    with open(emo_path, 'r') as f:
        for line in f:
            img, label = line.split(" ")
            identifier = "_".join(img.split("_")[:2])
            emo_dict[identifier] = int(label) - 1  # Subtracting 1 as in the original code
    return emo_dict

# Preload attribute data into a dictionary
def load_attribute_data(attr_path_template):
    attr_dict = {}
    for file_name in os.listdir(os.path.dirname(attr_path_template)):
        if file_name.endswith(".txt"):
            identifier = file_name.split("_")[0] + "_" + file_name.split("_")[1]
            attr_path = attr_path_template.format(identifier)
            if os.path.exists(attr_path):
                with open(attr_path, 'r') as f:
                    lines = f.readlines()[5:]  # Skip the first 5 lines
                    gender = int(lines[0].strip())
                    race = int(lines[1].strip())
                    age = int(lines[2].strip())
                    attr_dict[identifier] = {'gender': gender, 'race': race, 'age': age}
    return attr_dict

def get_names_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df["name"].tolist()


def split_dataset(csv_path):
    train_names = get_names_from_csv(os.path.join(csv_path, 'train.csv'))
    val_names = get_names_from_csv(os.path.join(csv_path, 'valid.csv'))
    test_names = get_names_from_csv(os.path.join(csv_path, 'test.csv'))
    return train_names, val_names, test_names

def AU_split(csv_path):
    train_names = [f"{name[:-4]}_aligned.jpg" for name in get_names_from_csv(os.path.join(csv_path, 'train.csv'))]
    val_names = [f"{name[:-4]}_aligned.jpg" for name in get_names_from_csv(os.path.join(csv_path, 'valid.csv'))]
    test_names = [f"{name[:-4]}_aligned.jpg" for name in get_names_from_csv(os.path.join(csv_path, 'test.csv'))]

    return train_names, val_names, test_names



def RAF_spliting(folder_path, csv_path, mode):
    # Read names from CSV files
    train_names, val_names, test_names = split_dataset(csv_path)

            # Preload the data once
    emo_path = "/vol/lian/datasets/basic/EmoLabel/list_patition_label.txt"
    attr_path_template = "/vol/lian/datasets/basic/Annotation/manual/{}_manu_attri.txt"

    emotion_data = load_emotion_data(emo_path)
    attribute_data = load_attribute_data(attr_path_template)

    # Initialize lists to store file names
    train_files, val_files, test_files = [], [], []

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.replace('_aligned', "") in train_names:
            train_files.append(filename)
        elif filename.replace('_aligned', "") in val_names:
            val_files.append(filename)
        elif filename.replace('_aligned', "") in test_names:
            test_files.append(filename)

    if mode == 'gender':
        train_files = [f for f in train_files if RAF_GetLabel(f, emotion_data, attribute_data)["gender"] <= 1]
        val_files = [f for f in val_files if RAF_GetLabel(f, emotion_data, attribute_data)["gender"] <= 1]
        test_files = [f for f in test_files if RAF_GetLabel(f, emotion_data, attribute_data)["gender"] <= 1]

    return train_files, val_files, test_files

# Function to map string values to integer labels
def get_label_from_string(value, dictionary):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None  # Return None if the value doesn't match any key

# Function to aggregate the age column into new categories
def aggregate_age(age):
    if age in ["00-02"]:
        return 0
    elif age in ["03-09", "10-19"]:
        return 1
    elif age in ["20-29", "30-39"]:
        return 2
    elif age in ["40-49", "50-59", "60-69"]:
        return 3
    elif age == "70+":
        return 4
    return None  # Handle unexpected age values


def Aff_GetLabel(item, df):
    # Reference dictionaries
    expression_dict = {0: "Surprise", 1: "Fear", 2: "Disgust", 3: "Happiness", 4: "Sadness", 5: "Anger", 6: "Neutral"}
    gender_dict = {0: "Male", 1: "Female"}
    race_dict = {0: "White", 1: "Black", 2: "Asian", 3: "Indian"}
    # # Updated age dictionary
    # age_dict = {0: "00-02", 1: "03-19", 2: "20-39", 3: "40-69", 4: "70+"}

    if item in df.index:
        row = df.loc[item]
        # Extract values from the row
        expression = row['expression']
        gender = row['gender']
        race = row['race']
        age = row['age']

        # Map the values to their corresponding integer labels
        emo = get_label_from_string(expression, expression_dict)
        gender = get_label_from_string(gender, gender_dict)
        race = get_label_from_string(race, race_dict)
        age = aggregate_age(age)  # Aggregate age using the new age categories

        # Create and return the labels dictionary
        labels = {
            'emotion': emo,
            'gender': gender,
            'race': race,
            'age': age
        }

        return labels
    else:
        return None  # Return None if the item is not found in the CSV


def RAF_GetLabel(item, emotion_data, attribute_data):
    identifier = "_".join(item.split("_")[:2]) + ".jpg"
    identifier_attr = "_".join(item.split("_")[:2]) 

    # Fetch the preloaded emotion and attribute data
    emo = emotion_data.get(identifier, None)
    attr = attribute_data.get(identifier_attr, None)

    if emo is not None and attr is not None:
        # Create a dictionary with the labels
        labels = {
            'emotion': emo,
            'gender': attr['gender'],
            'race': attr['race'],
            'age': attr['age']
        }
        return labels
    else:
        return None  # Handle cases where the identifier is not found


def get_au_labels(item, df, au_columns, gender_dict, race_dict):
    if item in df.index:
        row = df.loc[item]
        # Extract Action Unit labels
        au_labels = [row[au] for au in au_columns]
        au_labels = [1 if x == -1 else x for x in au_labels]
        
        # Extract and map other attributes
        gender = get_label_from_string(row['gender'], gender_dict)
        race = get_label_from_string(row['race'], race_dict)
        age = aggregate_age(row['age'])  # Aggregate age using the new age categories

        # Create and return the labels dictionary
        labels = {
            'emotion': au_labels,
            'gender': gender,
            'race': race,
            'age': age
        }

        return labels
    else:
        return None  # Return None if the item is not found in the CSV

# Example usage for the Emotio dataset
def Emotio_GetLabel(item, df):
    gender_dict = {0: "Male", 1: "Female"}
    race_dict = {0: "White", 1: "Black", 2: "Asian", 3: "Indian"}
    au_columns = ['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU9', 'AU12', 'AU17', 'AU20', 'AU25', 'AU26']
    return get_au_labels(item, df, au_columns, gender_dict, race_dict)

# Example usage for the AU dataset
def AU_GetLabel(item, df):
    gender_dict = {0: "Male", 1: "Female"}
    race_dict = {0: "White", 1: "Black", 2: "Asian", 3: "Indian"}
    au_columns = ['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU9', 'AU10', 'AU12', 'AU16', 'AU17', 'AU25', 'AU26', 'AU27']
    return get_au_labels(item, df, au_columns, gender_dict, race_dict)
