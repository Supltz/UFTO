import os
import pandas as pd


EMOTIONET_AU_COLUMNS = ["AU1", "AU4", "AU6", "AU12", "AU25"]
RAFAU_AU_COLUMNS = ["AU4", "AU9", "AU10", "AU25", "AU26"]
RACE_LABEL_MAP = {0: "White", 1: "Black", 2: "Asian"}
EXCLUDED_RACE_GROUPS = {"Indian"}


def load_emotion_data(emo_path):
    emo_dict = {}
    with open(emo_path, "r") as handle:
        for line in handle:
            img, label = line.split(" ")
            identifier = "_".join(img.split("_")[:2])
            emo_dict[identifier] = int(label) - 1
    return emo_dict


def load_attribute_data(attr_path_template):
    attr_dict = {}
    attr_dir = os.path.dirname(attr_path_template)
    for file_name in os.listdir(attr_dir):
        if not file_name.endswith(".txt"):
            continue
        parts = file_name.split("_")
        if len(parts) < 2:
            continue
        identifier = f"{parts[0]}_{parts[1]}"
        attr_path = attr_path_template.format(identifier)
        if not os.path.exists(attr_path):
            continue
        with open(attr_path, "r") as handle:
            lines = handle.readlines()[5:]
        gender = int(lines[0].strip())
        race = int(lines[1].strip())
        age = int(lines[2].strip())
        attr_dict[identifier] = {
            "gender": gender,
            "race": race,
            "age": age,
        }
    return attr_dict


def get_names_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df["name"].tolist()


def split_dataset(csv_path):
    train_names = get_names_from_csv(os.path.join(csv_path, "train.csv"))
    val_names = get_names_from_csv(os.path.join(csv_path, "valid.csv"))
    test_names = get_names_from_csv(os.path.join(csv_path, "test.csv"))
    return train_names, val_names, test_names


def AU_split(csv_path):
    train_names = get_names_from_csv(os.path.join(csv_path, "train.csv"))
    val_names = get_names_from_csv(os.path.join(csv_path, "valid.csv"))
    test_names = get_names_from_csv(os.path.join(csv_path, "test.csv"))
    return train_names, val_names, test_names


def get_label_from_string(value, dictionary):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None


def aggregate_age(age):
    if pd.isna(age):
        return None
    age = str(age).strip()
    if age in ["00-02", "00-03"]:
        return 0
    if age in ["03-09", "04-19", "10-19"]:
        return 1
    if age in ["20-29", "30-39", "20-39"]:
        return 2
    if age in ["40-49", "50-59", "60-69", "40-69"]:
        return 3
    if age == "70+":
        return 4
    return None


def Aff_GetLabel(item, df):
    expression_dict = {
        0: "Surprise",
        1: "Fear",
        2: "Disgust",
        3: "Happiness",
        4: "Sadness",
        5: "Anger",
        6: "Neutral",
    }
    gender_dict = {0: "Male", 1: "Female"}
    race_dict = RACE_LABEL_MAP

    if item not in df.index:
        return None

    row = df.loc[item]
    labels = {
        "emotion": get_label_from_string(row["expression"], expression_dict),
        "gender": get_label_from_string(row["gender"], gender_dict),
        "race": get_label_from_string(row["race"], race_dict),
        "age": aggregate_age(row["age"]),
    }
    return labels


def RAF_GetLabel(item, df):
    return Aff_GetLabel(item, df)


def get_au_labels(item, df, au_columns, gender_dict, race_dict):
    if item not in df.index:
        return None

    row = df.loc[item]
    au_labels = [row[au] for au in au_columns]

    return {
        "emotion": au_labels,
        "gender": get_label_from_string(row["gender"], gender_dict),
        "race": get_label_from_string(row["race"], race_dict),
        "age": aggregate_age(row["age"]),
    }


def Emotio_GetLabel(item, df):
    gender_dict = {0: "Male", 1: "Female"}
    race_dict = RACE_LABEL_MAP
    return get_au_labels(item, df, EMOTIONET_AU_COLUMNS, gender_dict, race_dict)


def AU_GetLabel(item, df):
    gender_dict = {0: "Male", 1: "Female"}
    race_dict = RACE_LABEL_MAP
    return get_au_labels(item, df, RAFAU_AU_COLUMNS, gender_dict, race_dict)
