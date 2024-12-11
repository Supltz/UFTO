import sys
import os
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score
import argparse
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import Swin
from expression_datasets import *
from data_prep import RAF_spliting, split_dataset, AU_split
from fairness_criterion import calculate_dp, calculate_eo, calculate_eod

warnings.filterwarnings("ignore")

# Argument parser setup
parser = argparse.ArgumentParser('parameters', add_help=False)
parser.add_argument('--dataset', required=True, type=str, choices=['RAF', 'Aff', 'AU', 'Emotio'], help='Dataset to use')
parser.add_argument('--attribute', required=True, type=str, help='Attribute model to load: race, gender, age')
parser.add_argument('--eva_metric', required=True, type=str, choices=['dp', 'eo', 'eod'], help='Fairness evaluation metric')
args = parser.parse_args()



def freeze_model_layers(model):
    for param in model.parameters():
        param.requires_grad = False

def load_model(model, model_path):
    """Load a model's state dict but exclude the final fully connected (FC) layer."""
    state_dict = torch.load(model_path)
    
    # Filter out the fc layer from the state dict
    state_dict = {k: v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    return model


def test(valloader, model, device, number_classes):
    model.eval()
    all_targets, all_predictions, all_sensitive = [], [], []

    with torch.no_grad():
        for data, target in tqdm(valloader, desc=f'Testing'):
            data, target = data.to(device), target.to(device)
            outputs, _ = model(data)

            # Multi-label (AU/Emotio) or multi-class (RAF/Aff) predictions
            if args.dataset in ["AU", "Emotio"]:
                prediction = (torch.sigmoid(outputs) > 0.5).float()
                all_targets.append(target[:, :number_classes].cpu())
            else:
                _, prediction = torch.max(outputs.data, 1)
                all_targets.append(target[:, 0].cpu())
            all_predictions.append(prediction.cpu())

            # Handle sensitive attribute based on args.attribute
            sensitive_index = number_classes if args.dataset in ["AU", "Emotio"] else 1
            if args.attribute == "gender":
                all_sensitive.append(target[:, sensitive_index].cpu())
            elif args.attribute == "age":
                all_sensitive.append(target[:, sensitive_index + 1].cpu())
            elif args.attribute == "race":
                all_sensitive.append(target[:, sensitive_index + 2].cpu())

    all_targets = torch.cat(all_targets).numpy()
    all_predictions = torch.cat(all_predictions).numpy()
    all_sensitive = torch.cat(all_sensitive).numpy()

    # Calculate F1 score
    f1 = f1_score(all_targets, all_predictions, average='weighted')

    # Fairness Metric Calculation
    if args.eva_metric == 'dp':
        overall_diff = calculate_dp(all_predictions, all_sensitive)
    elif args.eva_metric == 'eo':
        overall_diff = calculate_eo(all_targets, all_predictions, all_sensitive)
    elif args.eva_metric == 'eod':
        overall_diff = calculate_eod(all_targets, all_predictions, all_sensitive)

    # Print F1 and fairness
    print(f'Test, F1 Score: {f1:.4f}, {args.eva_metric}: {overall_diff:.4f}')

    return f1, overall_diff

def main():
    f1_fairness_dict = {}
    directory = "./Trade_Off/checkpoints/" 

    if args.dataset == "RAF" or args.dataset == "Aff":
        num_classes = 7
    elif args.dataset == "AU" or args.dataset == "Emotio":
        num_classes = 13 if args.dataset == "AU" else 11

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    base_Swin = Swin(pretrained=False)
    model = SwinWithPenultimate(base_model=base_Swin, num_classes=num_classes).to(device)



    for lambda_1 in [round(x * 0.01, 1) for x in range(-10, 10)]:
        print(f'Running ({args.dataset}, {args.attribute}) for lambda_1: {lambda_1} and {args.eva_metric}')
        model_file = f"{args.dataset}_{args.attribute}_dcor_{args.eva_metric}_{lambda_1}.pth"
        model_path = os.path.join(directory, model_file)

        load_model(model, model_path)
        freeze_model_layers(model)

        test_f1, test_fairness = test(testloader, model, device, num_classes)

        f1_fairness_dict[lambda_1] = (test_f1, test_fairness)

    with open(f'./Trade_Off/logs/test_{args.dataset}_{args.attribute}_{args.eva_metric}.txt', 'w') as f:
        for lambda_1, metrics in f1_fairness_dict.items():
            f.write(f'Lambda: {lambda_1}, F1: {metrics[0]}, Fairness: {metrics[1]}\n')


if __name__ == "__main__":

    # Dataset-specific logic and data loading
    if args.dataset == "RAF":
        data_path = '/vol/lian/datasets/basic/Image/aligned/'
        csv_path = '/vol/lian/datasets/Annotations/RAF-DB/annotations/'
        _, _, test_list = RAF_spliting(data_path, csv_path, args.attribute)
        testset = RAFDataset(test_list, data_path, mode='test')
    
    elif args.dataset == "Aff":
        data_path = '/vol/lian/datasets/AffectNet/dimitrios/'
        csv_path = '/vol/lian/datasets/Annotations/AffectNet/annotations/EXPR/expr7/'
        _, _, test_list = split_dataset(csv_path)
        testset = AffDataset(test_list, data_path, csv_path, mode='test')
    
    elif args.dataset == "AU":
        data_path = '/vol/lian/datasets/RAF-AU/aligned/'
        csv_path = '/vol/lian/datasets/Annotations/RAF-AU/annotations/'
        _, _, test_list = AU_split(csv_path)
        testset = AUDataset(test_list, data_path, csv_path, mode='test')

    elif args.dataset == "Emotio":
        data_path = '/vol/lian/datasets/EmotioNet/'
        csv_path = '/vol/lian/datasets/Annotations/EmotioNet/annotations/'
        _, _, test_list = split_dataset(csv_path)
        testset = EmotioDataset(test_list, data_path, csv_path, mode='test')

    testloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)

    main()