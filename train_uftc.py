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
from dependence_metric import hsic, MINE, mine_loss, distance_correlation  # Dependence metric import
from fairness_criterion import calculate_dp, calculate_eo, calculate_eod

warnings.filterwarnings("ignore")

# Argument parser setup
parser = argparse.ArgumentParser('parameters', add_help=False)
parser.add_argument('--dataset', required=True, type=str, choices=['RAF', 'Aff', 'AU', 'Emotio'], help='Dataset to use')
parser.add_argument('--attribute', required=True, type=str, help='Attribute model to load: race, gender, age')
parser.add_argument('--metric', required=True, type=str, choices=['hsic', 'mine', 'dcor'], help='Dependence metric')
parser.add_argument('--eva_metric', required=True, type=str, choices=['dp', 'eo', 'eod'], help='Fairness evaluation metric')
args = parser.parse_args()



# Load pre-trained attribute model (Swin for race, gender, age)
def load_attribute_model(dataset, attribute, device):
    model_path = f'./Feature_Entangle/checkpoints/best_model_{dataset}_{attribute}.pth'  # Dataset and attribute-specific path
    model = Swin(pretrained=False)
    model.fc = nn.Identity()  # Remove final classification layer
    model.load_state_dict(torch.load(model_path), strict=False)
    model = model.to(device)
    
    # Freeze the attribute model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    model.eval()
    return model

def calculate_loss(outputs, target, penultimate_layer, attr_penultimate_layer, criterion, lambda_1, mine_model=None, mine_optimizer=None, number_classes=None, condition=None):
    """
    Calculates classification and dependence loss.
    If 'condition' is provided, applies the condition (used for EO).
    Updates MINE model if it's used for dependence metric.
    """
    # Multi-label loss for AU/Emotio
    if number_classes > 7:
        classification_loss = criterion(outputs, target[:, :number_classes].float())
    else:
        classification_loss = criterion(outputs, target[:, 0])  # Emotion classification loss

    if condition is not None:  # Apply condition for EO
        penultimate_layer = penultimate_layer[condition]
        attr_penultimate_layer = attr_penultimate_layer[condition]
        if penultimate_layer.shape[0] == 0 or attr_penultimate_layer.shape[0] == 0 or penultimate_layer.shape[0] == 1 or attr_penultimate_layer.shape[0] == 1:
            dep_loss = torch.tensor(0.0, device=outputs.device)  # Return a tensor 0.0 on the same device
            return classification_loss, dep_loss, classification_loss  # Return only classification loss

    # Dependence loss based on metric
    if args.metric == 'hsic':
        dep_loss = hsic(penultimate_layer.detach().cpu().numpy(), attr_penultimate_layer.detach().cpu().numpy())
    elif args.metric == 'mine':
        dep_loss = 0.0
        # Update the MINE model multiple times per batch
        for _ in range(4):  # Update MINE 4 times for each batch
            # Zero the gradients for MINE optimizer
            mine_optimizer.zero_grad()
            # Calculate MINE loss and backward pass
            current_dep_loss = mine_loss(mine_model, penultimate_layer, attr_penultimate_layer)
            current_dep_loss.backward(retain_graph=True)
            mine_optimizer.step()  # Update MINE model

            # Accumulate the dependence loss
            dep_loss += current_dep_loss

        # Average the dependence loss over the number of updates
        dep_loss /= 4
    elif args.metric == 'dcor':
        dep_loss = distance_correlation(penultimate_layer.detach().cpu().numpy(), attr_penultimate_layer.detach().cpu().numpy())

    # Final loss combines classification loss and dependence loss with lambda weighting
    total_loss = (1 - abs(lambda_1)) * classification_loss + lambda_1 * dep_loss
    # # Print losses
    # print(f'Loss: {total_loss:.4f}, Classification Loss: {classification_loss:.4f}, '
    #       f'Dependence Loss: {dep_loss:.4f}')
    return classification_loss, dep_loss, total_loss

def train(epoch, trainloader, model, attr_model, optimizer, criterion, lambda_1, device, number_classes):
    model.train()
    running_loss, running_classification_loss, running_dep_loss = 0.0, 0.0, 0.0
    all_targets, all_predictions, all_sensitive = [], [], []
    
    # Initialize MINE model and optimizer if using MINE
    mine_model, mine_optimizer = None, None
    if args.metric == 'mine':
        mine_model = MINE(512,512).to(device)
        mine_optimizer = optim.Adam(mine_model.parameters(), lr=1e-4)

    for data, target in tqdm(trainloader, desc=f'Training Epoch {epoch+1}'):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Forward pass through the model
        outputs, penultimate_layer = model(data)
        with torch.no_grad():
            attr_penultimate_layer = attr_model(data)

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

        # Loss calculation based on evaluation metric
        if args.eva_metric == 'dp':
            classification_loss, dep_loss, loss = calculate_loss(outputs, target, penultimate_layer, attr_penultimate_layer, criterion, lambda_1, mine_model=mine_model, mine_optimizer=mine_optimizer, number_classes=number_classes)
        elif args.eva_metric == 'eo':
            correct_mask = torch.any(prediction == target[:, :number_classes], dim=1) if args.dataset in ["AU", "Emotio"] else (prediction == target[:, 0])
            classification_loss, dep_loss, loss = calculate_loss(outputs, target, penultimate_layer, attr_penultimate_layer, criterion, lambda_1, condition=correct_mask, mine_model=mine_model, mine_optimizer=mine_optimizer, number_classes=number_classes)
        elif args.eva_metric == 'eod':
            classification_loss, dep_loss, loss = calculate_loss(outputs, target, penultimate_layer, attr_penultimate_layer, criterion, lambda_1, mine_model=mine_model, mine_optimizer=mine_optimizer, number_classes=number_classes)

        # Backpropagate and update the model
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_classification_loss += classification_loss.item()
        running_dep_loss += dep_loss.item()

    avg_loss = running_loss / len(trainloader)
    avg_classification_loss = running_classification_loss / len(trainloader)
    avg_dep_loss = running_dep_loss / len(trainloader)

    # Combine and process targets, predictions, and sensitive attributes
    all_targets = torch.cat(all_targets).numpy()
    all_predictions = torch.cat(all_predictions).numpy()
    all_sensitive = torch.cat(all_sensitive).numpy()

    # Calculate F1 score
    f1 = f1_score(all_targets, all_predictions, average='weighted')

    # Fairness Metric Calculation for the chosen attribute (gender, age, or race)
    if args.eva_metric == 'dp':
        overall_diff = calculate_dp(all_predictions, all_sensitive)
    elif args.eva_metric == 'eo':
        overall_diff = calculate_eo(all_targets, all_predictions, all_sensitive)
    elif args.eva_metric == 'eod':
        overall_diff = calculate_eod(all_targets, all_predictions, all_sensitive)

    # Print losses, F1, and fairness metric
    print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Classification Loss: {avg_classification_loss:.4f}, '
          f'Dependence Loss: {avg_dep_loss:.4f}, F1 Score: {f1:.4f}, Overall {args.eva_metric} Difference: {overall_diff:.4f}')

    # Return the average losses, F1 score, and fairness metric (overall_diff)
    return avg_loss, avg_classification_loss, avg_dep_loss, f1, overall_diff

def val(epoch, valloader, model, device, number_classes):
    model.eval()
    all_targets, all_predictions, all_sensitive = [], [], []

    with torch.no_grad():
        for data, target in tqdm(valloader, desc=f'Validating Epoch {epoch+1}'):
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
    print(f'Validation - Epoch {epoch + 1}, F1 Score: {f1:.4f}, {args.eva_metric}: {overall_diff:.4f}')

    return f1, overall_diff

def main():
    best_val_f1, best_val_fairness, epochs_no_improve, early_stop_epochs = 0, 0, 0, 5
    f1_fairness_dict = {}
    best_model_state = None

    if args.dataset == "RAF" or args.dataset == "Aff":
        num_classes = 7
        criterion = nn.CrossEntropyLoss()
    elif args.dataset == "AU" or args.dataset == "Emotio":
        num_classes = 13 if args.dataset == "AU" else 11
        criterion = nn.BCEWithLogitsLoss()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    attr_model = load_attribute_model(args.dataset, args.attribute, device)

    for lambda_1 in [round(x * 0.01, 1) for x in range(-10, 10)]:
        print(f'Running ({args.dataset}, {args.attribute}) for lambda_1: {lambda_1} and {args.metric}')
        writer = SummaryWriter(f'./Trade_Off/runs/{args.dataset}_{args.attribute}_{args.metric}_{args.eva_metric}_{lambda_1}')
        base_Swin = Swin(pretrained=False)
        model = SwinWithPenultimate(base_model=base_Swin, num_classes=num_classes).to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        best_model_state = None
        last_train_f1, last_train_fairness = None, None

        for epoch in range(num_epochs):
            train_loss, train_classification_loss, train_dep_loss, train_f1, train_fairness = train(epoch, trainloader, model, attr_model, optimizer, criterion, lambda_1, device, num_classes)
            val_f1, val_fairness = val(epoch, valloader, model, device, num_classes)

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Classification Loss/train', train_classification_loss, epoch)
            writer.add_scalar('Dependence Loss/train', train_dep_loss, epoch)
            writer.add_scalar('F1/train', train_f1, epoch)
            writer.add_scalar(f'Fairness/train_{args.eva_metric}', train_fairness, epoch)
            writer.add_scalar('TradeOff/train', train_f1 - train_fairness, epoch)

            writer.add_scalar('F1/val', val_f1, epoch)
            writer.add_scalar(f'Fairness/val_{args.eva_metric}', val_fairness, epoch)
            writer.add_scalar('TradeOff/val', val_f1 - val_fairness, epoch)

            if scheduler.get_last_lr()[0] > 0.0001:
                scheduler.step()

            trade_off_val = val_f1 - val_fairness

            if trade_off_val > best_val_f1 - best_val_fairness:
                best_val_f1, best_val_fairness = val_f1, val_fairness
                epochs_no_improve = 0
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1

            last_train_f1, last_train_fairness = train_f1, train_fairness

            if epochs_no_improve == early_stop_epochs:
                print("Early stopping triggered")
                break

        if best_model_state is not None:
            torch.save(best_model_state, f'./Trade_Off/checkpoints/{args.dataset}_{args.attribute}_{args.metric}_{args.eva_metric}_{lambda_1}.pth')
        else:
            print("Saving last model due to no improvement after max epochs.")
            torch.save(model.state_dict(), f'./Trade_Off/checkpoints/{args.dataset}_{args.attribute}_{args.metric}_{args.eva_metric}_{lambda_1}.pth')

        f1_fairness_dict[lambda_1] = (last_train_f1, last_train_fairness)
        writer.close()

    with open(f'./Trade_Off/logs/{args.dataset}_{args.attribute}_{args.metric}_{args.eva_metric}.txt', 'w') as f:
        for lambda_1, metrics in f1_fairness_dict.items():
            f.write(f'Lambda: {lambda_1}, F1: {metrics[0]}, Fairness: {metrics[1]}\n')


if __name__ == "__main__":
    num_epochs = 100
    lr = 0.01
    momentum = 0.9
    weight_decay = 5e-4

    
    # Dataset-specific logic and data loading
    if args.dataset == "RAF":
        batchsize = 32
        data_path = '/vol/lian/datasets/basic/Image/aligned/'
        csv_path = '/vol/lian/datasets/Annotations/RAF-DB/annotations/'
        train_list, val_list, _ = RAF_spliting(data_path, csv_path, args.attribute)
        trainset = RAFDataset(train_list, data_path, mode='train')
        valset = RAFDataset(val_list, data_path, mode='val')
    
    elif args.dataset == "Aff":
        batchsize = 128
        data_path = '/vol/lian/datasets/AffectNet/dimitrios/'
        csv_path = '/vol/lian/datasets/Annotations/AffectNet/annotations/EXPR/expr7/'
        train_list, val_list, _ = split_dataset(csv_path)
        trainset = AffDataset(train_list, data_path, csv_path, mode='train')
        valset = AffDataset(val_list, data_path, csv_path, mode='valid')
    
    elif args.dataset == "AU":
        batchsize = 16
        data_path = '/vol/lian/datasets/RAF-AU/aligned/'
        csv_path = '/vol/lian/datasets/Annotations/RAF-AU/annotations/'
        train_list, val_list, _ = AU_split(csv_path)
        trainset = AUDataset(train_list, data_path, csv_path, mode='train')
        valset = AUDataset(val_list, data_path, csv_path, mode='valid')

    elif args.dataset == "Emotio":
        batchsize = 64
        data_path = '/vol/lian/datasets/EmotioNet/'
        csv_path = '/vol/lian/datasets/Annotations/EmotioNet/annotations/'
        train_list, val_list, _ = split_dataset(csv_path)
        trainset = EmotioDataset(train_list, data_path, csv_path, mode='train')
        valset = EmotioDataset(val_list, data_path, csv_path, mode='valid')

    trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2, drop_last=True)
    valloader = DataLoader(valset, batch_size=1, shuffle=True, num_workers=2)

    main()