import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import yaml

from expression_datasets import (
    build_dataset_splits,
    extract_sensitive_targets,
    infer_sensitive_num_classes,
)
from models import build_swin_base


SENSITIVE_ATTRIBUTES = ("gender", "age", "race")


def pick_cuda_device():
    if not torch.cuda.is_available():
        return torch.device("cpu"), []
    device_count = torch.cuda.device_count()
    free_mem = []
    for idx in range(device_count):
        free_bytes, _ = torch.cuda.mem_get_info(idx)
        free_mem.append((free_bytes, idx))
    free_mem.sort(reverse=True)
    best_idx = free_mem[0][1]
    device_ids = [best_idx] + [idx for idx in range(device_count) if idx != best_idx]
    return torch.device(f"cuda:{best_idx}"), device_ids


def save_checkpoint(model, path):
    module = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(module.state_dict(), path)


def train_sensitive_encoder(
    model,
    loader,
    optimizer,
    config,
    device,
    sensitive_name,
    run_name,
    checkpoint_path,
    dataset_name,
):
    criterion = nn.CrossEntropyLoss()
    best_loss = float("inf")
    patience = int(config.get("patience", 5))
    min_delta = float(config.get("min_delta", 0.0))
    patience_counter = 0

    wandb.init(project="fairness-benchmarks", name=run_name, config=config)

    for epoch in range(1, config["epochs"] + 1):
        start_time = time.time()
        model.train()
        total_loss = 0.0
        total_samples = 0

        for images, labels in tqdm(loader, desc=f"Train {sensitive_name}", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            sensitive_targets = extract_sensitive_targets(labels, dataset_name)[sensitive_name]

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, sensitive_targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * sensitive_targets.size(0)
            total_samples += sensitive_targets.size(0)

        train_loss = total_loss / total_samples if total_samples else 0.0
        epoch_time = time.time() - start_time
        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "epoch_time": epoch_time,
            }
        )
        print(f"[{run_name} | Epoch {epoch}] Train Loss: {train_loss:.4f}")

        if train_loss < best_loss - min_delta:
            best_loss = train_loss
            patience_counter = 0
            save_checkpoint(model, checkpoint_path)
            wandb.save(checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    wandb.finish()


def main(dataset_name):
    with open("config.yaml", "r") as handle:
        base_config = yaml.safe_load(handle)[dataset_name]
    stage1_config = {**base_config, **base_config.get("stage1", {})}
    for key in ("learning_rate", "weight_decay", "momentum"):
        if key in stage1_config:
            stage1_config[key] = float(stage1_config[key])

    device, device_ids = pick_cuda_device()
    if device.type == "cuda":
        print(f"Using device {device} (available GPUs: {len(device_ids)})")
    else:
        print("Using CPU")

    train_dataset, _, _ = build_dataset_splits(dataset_name, stage1_config)
    loader = DataLoader(
        train_dataset,
        batch_size=stage1_config["batch_size"],
        shuffle=True,
        num_workers=int(stage1_config.get("num_workers", 4)),
    )
    sensitive_num_classes = infer_sensitive_num_classes(train_dataset, dataset_name)

    for sensitive_name in SENSITIVE_ATTRIBUTES:
        model = build_swin_base(sensitive_num_classes[sensitive_name]).to(device)
        if device.type == "cuda" and len(device_ids) > 1:
            model = nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=stage1_config["learning_rate"],
            momentum=stage1_config.get("momentum", 0.9),
            weight_decay=stage1_config["weight_decay"],
        )

        train_sensitive_encoder(
            model,
            loader,
            optimizer,
            stage1_config,
            device,
            sensitive_name,
            f"{dataset_name}-{sensitive_name}-encoder",
            f"{dataset_name}_{sensitive_name}_encoder.pt",
            dataset_name,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["rafdb", "affectnet", "emotionet", "rafau"],
    )
    args = parser.parse_args()
    main(args.dataset)
