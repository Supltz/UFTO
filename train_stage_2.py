import argparse
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import yaml

from data_prep import RAFAU_AU_COLUMNS
from expression_datasets import (
    DATASET_METADATA,
    build_dataset_splits,
    extract_sensitive_targets,
    extract_task_targets,
    infer_sensitive_num_classes,
    is_multilabel_task,
)
from models import ConditionalSwinBase, build_swin_base
from surrogates import conditional_distance_correlation_joint


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


def get_classifier_module(model):
    module = model.module if isinstance(model, nn.DataParallel) else model
    if hasattr(module, "base"):
        module = module.base
    if hasattr(module, "head"):
        return module.head
    raise ValueError("No classifier head found for feature capture.")


def register_feature_hook(model):
    feature_store = {}
    classifier = get_classifier_module(model)

    def hook(_, inputs, __):
        if isinstance(model, nn.DataParallel):
            feature_store[str(inputs[0].device)] = inputs[0]
        else:
            feature_store["value"] = inputs[0]

    handle = classifier.register_forward_hook(hook)
    return feature_store, handle


def gather_features(feature_store, device, device_ids):
    if "value" in feature_store:
        return feature_store["value"]
    features = []
    for dev_id in device_ids:
        key = f"cuda:{dev_id}" if device.type == "cuda" else "cpu"
        feat = feature_store.get(key)
        if feat is not None:
            features.append(feat.to(device))
    if not features:
        return None
    return torch.cat(features, dim=0)


def sample_lambda(config, device):
    lam_min = float(config.get("lambda_min", 0.0))
    lam_max = float(config.get("lambda_max", 1.0))
    lam_step = float(config.get("lambda_step", 0.01))
    if lam_step <= 0:
        return torch.tensor(lam_min, device=device)
    steps = int(round((lam_max - lam_min) / lam_step))
    idx = torch.randint(0, max(steps, 0) + 1, (1,), device=device)
    return (lam_min + idx * lam_step).clamp(lam_min, lam_max)


def load_checkpoint(model, checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device)
    if any(key.startswith("module.") for key in state.keys()):
        state = {key.replace("module.", "", 1): value for key, value in state.items()}
    model.load_state_dict(state)


def build_task_criterion(dataset_name, train_dataset=None):
    if not is_multilabel_task(dataset_name):
        return nn.CrossEntropyLoss()

    if dataset_name == "rafau" and getattr(train_dataset, "df", None) is not None:
        targets = torch.as_tensor(
            train_dataset.df.loc[train_dataset.data, RAFAU_AU_COLUMNS].to_numpy(),
            dtype=torch.float32,
        )
        positive_counts = targets.sum(dim=0)
        total_samples = float(targets.size(0))
        negative_counts = total_samples - positive_counts
        pos_weight = negative_counts / positive_counts.clamp_min(1.0)
        print(f"Using weighted BCE loss for {dataset_name} with pos_weight={pos_weight.tolist()}")
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    return nn.BCEWithLogitsLoss()


def train_and_validate(
    model,
    loaders,
    criterion,
    optimizer,
    scheduler,
    config,
    dataset_name,
    sensitive_name,
    device,
    device_ids,
    sensitive_encoder,
):
    best_val_loss = float("inf")
    patience_counter = 0
    min_delta = float(config.get("min_delta", 0.0))

    feature_store, hook_handle = register_feature_hook(model)
    sensitive_store, sensitive_handle = register_feature_hook(sensitive_encoder)
    sensitive_encoder.eval()
    for param in sensitive_encoder.parameters():
        param.requires_grad = False

    try:
        for epoch in range(1, config["epochs"] + 1):
            start_time = time.time()
            print(
                f"Starting epoch {epoch}/{config['epochs']} "
                f"for {dataset_name}-{sensitive_name}..."
            )

            model.train()
            train_loss = 0.0
            train_dcor_sensitive = 0.0
            train_total = 0

            for images, labels in tqdm(loaders["train"], desc="Train", leave=False):
                images = images.to(device)
                labels = labels.to(device)
                task_targets = extract_task_targets(labels, dataset_name)
                sensitive_targets = extract_sensitive_targets(labels, dataset_name)

                optimizer.zero_grad()
                feature_store.clear()
                lambda_val = sample_lambda(config, device)
                lambda_scalar = float(lambda_val.item())
                logits = model(images, lambda_scalar)
                task_loss = criterion(logits, task_targets)

                features = gather_features(feature_store, device, device_ids)
                if features is None:
                    raise ValueError("Failed to capture penultimate features.")

                feat_norm = nn.functional.normalize(features, p=2, dim=1)
                if lambda_scalar > 0:
                    sensitive_store.clear()
                    with torch.no_grad():
                        sensitive_encoder(images)
                    sensitive_features = gather_features(
                        sensitive_store,
                        device,
                        device_ids,
                    )
                    if sensitive_features is None:
                        raise ValueError(
                            f"Failed to capture {sensitive_name} features."
                        )
                    sensitive_norm = nn.functional.normalize(
                        sensitive_features.detach(),
                        p=2,
                        dim=1,
                    )
                    dcor_sensitive = conditional_distance_correlation_joint(
                        feat_norm,
                        sensitive_norm,
                        task_targets,
                        sensitive_targets[sensitive_name],
                    )
                else:
                    dcor_sensitive = torch.tensor(0.0, device=device)

                loss = (1.0 - lambda_val) * task_loss + lambda_val * dcor_sensitive
                loss.backward()
                optimizer.step()

                batch_size = images.size(0)
                train_loss += loss.item() * batch_size
                train_dcor_sensitive += dcor_sensitive.item() * batch_size
                train_total += batch_size

            train_loss /= train_total
            train_dcor_sensitive /= train_total

            model.eval()
            val_loss = 0.0
            eval_lambda = float(config.get("eval_lambda", 0.5))
            with torch.no_grad():
                for images, labels in tqdm(loaders["val"], desc="Val", leave=False):
                    images = images.to(device)
                    labels = labels.to(device)
                    task_targets = extract_task_targets(labels, dataset_name)
                    logits = model(images, eval_lambda)
                    loss = criterion(logits, task_targets)
                    val_loss += loss.item() * images.size(0)

            val_loss /= len(loaders["val"].dataset)
            epoch_time = time.time() - start_time

            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/dcor_sensitive": train_dcor_sensitive,
                    "val/loss": val_loss,
                    "epoch_time": epoch_time,
                }
            )

            print(
                f"[{dataset_name.upper()} | {sensitive_name} | Epoch {epoch}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Dcor Sensitive: {train_dcor_sensitive:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                checkpoint_path = f"{dataset_name}_{sensitive_name}_best.pt"
                torch.save(model.state_dict(), checkpoint_path)
                wandb.save(checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= int(config["patience"]):
                    print("Early stopping triggered.")
                    break

            if scheduler is not None:
                scheduler.step(val_loss)
    finally:
        hook_handle.remove()
        sensitive_handle.remove()


def main(dataset_name):
    with open("config.yaml", "r") as handle:
        config = yaml.safe_load(handle)[dataset_name]

    seed = config.get("seed")
    if seed is not None:
        seed = int(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    for key in ("learning_rate", "weight_decay", "momentum"):
        if key in config:
            config[key] = float(config[key])
    for key in ("min_learning_rate", "lr_decay_factor"):
        if key in config:
            config[key] = float(config[key])
    if "lr_scheduler_patience" in config:
        config["lr_scheduler_patience"] = int(config["lr_scheduler_patience"])

    device, device_ids = pick_cuda_device()
    if device.type == "cuda":
        print(f"Using device {device} (available GPUs: {len(device_ids)})")
    else:
        print("Using CPU")

    train_dataset, val_dataset, _ = build_dataset_splits(dataset_name, config)
    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=int(config.get("num_workers", 4)),
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=int(config.get("num_workers", 4)),
        ),
    }

    num_outputs = DATASET_METADATA[dataset_name]["num_outputs"]
    sensitive_num_classes = infer_sensitive_num_classes(train_dataset, dataset_name)
    # BCEWithLogitsLoss stores pos_weight as a buffer, so it must live on the
    # same device as logits/targets for multilabel training.
    criterion = build_task_criterion(dataset_name, train_dataset).to(device)

    for sensitive_name in SENSITIVE_ATTRIBUTES:
        wandb.init(
            project="fairness-benchmarks",
            name=f"{dataset_name}-{sensitive_name}-training",
            config={**config, "sensitive_attribute": sensitive_name},
        )

        model = ConditionalSwinBase(num_outputs).to(device)
        if device.type == "cuda" and len(device_ids) > 1:
            model = nn.DataParallel(
                model,
                device_ids=device_ids,
                output_device=device_ids[0],
            )

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config["learning_rate"],
            momentum=config.get("momentum", 0.9),
            weight_decay=config["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.get("lr_decay_factor", 0.5),
            patience=config.get("lr_scheduler_patience", 3),
            min_lr=config.get("min_learning_rate", 1e-4),
        )

        encoder = build_swin_base(sensitive_num_classes[sensitive_name]).to(device)
        checkpoint_path = config.get(
            f"{sensitive_name}_encoder_ckpt",
            f"{dataset_name}_{sensitive_name}_encoder.pt",
        )
        load_checkpoint(encoder, checkpoint_path, device)
        if device.type == "cuda" and len(device_ids) > 1:
            encoder = nn.DataParallel(
                encoder,
                device_ids=device_ids,
                output_device=device_ids[0],
            )

        train_and_validate(
            model,
            loaders,
            criterion,
            optimizer,
            scheduler,
            config,
            dataset_name,
            sensitive_name,
            device,
            device_ids,
            encoder,
        )
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["rafdb", "affectnet", "emotionet", "rafau"],
    )
    args = parser.parse_args()
    main(args.dataset)
