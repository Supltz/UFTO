import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
import yaml
from sklearn.metrics import f1_score
from tqdm import tqdm

from expression_datasets import (
    build_dataset_splits,
    extract_sensitive_targets,
    extract_task_targets,
    is_multilabel_task,
)
from metrics import (
    compute_dpv_eov,
)
from models import ConditionalSwinBase


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


def build_test_loader(dataset_name, config):
    _, _, test_dataset = build_dataset_splits(dataset_name, config)
    return DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=int(config.get("num_workers", 4)),
    )


def predict_from_logits(logits, dataset_name):
    if is_multilabel_task(dataset_name):
        return (torch.sigmoid(logits) >= 0.5).float()
    return logits.argmax(dim=1)


def compute_micro_f1(preds, targets, dataset_name):
    if is_multilabel_task(dataset_name):
        return float(
            f1_score(
                targets.int().cpu().numpy(),
                preds.int().cpu().numpy(),
                average="micro",
                zero_division=0,
            )
        )
    return float(
        f1_score(
            targets.cpu().numpy(),
            preds.cpu().numpy(),
            average="micro",
            zero_division=0,
        )
    )


def evaluate(model, loader, dataset_name, lambda_val, sensitive_attribute, device):
    model.eval()
    all_targets = []
    all_preds = []
    sensitive_values = []

    with torch.no_grad():
        for images, labels in tqdm(
            loader,
            desc=f"Eval {dataset_name}-{sensitive_attribute} lambda={lambda_val:.2f}",
            leave=False,
        ):
            images = images.to(device)
            labels = labels.to(device)
            task_targets = extract_task_targets(labels, dataset_name)
            sensitive_targets = extract_sensitive_targets(labels, dataset_name)
            logits = model(images, lambda_val)
            preds = predict_from_logits(logits, dataset_name)

            all_targets.append(task_targets.cpu())
            all_preds.append(preds.cpu())
            sensitive_values.append(sensitive_targets[sensitive_attribute].cpu())

    targets = torch.cat(all_targets, dim=0).numpy()
    preds = torch.cat(all_preds, dim=0).numpy()
    sensitive = torch.cat(sensitive_values, dim=0).numpy()
    dpv, eov = compute_dpv_eov(targets, preds, sensitive)

    return {
        "Micro F1": compute_micro_f1(
            torch.as_tensor(preds),
            torch.as_tensor(targets),
            dataset_name,
        ),
        "dpv": dpv,
        "eov": eov,
        "sensitive_attribute": sensitive_attribute,
    }


def load_checkpoint(model, checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device)
    if any(key.startswith("module.") for key in state.keys()):
        state = {key.replace("module.", "", 1): value for key, value in state.items()}
    model.load_state_dict(state)


def _normalize(values):
    min_value = min(values)
    max_value = max(values)
    span = max_value - min_value
    if span == 0:
        return [0.0 for _ in values]
    return [(value - min_value) / span for value in values]


def select_by_average_utopia_distance(all_results):
    if not all_results:
        return None

    dpv_norm = _normalize([row["dpv"] for row in all_results])
    eov_norm = _normalize([row["eov"] for row in all_results])
    f1_norm = _normalize([row["Micro F1"] for row in all_results])

    best_idx = None
    best_score = None
    for idx, row in enumerate(all_results):
        dpv_distance = (dpv_norm[idx] ** 2 + (1.0 - f1_norm[idx]) ** 2) ** 0.5
        eov_distance = (eov_norm[idx] ** 2 + (1.0 - f1_norm[idx]) ** 2) ** 0.5
        avg_distance = 0.5 * (dpv_distance + eov_distance)

        if best_idx is None or avg_distance < best_score:
            best_idx = idx
            best_score = avg_distance

    selected = dict(all_results[best_idx])
    selected["avg_utopia_distance"] = best_score
    return selected


def strip_f1_from_result(result):
    if result is None:
        return None
    return {
        key: value
        for key, value in result.items()
        if key != "Micro F1"
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["rafdb", "affectnet", "emotionet", "rafau"],
    )
    args = parser.parse_args()

    with open("config.yaml", "r") as handle:
        config = yaml.safe_load(handle)[args.dataset]
    device, device_ids = pick_cuda_device()
    if device.type == "cuda":
        print(f"Using device {device} for testing (available GPUs: {len(device_ids)})")
    else:
        print("Using CPU for testing")

    loader = build_test_loader(args.dataset, config)
    lambda_grid = [i / 100.0 for i in range(1, 100)]
    results_by_attribute = {}
    best_micro_f1 = None
    print(f"Starting test sweep for dataset: {args.dataset}")
    print(f"Evaluating sensitive attributes: {', '.join(SENSITIVE_ATTRIBUTES)}")

    for sensitive_attribute in SENSITIVE_ATTRIBUTES:
        print(f"\nLoading checkpoint for sensitive attribute: {sensitive_attribute}")
        checkpoint_path = f"{args.dataset}_{sensitive_attribute}_best.pt"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model = ConditionalSwinBase(config["num_classes"]).to(device)
        load_checkpoint(model, checkpoint_path, device)
        if device.type == "cuda" and len(device_ids) > 1:
            model = torch.nn.DataParallel(
                model,
                device_ids=device_ids,
                output_device=device_ids[0],
            )
        print(f"Loaded checkpoint: {checkpoint_path}")
        print(f"Running lambda sweep for {args.dataset}-{sensitive_attribute}")

        all_results = []
        for lambda_val in tqdm(
            lambda_grid,
            desc=f"Lambda sweep {args.dataset}-{sensitive_attribute}",
        ):
            metrics = evaluate(
                model,
                loader,
                args.dataset,
                lambda_val,
                sensitive_attribute,
                device,
            )
            all_results.append({"lambda": lambda_val, **metrics})

        selected_result = select_by_average_utopia_distance(all_results)
        if selected_result is not None:
            selected_micro_f1 = selected_result["Micro F1"]
            if best_micro_f1 is None or selected_micro_f1 > best_micro_f1:
                best_micro_f1 = selected_micro_f1

        results_by_attribute[sensitive_attribute] = {
            "checkpoint": checkpoint_path,
            "selected_result": strip_f1_from_result(selected_result),
        }
        print(f"Finished evaluation for {sensitive_attribute}")

    output_path = f"{args.dataset}_test_results.json"
    with open(output_path, "w") as handle:
        json.dump(
            {
                "dataset": args.dataset,
                "Micro F1": best_micro_f1,
                "results_by_attribute": results_by_attribute,
            },
            handle,
            indent=2,
        )
    print(f"\nSaved test results to: {output_path}")


if __name__ == "__main__":
    main()
