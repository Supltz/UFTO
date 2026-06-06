import itertools

import numpy as np


def _group_pairs(groups):
    return list(itertools.combinations(groups, 2))


def _safe_mean(values):
    return float(np.mean(values)) if values else 0.0


def compute_dpv_eov(targets, preds, sensitive):
    targets = np.asarray(targets)
    preds = np.asarray(preds)
    sensitive = np.asarray(sensitive)

    groups = sorted(set(int(value) for value in sensitive.tolist()))
    pairs = _group_pairs(groups)
    if not pairs:
        return 0.0, 0.0

    if targets.ndim == 1:
        classes = sorted(
            set(int(value) for value in np.concatenate([targets, preds]).tolist())
        )
        dpv_terms = []
        eov_terms = []

        for cls in classes:
            group_positive_rates = {}
            group_tpr_rates = {}
            for group in groups:
                mask_group = sensitive == group
                group_count = mask_group.sum()
                if group_count > 0:
                    group_positive_rates[group] = float((preds[mask_group] == cls).mean())
                else:
                    group_positive_rates[group] = 0.0

                positive_mask = mask_group & (targets == cls)
                positive_count = positive_mask.sum()
                if positive_count > 0:
                    group_tpr_rates[group] = float((preds[positive_mask] == cls).mean())
                else:
                    group_tpr_rates[group] = 0.0

            for group_a, group_b in pairs:
                dpv_terms.append(
                    abs(group_positive_rates[group_a] - group_positive_rates[group_b])
                )
                eov_terms.append(
                    abs(group_tpr_rates[group_a] - group_tpr_rates[group_b])
                )

        return _safe_mean(dpv_terms), _safe_mean(eov_terms)

    dpv_terms = []
    eov_terms = []
    num_labels = targets.shape[1]
    for label_idx in range(num_labels):
        group_positive_rates = {}
        group_tpr_rates = {}
        for group in groups:
            mask_group = sensitive == group
            group_count = mask_group.sum()
            if group_count > 0:
                group_positive_rates[group] = float(preds[mask_group, label_idx].mean())
            else:
                group_positive_rates[group] = 0.0

            positive_mask = mask_group & (targets[:, label_idx] == 1)
            positive_count = positive_mask.sum()
            if positive_count > 0:
                group_tpr_rates[group] = float(preds[positive_mask, label_idx].mean())
            else:
                group_tpr_rates[group] = 0.0

        for group_a, group_b in pairs:
            dpv_terms.append(
                abs(group_positive_rates[group_a] - group_positive_rates[group_b])
            )
            eov_terms.append(
                abs(group_tpr_rates[group_a] - group_tpr_rates[group_b])
            )

    return _safe_mean(dpv_terms), _safe_mean(eov_terms)


def compute_pareto_front(solutions):
    pareto = []
    for i, sol_i in enumerate(solutions):
        dominated = False
        for j, sol_j in enumerate(solutions):
            if i == j:
                continue
            if (
                sol_j["V"] <= sol_i["V"]
                and sol_j["U"] >= sol_i["U"]
                and (sol_j["V"] < sol_i["V"] or sol_j["U"] > sol_i["U"])
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(sol_i)
    return pareto


def normalize_pareto(pareto):
    if not pareto:
        return []
    v_values = [point["V"] for point in pareto]
    u_values = [point["U"] for point in pareto]
    v_min = min(v_values)
    v_max = max(v_values)
    u_min = min(u_values)
    u_max = max(u_values)
    dv = v_max - v_min or 1.0
    du = u_max - u_min or 1.0
    normalized = []
    for point in pareto:
        normalized.append(
            {
                **point,
                "V_norm": (point["V"] - v_min) / dv,
                "U_norm": (point["U"] - u_min) / du,
            }
        )
    return normalized


def select_global_criterion(pareto_norm, v_star=0.0, u_star=1.0):
    best = None
    best_dist = None
    for point in pareto_norm:
        dist = ((point["V_norm"] - v_star) ** 2 + (u_star - point["U_norm"]) ** 2) ** 0.5
        if best is None or dist < best_dist:
            best = point
            best_dist = dist
    return best
