import numpy as np

def calculate_dp(all_predictions, all_sensitive):
    """
    Calculate the overall fairness difference for Demographic Parity (DP).
    Handles both multi-label and multi-class cases.
    """
    DP = {}

    # Multi-label scenario (e.g., AU datasets)
    if len(all_predictions.shape) == 2:
        for label_idx in range(all_predictions.shape[1]):  # Iterate over each label
            DP[label_idx] = {}
            for sensitive_idx in set(all_sensitive):
                positive_rate = (all_predictions[:, label_idx][all_sensitive == sensitive_idx] == 1).mean()
                DP[label_idx][sensitive_idx] = positive_rate
    else:  # Multi-class scenario
        for class_idx in set(all_predictions):  # Iterate over unique classes
            DP[class_idx] = {}
            for sensitive_idx in set(all_sensitive):
                positive_rate = (all_predictions[all_sensitive == sensitive_idx] == class_idx).mean()
                DP[class_idx][sensitive_idx] = positive_rate

    # First level: calculate pairwise differences for each label or class
    avg_group_diff_per_class = {}
    for class_idx in DP.keys():
        pairwise_diffs = [abs(DP[class_idx][i] - DP[class_idx][j]) for i in DP[class_idx] for j in DP[class_idx] if i != j]
        avg_group_diff_per_class[class_idx] = np.mean(pairwise_diffs) if pairwise_diffs else 0

    # Second level: max-min difference across labels or classes
    overall_diff = max(avg_group_diff_per_class.values()) - min(avg_group_diff_per_class.values()) if avg_group_diff_per_class else 0

    return overall_diff


def calculate_eo(all_targets, all_predictions, all_sensitive):
    """
    Calculate the overall fairness difference for Equal Opportunity (EO).
    """
    EO = {}

    # Multi-label scenario (e.g., AU datasets)
    if len(all_targets.shape) == 2:
        for class_idx in range(all_targets.shape[1]):  # Iterate over each label
            EO[class_idx] = {}
            for sensitive_idx in set(all_sensitive):
                true_positive = ((all_targets[:, class_idx] == 1) & (all_predictions[:, class_idx] == 1) & (all_sensitive == sensitive_idx)).sum()
                actual_positive = ((all_targets[:, class_idx] == 1) & (all_sensitive == sensitive_idx)).sum()
                eo_value = true_positive / actual_positive if actual_positive > 0 else 0
                EO[class_idx][sensitive_idx] = eo_value
    else:  # Multi-class scenario
        for class_idx in set(all_targets):
            EO[class_idx] = {}
            for sensitive_idx in set(all_sensitive):
                true_positive = ((all_targets == class_idx) & (all_predictions == class_idx) & (all_sensitive == sensitive_idx)).sum()
                actual_positive = ((all_targets == class_idx) & (all_sensitive == sensitive_idx)).sum()
                eo_value = true_positive / actual_positive if actual_positive > 0 else 0
                EO[class_idx][sensitive_idx] = eo_value

    # First level: calculate pairwise differences for each emotion class
    avg_group_diff_per_class = {}
    for class_idx in EO.keys():
        pairwise_diffs = [abs(EO[class_idx][i] - EO[class_idx][j]) for i in EO[class_idx].keys() for j in EO[class_idx].keys() if i != j]
        avg_group_diff_per_class[class_idx] = np.mean(pairwise_diffs) if pairwise_diffs else 0

    # Second level: max-min difference across emotion classes
    overall_diff = max(avg_group_diff_per_class.values()) - min(avg_group_diff_per_class.values()) if avg_group_diff_per_class else 0

    return overall_diff


def calculate_eod(all_targets, all_predictions, all_sensitive):
    """
    Calculate the overall fairness difference for Equalized Odds (EOD).
    """
    EOD = {}

    # Multi-label scenario (e.g., AU datasets)
    if len(all_targets.shape) == 2:
        for class_idx in range(all_targets.shape[1]):  # Iterate over each label
            EOD[class_idx] = {}
            for sensitive_idx in set(all_sensitive):
                true_positive = ((all_targets[:, class_idx] == 1) & (all_predictions[:, class_idx] == 1) & (all_sensitive == sensitive_idx)).sum()
                actual_positive = ((all_targets[:, class_idx] == 1) & (all_sensitive == sensitive_idx)).sum()
                true_negative = ((all_targets[:, class_idx] == 0) & (all_predictions[:, class_idx] == 0) & (all_sensitive == sensitive_idx)).sum()
                actual_negative = ((all_targets[:, class_idx] == 0) & (all_sensitive == sensitive_idx)).sum()

                tp_rate = true_positive / actual_positive if actual_positive > 0 else 0
                tn_rate = true_negative / actual_negative if actual_negative > 0 else 0
                EOD[class_idx][sensitive_idx] = (tp_rate, tn_rate)
    else:  # Multi-class scenario
        for class_idx in set(all_targets):
            EOD[class_idx] = {}
            for sensitive_idx in set(all_sensitive):
                true_positive = ((all_targets == class_idx) & (all_predictions == class_idx) & (all_sensitive == sensitive_idx)).sum()
                actual_positive = ((all_targets == class_idx) & (all_sensitive == sensitive_idx)).sum()
                true_negative = ((all_targets == 0) & (all_predictions == 0) & (all_sensitive == sensitive_idx)).sum()
                actual_negative = ((all_targets == 0) & (all_sensitive == sensitive_idx)).sum()

                tp_rate = true_positive / actual_positive if actual_positive > 0 else 0
                tn_rate = true_negative / actual_negative if actual_negative > 0 else 0
                EOD[class_idx][sensitive_idx] = (tp_rate, tn_rate)

    # First level: calculate pairwise differences for each emotion class
    avg_group_diff_per_class = {}
    for class_idx in EOD.keys():
        tp_diffs = [abs(EOD[class_idx][i][0] - EOD[class_idx][j][0]) for i in EOD[class_idx].keys() for j in EOD[class_idx].keys() if i != j]
        tn_diffs = [abs(EOD[class_idx][i][1] - EOD[class_idx][j][1]) for i in EOD[class_idx].keys() for j in EOD[class_idx].keys() if i != j]
        avg_group_diff_per_class[class_idx] = (np.mean(tp_diffs) + np.mean(tn_diffs)) / 2 if tp_diffs and tn_diffs else 0

    # Second level: max-min difference across emotion classes
    overall_diff = max(avg_group_diff_per_class.values()) - min(avg_group_diff_per_class.values()) if avg_group_diff_per_class else 0

    return overall_diff