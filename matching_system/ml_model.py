"""Train and evaluate a Logistic Regression model for lost/found report pairs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .feature_builder import FEATURE_NAMES


def new_classifier():
    # Scaling is fitted inside each fold, so validation rows cannot influence it.
    # Class weighting offsets the constructed exercise's 1:2 label ratio;
    # it does not correct the uncertainty of the assumed negative labels.
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    return make_pipeline(StandardScaler(), model)


def metrics(y_true, probabilities, threshold=0.5):
    # Apply one explicit threshold to every split so precision and recall
    # describe the same decision rule throughout the experiment.
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
    }


def run_experiment(table_path: Path, split_path: Path, results_dir: Path) -> dict:
    table = pd.read_csv(table_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))["splits"]
    development = table[table.lost_report_id.isin(split["development"]["lost_ids"])].copy()
    test = table[table.lost_report_id.isin(split["test"]["lost_ids"])].copy()
    if len(development) + len(test) != len(table):
        raise ValueError("Split manifest does not cover all labelled pairs")
    if set(development.lost_report_id) & set(test.lost_report_id):
        raise ValueError("Lost report leakage between development and test")
    if set(development.found_report_id) & set(test.found_report_id):
        raise ValueError("Found report leakage between development and test")
    if set(development.target) != {0, 1} or set(test.target) != {0, 1}:
        raise ValueError("Both partitions need positive and negative examples")

    # Use only the development partition for grouped cross-validation.
    folds = GroupKFold(n_splits=min(5, development.lost_report_id.nunique()))
    fold_results = []
    for train_idx, valid_idx in folds.split(
        development[list(FEATURE_NAMES)], development.target, development.lost_report_id
    ):
        train, valid = development.iloc[train_idx], development.iloc[valid_idx]
        if set(train.found_report_id) & set(valid.found_report_id):
            raise ValueError("Found report leakage inside a development fold")
        if set(train.target) != {0, 1} or set(valid.target) != {0, 1}:
            # Do not report a mathematically invalid fold as a successful validation.
            continue
        classifier = new_classifier().fit(train[list(FEATURE_NAMES)], train.target)
        probabilities = classifier.predict_proba(valid[list(FEATURE_NAMES)])[:, 1]
        fold_results.append({
            "train_pairs": len(train),
            "validation_pairs": len(valid),
            "validation_lost_ids": sorted(valid.lost_report_id.unique().tolist()),
            **metrics(valid.target, probabilities),
        })
    if not fold_results:
        raise ValueError("No valid grouped validation folds")
    # GroupKFold can enumerate equally sized groups in a different order across versions.
    fold_results.sort(key=lambda row: row["validation_lost_ids"])
    for number, row in enumerate(fold_results, 1):
        row["fold"] = number

    classifier = new_classifier().fit(development[list(FEATURE_NAMES)], development.target)
    probabilities = classifier.predict_proba(test[list(FEATURE_NAMES)])[:, 1]
    predictions = test[["lost_report_id", "found_report_id", "original_label", "target"]].copy()
    predictions["match_probability"] = probabilities
    predictions["predicted_class"] = (probabilities >= 0.5).astype(int)
    coefficients = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "standardized_coefficient": classifier.named_steps["logisticregression"].coef_[0],
    })
    coefficients = coefficients.sort_values("standardized_coefficient", ascending=False)

    # Only the older campus exercise has an uncertain 'plausible' label.
    sensitivity = None
    if "plausible" in set(table.original_label):
        clear_dev = development[development.original_label != "plausible"]
        clear_test = test[test.original_label != "plausible"]
        clear_classifier = new_classifier().fit(clear_dev[list(FEATURE_NAMES)], clear_dev.target)
        clear_probabilities = clear_classifier.predict_proba(clear_test[list(FEATURE_NAMES)])[:, 1]
        sensitivity = {
            "development_pairs": len(clear_dev),
            "test_pairs": len(clear_test),
            **metrics(clear_test.target, clear_probabilities),
        }

    results_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(results_dir / "test_predictions.csv", index=False)
    coefficients.to_csv(results_dir / "coefficients.csv", index=False)
    result = {
        "target_mapping": {label: int(table.loc[table.original_label == label, "target"].iloc[0])
                           for label in sorted(table.original_label.unique())},
        "label_caution": ("Plausible means unconfirmed; class 0 is a coursework assumption."
                          if sensitivity else "Review label provenance before interpreting these metrics."),
        "split": {
            "development_pairs": len(development),
            "test_pairs": len(test),
            "development_lost_reports": development.lost_report_id.nunique(),
            "test_lost_reports": test.lost_report_id.nunique(),
        },
        "grouped_cv": fold_results,
        "held_out_test": metrics(test.target, probabilities),
        "plausible_excluded_sensitivity": sensitivity,
        "limitations": [
            "Designed or constructed pair labels are not verified real handovers",
            "No independent real lost-and-found field test",
            "Grouped validation does not remove label-provenance limitations",
        ],
    }
    (results_dir / "validation_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
