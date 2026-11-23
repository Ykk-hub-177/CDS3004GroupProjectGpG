from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


ROOT_DIR = Path(__file__).resolve().parent.parent
CLASSIFICATION_DATA_PATH = (
    ROOT_DIR / "data" / "processed" / "steam_classification_ready.csv"
)
CLUSTERED_DATA_PATH = (
    ROOT_DIR / "data" / "processed" / "steam_classification_ready_clustered.csv"
)
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"

TARGET_COLUMN = "is_highly_rated"
IDENTIFIER_COLUMNS = {"appid"}
LEAKAGE_COLUMNS = {
    "positive_ratings",
    "negative_ratings",
    "positive_ratio",
    "total_reviews",
}
N_SPLITS = 5
RANDOM_STATE = 42


def load_datasets():
    if not CLASSIFICATION_DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Classification data not found: {CLASSIFICATION_DATA_PATH}"
        )
    if not CLUSTERED_DATA_PATH.is_file():
        raise FileNotFoundError(f"Clustered classification data not found: {CLUSTERED_DATA_PATH}")

    data = pd.read_csv(CLASSIFICATION_DATA_PATH)
    clustered_data = pd.read_csv(CLUSTERED_DATA_PATH)
    required_columns = {"appid", TARGET_COLUMN}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Classification data is missing columns: {sorted(missing_columns)}")
    if data["appid"].isna().any() or data["appid"].duplicated().any():
        raise ValueError("Classification data must have a unique, non-missing appid per row")
    if data[TARGET_COLUMN].isna().any():
        raise ValueError(f"Classification target contains missing values: {TARGET_COLUMN}")

    assignment_columns = {"appid", "cluster_id"}
    missing_assignments = assignment_columns.difference(clustered_data.columns)
    if missing_assignments:
        raise ValueError(
            f"Clustered classification data is missing columns: {sorted(missing_assignments)}"
        )
    assignments = clustered_data[["appid", "cluster_id"]]
    if assignments["appid"].isna().any() or assignments["appid"].duplicated().any():
        raise ValueError("Cluster assignments must have a unique, non-missing appid per row")
    if len(assignments) != len(data) or set(assignments["appid"]) != set(data["appid"]):
        raise ValueError("Cluster assignments do not match the classification dataset")

    data = data.merge(assignments, on="appid", how="left", validate="one_to_one")
    if data["cluster_id"].isna().any():
        raise ValueError("Some classification games have no cluster assignment")
    data["cluster_id"] = pd.to_numeric(data["cluster_id"], errors="raise")
    if data[TARGET_COLUMN].nunique() != 2 or set(data[TARGET_COLUMN].unique()) != {0, 1}:
        raise ValueError(f"Expected binary target labels 0 and 1 in {TARGET_COLUMN}")

    excluded_columns = IDENTIFIER_COLUMNS | LEAKAGE_COLUMNS | {
        TARGET_COLUMN,
        "cluster_id",
    }
    numeric_features = [
        column
        for column in data.select_dtypes(include=np.number).columns
        if column not in excluded_columns
    ]
    if not numeric_features:
        raise ValueError("No numeric predictor columns were found")

    return data, numeric_features


def create_preprocessor(numeric_features, include_cluster):
    transformers = [
        (
            "numeric",
            SimpleImputer(strategy="median"),
            numeric_features,
        )
    ]
    if include_cluster:
        cluster_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cluster", cluster_transformer, ["cluster_id"]))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def evaluate_configuration(
    data,
    target,
    numeric_features,
    model_name,
    estimator,
    include_cluster,
    folds,
):
    feature_columns = list(numeric_features)
    if include_cluster:
        feature_columns.append("cluster_id")
    features = data[feature_columns]

    out_of_fold_predictions = np.empty(len(target), dtype=int)
    out_of_fold_probabilities = np.empty(len(target), dtype=float)
    fold_results = []

    for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
        pipeline = Pipeline(
            [
                (
                    "preprocessor",
                    create_preprocessor(numeric_features, include_cluster),
                ),
                ("classifier", clone(estimator)),
            ]
        )
        x_train = features.iloc[train_indices]
        x_validation = features.iloc[validation_indices]
        y_train = target.iloc[train_indices]
        y_validation = target.iloc[validation_indices]

        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_validation)
        positive_class_index = list(pipeline.classes_).index(1)
        probabilities = pipeline.predict_proba(x_validation)[:, positive_class_index]
        out_of_fold_predictions[validation_indices] = predictions
        out_of_fold_probabilities[validation_indices] = probabilities

        fold_results.append(
            {
                "model": model_name,
                "includes_cluster": include_cluster,
                "fold": fold_number,
                "accuracy": accuracy_score(y_validation, predictions),
                "precision": precision_score(
                    y_validation, predictions, pos_label=1, zero_division=0
                ),
                "recall": recall_score(
                    y_validation, predictions, pos_label=1, zero_division=0
                ),
                "f1_score": f1_score(
                    y_validation, predictions, pos_label=1, zero_division=0
                ),
                "roc_auc": roc_auc_score(y_validation, probabilities),
            }
        )

    out_of_fold_metrics = {
        "model": model_name,
        "includes_cluster": include_cluster,
        "accuracy": accuracy_score(target, out_of_fold_predictions),
        "precision": precision_score(
            target, out_of_fold_predictions, pos_label=1, zero_division=0
        ),
        "recall": recall_score(
            target, out_of_fold_predictions, pos_label=1, zero_division=0
        ),
        "f1_score": f1_score(
            target, out_of_fold_predictions, pos_label=1, zero_division=0
        ),
        "roc_auc": roc_auc_score(target, out_of_fold_probabilities),
    }
    matrix = confusion_matrix(target, out_of_fold_predictions, labels=[0, 1])
    return fold_results, out_of_fold_metrics, matrix


def save_confusion_matrix_plot(confusion_matrices):
    figure, axes = plt.subplots(2, 2, figsize=(11, 9))
    for axis, (model_name, include_cluster, matrix) in zip(
        axes.flat, confusion_matrices
    ):
        ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=["Not highly rated (0)", "Highly rated (1)"],
        ).plot(
            ax=axis,
            cmap="Blues",
            colorbar=False,
            values_format="d",
        )
        cluster_label = "with cluster_id" if include_cluster else "without cluster_id"
        axis.set_title(f"{model_name} — {cluster_label}")
    figure.suptitle("Out-of-Fold Confusion Matrices (5-Fold Stratified CV)")
    figure.tight_layout()
    return figure


def run_cross_validation():
    data, numeric_features = load_datasets()
    target = data[TARGET_COLUMN].astype(int)
    cross_validator = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    folds = list(cross_validator.split(data, target))
    estimators = {
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    fold_results = []
    out_of_fold_results = []
    confusion_matrices = []
    for model_name, estimator in estimators.items():
        for include_cluster in (False, True):
            results, overall_metrics, matrix = evaluate_configuration(
                data,
                target,
                numeric_features,
                model_name,
                estimator,
                include_cluster,
                folds,
            )
            fold_results.extend(results)
            out_of_fold_results.append(overall_metrics)
            confusion_matrices.append((model_name, include_cluster, matrix))

    fold_metrics = pd.DataFrame(fold_results)
    metric_columns = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    summary = (
        fold_metrics.groupby(["model", "includes_cluster"])[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    overall_metrics = pd.DataFrame(out_of_fold_results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "classification_cv_metrics.csv", index=False)
    overall_metrics.to_csv(
        OUTPUT_DIR / "classification_cv_oof_metrics.csv", index=False
    )
    figure = save_confusion_matrix_plot(confusion_matrices)
    figure.savefig(
        FIGURES_DIR / "classification_cv_confusion_matrices.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"Predictor count without cluster_id: {len(numeric_features)}")
    print(f"Target counts: {target.value_counts().sort_index().to_dict()}")
    print(f"Saved fold-mean metrics: {OUTPUT_DIR / 'classification_cv_metrics.csv'}")
    print(f"Saved out-of-fold metrics: {OUTPUT_DIR / 'classification_cv_oof_metrics.csv'}")
    print(
        "Saved confusion matrix plot: "
        f"{FIGURES_DIR / 'classification_cv_confusion_matrices.png'}"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run_cross_validation()
