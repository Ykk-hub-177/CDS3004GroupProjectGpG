from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parent.parent
FULL_DATA_PATH = ROOT_DIR / "data" / "processed" / "steam_features_full.csv"
CLASSIFICATION_DATA_PATH = (
    ROOT_DIR / "data" / "processed" / "steam_classification_ready.csv"
)
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"

FEATURE_COLUMNS = [
    "log_price",
    "log_avg_playtime",
    "log_median_playtime",
    "log_achievements",
    "log_owners_midpoint",
    "platform_count",
    "game_age_years",
]
PROFILE_COLUMNS = [
    "price",
    "average_playtime",
    "median_playtime",
    "achievements",
    "owners_midpoint",
    "platform_count",
    "game_age_years",
]
K_VALUES = range(2, 9)
SILHOUETTE_SAMPLE_SIZE = 5_000
RANDOM_STATE = 42


def load_full_data():
    if not FULL_DATA_PATH.is_file():
        raise FileNotFoundError(f"Full feature data not found: {FULL_DATA_PATH}")

    data = pd.read_csv(FULL_DATA_PATH)
    required_columns = {"appid", *FEATURE_COLUMNS, *PROFILE_COLUMNS}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"Full feature data is missing columns: {sorted(missing_columns)}"
        )
    if data["appid"].isna().any():
        raise ValueError("Full feature data contains missing appid values")
    if data["appid"].duplicated().any():
        raise ValueError("Full feature data contains duplicate appid values")
    if len(data) <= max(K_VALUES):
        raise ValueError(
            f"At least {max(K_VALUES) + 1} games are required for K selection"
        )

    for column in set(FEATURE_COLUMNS).union(PROFILE_COLUMNS):
        data[column] = pd.to_numeric(data[column], errors="raise")

    feature_values = data[FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(feature_values).all():
        raise ValueError("Clustering features contain missing or non-finite values")
    profile_values = data[PROFILE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(profile_values).all():
        raise ValueError("Cluster profile features contain missing or non-finite values")

    return data, feature_values


def select_k(feature_values):
    scaled_features = StandardScaler().fit_transform(feature_values)
    sample_size = min(SILHOUETTE_SAMPLE_SIZE, len(scaled_features))
    sample_indices = np.random.default_rng(RANDOM_STATE).choice(
        len(scaled_features), size=sample_size, replace=False
    )

    metrics = []
    fitted_models = {}
    for k in K_VALUES:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        model.fit(scaled_features)
        sample_labels = model.labels_[sample_indices]
        if np.unique(sample_labels).size < 2:
            raise ValueError(
                f"Silhouette score for K={k} requires at least two sampled clusters"
            )
        metrics.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette_score": silhouette_score(
                    scaled_features[sample_indices], sample_labels
                ),
                "silhouette_sample_size": sample_size,
            }
        )
        fitted_models[k] = model

    metrics_data = pd.DataFrame(metrics)
    selected_k = int(metrics_data.loc[metrics_data["silhouette_score"].idxmax(), "k"])
    return metrics_data, fitted_models[selected_k], selected_k


def save_selection_plot(metrics_data, selected_k):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(metrics_data["k"], metrics_data["inertia"], marker="o")
    axes[0].set(title="K-Means Elbow", xlabel="K", ylabel="Inertia")
    axes[1].plot(
        metrics_data["k"], metrics_data["silhouette_score"], marker="o", color="#C0504D"
    )
    axes[1].set(
        title="Silhouette Score (sampled)",
        xlabel="K",
        ylabel="Silhouette score",
    )
    for axis in axes:
        axis.axvline(
            selected_k,
            color="#70AD47",
            linestyle="--",
            label=f"Selected K={selected_k}",
        )
        axis.set_xticks(metrics_data["k"])
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.tight_layout()
    return figure


def create_cluster_profiles(data, labels):
    clustered_data = data.drop(columns=["cluster_id"], errors="ignore").copy()
    clustered_data["cluster_id"] = labels
    grouped = clustered_data.groupby("cluster_id", sort=True)
    profile = grouped.size().rename("game_count").to_frame()
    profile["share"] = profile["game_count"] / len(clustered_data)

    for column in PROFILE_COLUMNS:
        statistics = grouped[column].agg(["mean", "median"]).rename(
            columns={
                "mean": f"{column}_mean",
                "median": f"{column}_median",
            }
        )
        profile = profile.join(statistics)

    return clustered_data, profile.reset_index()


def save_profile_plot(model, profile):
    centers = model.cluster_centers_
    max_abs_value = max(float(np.abs(centers).max()), 1.0)
    figure, axis = plt.subplots(figsize=(11, max(4.5, len(centers) * 0.75)))
    image = axis.imshow(
        centers,
        aspect="auto",
        cmap="coolwarm",
        vmin=-max_abs_value,
        vmax=max_abs_value,
    )
    axis.set_xticks(range(len(FEATURE_COLUMNS)))
    axis.set_xticklabels(FEATURE_COLUMNS, rotation=35, ha="right")
    axis.set_yticks(
        range(len(profile)),
        [
            f"Cluster {row.cluster_id} (n={row.game_count:,})"
            for row in profile.itertuples(index=False)
        ],
    )
    axis.set_title("Cluster Profiles (standardized feature centers)")
    figure.colorbar(image, ax=axis, label="Standardized feature value")
    figure.tight_layout()
    return figure


def add_cluster_ids_to_classification(clustered_data):
    if not CLASSIFICATION_DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Classification feature data not found: {CLASSIFICATION_DATA_PATH}"
        )

    classification = pd.read_csv(CLASSIFICATION_DATA_PATH)
    if "appid" not in classification.columns:
        raise ValueError("Classification feature data is missing appid")
    if classification["appid"].isna().any():
        raise ValueError("Classification feature data contains missing appid values")
    if classification["appid"].duplicated().any():
        raise ValueError("Classification feature data contains duplicate appid values")

    assignments = clustered_data[["appid", "cluster_id"]]
    classification = classification.drop(columns=["cluster_id"], errors="ignore")
    clustered_classification = classification.merge(
        assignments, on="appid", how="left", validate="one_to_one"
    )
    if clustered_classification["cluster_id"].isna().any():
        raise ValueError("Some classification games have no matching cluster assignment")
    return clustered_classification


def create_clusters():
    data, feature_values = load_full_data()
    (
        metrics_data,
        selected_model,
        selected_k,
    ) = select_k(feature_values)
    clustered_data, profile = create_cluster_profiles(data, selected_model.labels_)
    clustered_classification = add_cluster_ids_to_classification(clustered_data)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    clustered_data.to_csv(PROCESSED_DIR / "steam_features_clustered.csv", index=False)
    clustered_classification.to_csv(
        PROCESSED_DIR / "steam_classification_ready_clustered.csv", index=False
    )
    profile.to_csv(OUTPUT_DIR / "steam_cluster_profiles.csv", index=False)
    metrics_data.to_csv(OUTPUT_DIR / "kmeans_selection_metrics.csv", index=False)

    figures = [
        (save_selection_plot(metrics_data, selected_k), "kmeans_selection.png"),
        (save_profile_plot(selected_model, profile), "steam_cluster_profiles.png"),
    ]
    for figure, filename in figures:
        output_path = FIGURES_DIR / filename
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved plot: {output_path}")

    print(f"Selected K by highest sampled silhouette score: {selected_k}")
    print(f"Silhouette sample size: {metrics_data['silhouette_sample_size'].iloc[0]:,}")
    print(f"Saved clustered feature data: {PROCESSED_DIR / 'steam_features_clustered.csv'}")
    print(
        "Saved clustered classification data: "
        f"{PROCESSED_DIR / 'steam_classification_ready_clustered.csv'}"
    )
    print(f"Saved cluster profiles: {OUTPUT_DIR / 'steam_cluster_profiles.csv'}")
    print(f"Cluster sizes: {profile.set_index('cluster_id')['game_count'].to_dict()}")


if __name__ == "__main__":
    create_clusters()
