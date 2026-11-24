from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "steam_classification_ready_clustered.csv"
)
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
DECISIONS_PATH = OUTPUT_DIR / "new_game_decisions.csv"
CHART_PATH = FIGURES_DIR / "new_game_decisions.png"

TARGET_COLUMN = "is_highly_rated"
RANDOM_STATE = 42
REFERENCE_YEAR = 2019
LEAKAGE_COLUMNS = {
    "appid",
    "name",
    "positive_ratings",
    "negative_ratings",
    "positive_ratio",
    "total_reviews",
    TARGET_COLUMN,
    "cluster_id",
}


PROFILES = [
    {
        "profile_name": "Free indie puzzle casual",
        "price": 0.0,
        "platform_count": 1,
        "release_year": 2019,
        "achievements": 20,
        "average_playtime": 0,
        "median_playtime": 0,
        "owners_midpoint": 10_000,
        "cluster_id": 1,
        "required_age": 0,
        "genre_indie": 1,
        "genre_casual": 1,
        "genre_action": 0,
        "genre_adventure": 0,
        "genre_strategy": 0,
        "genre_simulation": 0,
        "genre_rpg": 0,
        "genre_early_access": 0,
        "genre_free_to_play": 1,
        "genre_sports": 0,
        "tag_indie": 1,
        "tag_casual": 1,
        "tag_puzzle": 1,
        "tag_action": 0,
        "tag_adventure": 0,
        "tag_strategy": 0,
        "tag_simulation": 0,
        "tag_early_access": 0,
        "tag_rpg": 0,
        "tag_free_to_play": 1,
        "tag_vr": 0,
        "tag_sports": 0,
        "tag_racing": 0,
        "tag_platformer": 0,
        "tag_nudity": 0,
    },
    {
        "profile_name": "Low-cost action RPG",
        "price": 4.99,
        "platform_count": 3,
        "release_year": 2018,
        "achievements": 35,
        "average_playtime": 180,
        "median_playtime": 120,
        "owners_midpoint": 35_000,
        "cluster_id": 1,
        "required_age": 13,
        "genre_indie": 1,
        "genre_action": 1,
        "genre_casual": 0,
        "genre_adventure": 1,
        "genre_strategy": 0,
        "genre_simulation": 0,
        "genre_rpg": 1,
        "genre_early_access": 0,
        "genre_free_to_play": 0,
        "genre_sports": 0,
        "tag_indie": 1,
        "tag_action": 1,
        "tag_adventure": 1,
        "tag_rpg": 1,
        "tag_casual": 0,
        "tag_puzzle": 0,
        "tag_strategy": 0,
        "tag_simulation": 0,
        "tag_early_access": 0,
        "tag_free_to_play": 0,
        "tag_vr": 0,
        "tag_sports": 0,
        "tag_racing": 0,
        "tag_platformer": 0,
        "tag_nudity": 0,
    },
    {
        "profile_name": "Mid-priced strategy simulation",
        "price": 14.99,
        "platform_count": 2,
        "release_year": 2019,
        "achievements": 50,
        "average_playtime": 240,
        "median_playtime": 180,
        "owners_midpoint": 150_000,
        "cluster_id": 1,
        "required_age": 0,
        "genre_indie": 1,
        "genre_action": 0,
        "genre_casual": 0,
        "genre_adventure": 0,
        "genre_strategy": 1,
        "genre_simulation": 1,
        "genre_rpg": 0,
        "genre_early_access": 0,
        "genre_free_to_play": 0,
        "genre_sports": 0,
        "tag_indie": 0,
        "tag_action": 0,
        "tag_adventure": 0,
        "tag_rpg": 0,
        "tag_casual": 0,
        "tag_puzzle": 0,
        "tag_strategy": 1,
        "tag_simulation": 1,
        "tag_early_access": 0,
        "tag_free_to_play": 0,
        "tag_vr": 0,
        "tag_sports": 0,
        "tag_racing": 0,
        "tag_platformer": 0,
        "tag_nudity": 0,
    },
    {
        "profile_name": "Premium sports racing title",
        "price": 59.99,
        "platform_count": 2,
        "release_year": 2017,
        "achievements": 80,
        "average_playtime": 900,
        "median_playtime": 600,
        "owners_midpoint": 1_500_000,
        "cluster_id": 0,
        "required_age": 0,
        "genre_indie": 0,
        "genre_action": 0,
        "genre_casual": 0,
        "genre_adventure": 0,
        "genre_strategy": 0,
        "genre_simulation": 1,
        "genre_rpg": 0,
        "genre_early_access": 0,
        "genre_free_to_play": 0,
        "genre_sports": 1,
        "tag_indie": 0,
        "tag_action": 0,
        "tag_adventure": 0,
        "tag_rpg": 0,
        "tag_casual": 0,
        "tag_puzzle": 0,
        "tag_strategy": 0,
        "tag_simulation": 1,
        "tag_early_access": 0,
        "tag_free_to_play": 0,
        "tag_vr": 0,
        "tag_sports": 1,
        "tag_racing": 1,
        "tag_platformer": 0,
        "tag_nudity": 0,
    },
    {
        "profile_name": "Premium VR adventure title",
        "price": 29.99,
        "platform_count": 1,
        "release_year": 2019,
        "achievements": 45,
        "average_playtime": 420,
        "median_playtime": 300,
        "owners_midpoint": 350_000,
        "cluster_id": 0,
        "required_age": 16,
        "genre_indie": 1,
        "genre_action": 1,
        "genre_casual": 0,
        "genre_adventure": 1,
        "genre_strategy": 0,
        "genre_simulation": 0,
        "genre_rpg": 0,
        "genre_early_access": 0,
        "genre_free_to_play": 0,
        "genre_sports": 0,
        "tag_indie": 1,
        "tag_action": 1,
        "tag_adventure": 1,
        "tag_rpg": 0,
        "tag_casual": 0,
        "tag_puzzle": 0,
        "tag_strategy": 0,
        "tag_simulation": 0,
        "tag_early_access": 0,
        "tag_free_to_play": 0,
        "tag_vr": 1,
        "tag_sports": 0,
        "tag_racing": 0,
        "tag_platformer": 0,
        "tag_nudity": 0,
    },
]


def recommendation(probability):
    if probability >= 0.60:
        return "promote"
    if probability >= 0.40:
        return "review"
    return "hold"


def derive_profile_features(profile):
    row = dict(profile)
    row["english"] = 1
    row["is_free_to_play"] = int(row["price"] == 0)
    row["game_age_years"] = REFERENCE_YEAR - int(row["release_year"])
    row["log_price"] = np.log1p(row["price"])
    row["log_avg_playtime"] = np.log1p(row["average_playtime"])
    row["log_median_playtime"] = np.log1p(row["median_playtime"])
    row["log_achievements"] = np.log1p(row["achievements"])
    row["log_owners_midpoint"] = np.log1p(row["owners_midpoint"])
    row["avg_to_median_playtime_ratio"] = (
        row["average_playtime"] / (row["median_playtime"] + 1)
    )
    return row


def save_decision_chart(results):
    chart_data = results.sort_values("probability_highly_rated", ascending=True)
    labels = [
        f"{row.profile_name} (cluster {row.cluster_id})"
        for row in chart_data.itertuples(index=False)
    ]
    colors = [
        "#2E8B57"
        if row.recommendation == "promote"
        else "#D69E2E"
        if row.recommendation == "review"
        else "#C94C4C"
        for row in chart_data.itertuples(index=False)
    ]
    figure, axis = plt.subplots(figsize=(11, 6))
    bars = axis.barh(
        labels,
        chart_data["probability_highly_rated"],
        color=colors,
        edgecolor="white",
    )
    axis.axvline(0.40, color="#D69E2E", linestyle="--", linewidth=1.2)
    axis.axvline(0.50, color="#6B7280", linestyle=":", linewidth=1.2)
    axis.axvline(0.60, color="#2E8B57", linestyle="--", linewidth=1.2)
    axis.set(
        title="New-Game High-Reception Probability",
        xlabel="Predicted probability",
        xlim=(0, 1),
    )
    axis.grid(axis="x", alpha=0.25)
    for bar, row in zip(bars, chart_data.itertuples(index=False)):
        axis.text(
            max(0.015, bar.get_width() - 0.015),
            bar.get_y() + bar.get_height() / 2,
            f"{row.probability_highly_rated:.3f} - {row.recommendation}",
            ha="right",
            va="center",
            fontsize=9,
            color="#FFFFFF" if bar.get_width() >= 0.50 else "#202020",
        )
    figure.tight_layout()
    figure.savefig(CHART_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)


def run_decision_simulation():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Clustered classification data not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    required_columns = {TARGET_COLUMN, "cluster_id"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"Classification data is missing columns: {sorted(missing_columns)}"
        )
    if data[TARGET_COLUMN].isna().any():
        raise ValueError("Target contains missing values")
    if set(data[TARGET_COLUMN].dropna().unique()) != {0, 1}:
        raise ValueError("Target must contain binary values 0 and 1")

    excluded = LEAKAGE_COLUMNS | {
        "release_date",
        "developer",
        "publisher",
        "platforms",
        "categories",
        "genres",
        "steamspy_tags",
        "owners",
    }
    numeric_features = [
        column
        for column in data.select_dtypes(include=np.number).columns
        if column not in excluded
    ]
    if not numeric_features:
        raise ValueError("No numeric predictor columns were found")
    if "cluster_id" not in data.columns:
        raise ValueError("Cluster assignment column is missing")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric_features,
            ),
            (
                "cluster",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["cluster_id"],
            ),
        ],
        remainder="drop",
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    x_train = data[numeric_features + ["cluster_id"]]
    y_train = data[TARGET_COLUMN].astype(int)
    model.fit(x_train, y_train)

    profile_rows = []
    for profile in PROFILES:
        derived = derive_profile_features(profile)
        profile_rows.append(derived)
    profile_frame = pd.DataFrame(profile_rows)

    for numeric_feature in numeric_features:
        if numeric_feature not in profile_frame.columns:
            profile_frame[numeric_feature] = data[numeric_feature].median()
    profile_frame["cluster_id"] = profile_frame["cluster_id"].astype(int)
    x_new = profile_frame[numeric_features + ["cluster_id"]]

    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(x_new)[:, positive_index]
    predicted_classes = (probabilities >= 0.50).astype(int)

    results = pd.DataFrame(
        {
            "profile_name": profile_frame["profile_name"],
            "price": profile_frame["price"],
            "platform_count": profile_frame["platform_count"],
            "release_year": profile_frame["release_year"],
            "game_age_years": profile_frame["game_age_years"],
            "cluster_id": profile_frame["cluster_id"],
            "probability_highly_rated": probabilities,
            "predicted_is_highly_rated": predicted_classes,
            "recommendation": [
                recommendation(probability) for probability in probabilities
            ],
        }
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(DECISIONS_PATH, index=False)
    save_decision_chart(results)

    print(f"Training rows: {len(data):,}")
    print(f"Training target counts: {y_train.value_counts().sort_index().to_dict()}")
    print(f"Numeric predictors: {len(numeric_features)}")
    print(f"Clustered profiles scored: {len(results)}")
    print(f"Saved decision table: {DECISIONS_PATH}")
    print(f"Saved decision chart: {CHART_PATH}")
    print("\nDecision simulation results:")
    print(
        results[
            [
                "profile_name",
                "probability_highly_rated",
                "predicted_is_highly_rated",
                "recommendation",
            ]
        ].to_string(index=False, formatters={
            "probability_highly_rated": lambda value: f"{value:.4f}"
        })
    )
    print("\nLimitations:")
    print("- Data coverage: 1997-2019; results may not generalize to newer markets.")
    print("- Training excludes games with fewer than 20 reviews.")
    print("- Owners are interval-binned midpoints, not exact sales.")
    print("- Playtime and ownership are post-release assumptions in hypothetical profiles.")
    print("- The earlier Random Forest ROC-AUC was about 0.71; probabilities are uncertain.")
    print("- Cluster IDs are unsupervised and may shift with retraining.")
    print("- This demonstrates probabilistic decision support, not a certainty or causal effect.")


if __name__ == "__main__":
    run_decision_simulation()
