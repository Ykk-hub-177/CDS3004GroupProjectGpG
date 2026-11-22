import re
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT_DIR / "data" / "processed" / "steam_cleaned.csv"
FULL_OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "steam_features_full.csv"
CLASSIFICATION_OUTPUT_PATH = (
    ROOT_DIR / "data" / "processed" / "steam_classification_ready.csv"
)

MINIMUM_REVIEWS = 20
HIGH_RECEPTION_THRESHOLD = 0.80
TOP_GENRE_COUNT = 10
TOP_TAG_COUNT = 15
LEAKAGE_COLUMNS = [
    "positive_ratings",
    "negative_ratings",
    "positive_ratio",
    "total_reviews",
]


def split_terms(value):
    return [term.strip() for term in value.split(";") if term.strip()]


def top_terms(series, limit):
    counts = Counter(term for value in series for term in split_terms(value))
    return [
        term
        for term, _ in sorted(
            counts.items(), key=lambda item: (-item[1], item[0].casefold(), item[0])
        )[:limit]
    ]


def feature_slug(term):
    slug = re.sub(r"[^a-z0-9]+", "_", term.casefold()).strip("_")
    if not slug:
        raise ValueError(f"Cannot create a feature name from term {term!r}")
    return slug


def add_term_indicators(data, source_column, prefix, terms):
    feature_columns = []
    for term in terms:
        column = f"{prefix}{feature_slug(term)}"
        if column in data.columns:
            raise ValueError(f"Generated feature column already exists: {column}")
        data[column] = data[source_column].map(
            lambda value: int(term in split_terms(value))
        )
        feature_columns.append(column)
    return feature_columns


def create_datasets():
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Cleaned Steam data not found: {INPUT_PATH}")

    data = pd.read_csv(INPUT_PATH)
    required_columns = {
        "release_year",
        "platforms",
        "genres",
        "steamspy_tags",
        "price",
        "average_playtime",
        "median_playtime",
        "total_reviews",
        "positive_ratio",
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Input is missing required columns: {sorted(missing_columns)}")

    numeric_columns = (
        "release_year",
        "price",
        "average_playtime",
        "median_playtime",
        "total_reviews",
        "positive_ratio",
    )
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")

    reference_year = int(data["release_year"].max())
    data["platform_count"] = data["platforms"].map(
        lambda value: len(split_terms(value))
    )
    data["game_age_years"] = reference_year - data["release_year"]
    data["avg_to_median_playtime_ratio"] = data["average_playtime"] / (
        data["median_playtime"] + 1
    )
    data["is_free_to_play"] = (data["price"] == 0).astype(int)

    genre_terms = top_terms(data["genres"], TOP_GENRE_COUNT)
    tag_terms = top_terms(data["steamspy_tags"], TOP_TAG_COUNT)
    genre_features = add_term_indicators(data, "genres", "genre_", genre_terms)
    tag_features = add_term_indicators(data, "steamspy_tags", "tag_", tag_terms)

    engineered_columns = [
        "platform_count",
        "game_age_years",
        "avg_to_median_playtime_ratio",
        "is_free_to_play",
        *genre_features,
        *tag_features,
    ]

    classification = data.loc[data["total_reviews"] >= MINIMUM_REVIEWS].copy()
    classification["is_highly_rated"] = (
        classification["positive_ratio"] >= HIGH_RECEPTION_THRESHOLD
    ).astype(int)
    classification = classification.drop(columns=LEAKAGE_COLUMNS)

    FULL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(FULL_OUTPUT_PATH, index=False)
    classification.to_csv(CLASSIFICATION_OUTPUT_PATH, index=False)

    print(f"Saved full feature dataset: {FULL_OUTPUT_PATH}")
    print(f"  Shape: {data.shape}")
    print(f"Saved classification dataset: {CLASSIFICATION_OUTPUT_PATH}")
    print(f"  Shape: {classification.shape}")
    print(f"  Target counts: {classification['is_highly_rated'].value_counts().sort_index().to_dict()}")
    print(f"Engineered feature columns ({len(engineered_columns)}):")
    for column in engineered_columns:
        print(f"  {column}")


if __name__ == "__main__":
    create_datasets()
