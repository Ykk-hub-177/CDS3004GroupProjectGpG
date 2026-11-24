from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
FULL_DATA_PATH = ROOT_DIR / "data" / "processed" / "steam_features_full.csv"
CLASSIFICATION_DATA_PATH = (
    ROOT_DIR / "data" / "processed" / "steam_classification_ready.csv"
)
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"


def save_price_distribution(full_data):
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(full_data["log_price"], bins=50, color="#4472C4", edgecolor="white")
    axis.set(
        title="Steam Game Price Distribution",
        xlabel="log1p(price)",
        ylabel="Number of games",
    )
    axis.text(
        0.98,
        0.95,
        f"Free games: {(full_data['price'] == 0).sum():,}",
        ha="right",
        va="top",
        transform=axis.transAxes,
    )
    figure.tight_layout()
    return figure, "price_distribution.png"


def save_reception_by_year(classification_data):
    yearly = classification_data.groupby("release_year")["is_highly_rated"].agg(
        ["mean", "count"]
    )
    years = yearly.index.astype(int)
    figure, axis = plt.subplots(figsize=(10, 5))
    count_axis = axis.twinx()
    count_axis.bar(
        years,
        yearly["count"],
        color="#A6A6A6",
        alpha=0.22,
        width=0.8,
        label="Eligible game count",
    )
    axis.plot(
        years,
        yearly["mean"],
        color="#C0504D",
        marker="o",
        linewidth=2,
        label="High-reception rate",
    )
    axis.set(
        title="High-Reception Rate by Release Year",
        xlabel="Release year",
        ylabel="Share with positive_ratio >= 0.80",
        ylim=(0, 1),
    )
    count_axis.set_ylabel("Eligible game count")
    axis.set_xticks(years[::2])
    axis.grid(axis="y", alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    count_handles, count_labels = count_axis.get_legend_handles_labels()
    figure.legend(
        handles + count_handles,
        labels + count_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return figure, "high_reception_by_release_year.png"


def save_reception_by_genre(classification_data):
    genre_columns = [
        column for column in classification_data.columns if column.startswith("genre_")
    ]
    genre_summary = []
    for column in genre_columns:
        selected = classification_data[column] == 1
        count = int(selected.sum())
        if count:
            genre_summary.append(
                (column.removeprefix("genre_"), classification_data.loc[
                    selected, "is_highly_rated"
                ].mean(), count)
            )
    genre_summary.sort(key=lambda item: item[1])

    labels = [item[0].replace("_", " ").title() for item in genre_summary]
    rates = [item[1] for item in genre_summary]
    counts = [item[2] for item in genre_summary]
    figure, axis = plt.subplots(figsize=(10, 6))
    bars = axis.barh(labels, rates, color="#70AD47")
    axis.set(
        title="High-Reception Rate by Common Genre",
        xlabel="Share with positive_ratio >= 0.80",
        xlim=(0, 1),
    )
    axis.grid(axis="x", alpha=0.25)
    for bar, count in zip(bars, counts):
        axis.text(
            min(bar.get_width() + 0.015, 0.97),
            bar.get_y() + bar.get_height() / 2,
            f"n={count:,}",
            va="center",
            fontsize=8,
        )
    figure.tight_layout()
    return figure, "high_reception_by_genre.png"


def save_playtime_vs_owners(classification_data):
    figure, axis = plt.subplots(figsize=(9, 6))
    colors = {0: "#4472C4", 1: "#ED7D31"}
    labels = {0: "Not highly rated", 1: "Highly rated"}
    for target in (0, 1):
        selected = classification_data[
            classification_data["is_highly_rated"] == target
        ]
        axis.scatter(
            selected["log_owners_midpoint"],
            selected["log_avg_playtime"],
            s=14,
            alpha=0.28,
            color=colors[target],
            label=f"{labels[target]} (n={len(selected):,})",
            edgecolors="none",
        )
    axis.set(
        title="Average Playtime vs. Estimated Owners",
        xlabel="log1p(owners midpoint)",
        ylabel="log1p(average playtime in minutes)",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    return figure, "playtime_vs_owners_by_reception.png"


def create_visualizations():
    for input_path in (FULL_DATA_PATH, CLASSIFICATION_DATA_PATH):
        if not input_path.is_file():
            raise FileNotFoundError(f"Required feature dataset not found: {input_path}")

    full_data = pd.read_csv(FULL_DATA_PATH)
    classification_data = pd.read_csv(CLASSIFICATION_DATA_PATH)
    required_full_columns = {"price", "log_price"}
    required_classification_columns = {
        "release_year",
        "is_highly_rated",
        "log_owners_midpoint",
        "log_avg_playtime",
    }
    if missing := required_full_columns.difference(full_data.columns):
        raise ValueError(f"Full feature data is missing columns: {sorted(missing)}")
    if missing := required_classification_columns.difference(
        classification_data.columns
    ):
        raise ValueError(
            f"Classification data is missing columns: {sorted(missing)}"
        )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plots = [
        save_price_distribution(full_data),
        save_reception_by_year(classification_data),
        save_reception_by_genre(classification_data),
        save_playtime_vs_owners(classification_data),
    ]
    for figure, filename in plots:
        output_path = FIGURES_DIR / filename
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved plot: {output_path}")
    print(f"Generated plots: {len(plots)}")


if __name__ == "__main__":
    create_visualizations()
