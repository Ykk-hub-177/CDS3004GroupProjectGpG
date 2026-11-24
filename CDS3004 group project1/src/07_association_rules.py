from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT_DIR / "data" / "processed" / "steam_features_clustered.csv"
OUTPUT_DIR = ROOT_DIR / "outputs"
Plot_OUTPUT_DIR = OUTPUT_DIR / "figures"
RULES_OUTPUT_PATH = OUTPUT_DIR / "association_rules.csv"
PLOT_OUTPUT_PATH = Plot_OUTPUT_DIR / "association_rules_summary.png"

MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.50
MIN_LIFT = 1.20
MAX_ITEMSET_SIZE = 3
TOP_RULE_COUNT = 15


def create_transaction_matrix(data):
    transaction = pd.DataFrame(index=data.index)

    genre_columns = sorted(
        (column for column in data.columns if column.startswith("genre_")),
        key=lambda column: column,
    )
    tag_columns = sorted(
        (column for column in data.columns if column.startswith("tag_")),
        key=lambda column: column,
    )
    for column in genre_columns:
        transaction[column] = pd.to_numeric(data[column], errors="raise").eq(1)
    for column in tag_columns:
        transaction[column] = pd.to_numeric(data[column], errors="raise").eq(1)

    for cluster_id in sorted(pd.to_numeric(data["cluster_id"], errors="raise").dropna().unique()):
        transaction[f"cluster_{int(cluster_id)}"] = (
            pd.to_numeric(data["cluster_id"], errors="raise") == int(cluster_id)
        )

    transaction["platform_count_1"] = data["platform_count"].eq(1)
    transaction["platform_count_2"] = data["platform_count"].eq(2)
    transaction["platform_count_3"] = data["platform_count"].eq(3)

    price = pd.to_numeric(data["price"], errors="raise")
    transaction["price_free"] = price.eq(0)
    transaction["price_budget"] = price.gt(0) & price.le(3.99)
    transaction["price_mid"] = price.gt(3.99) & price.le(14.99)
    transaction["price_premium"] = price.gt(14.99)

    playtime = pd.to_numeric(data["average_playtime"], errors="raise")
    transaction["playtime_zero"] = playtime.eq(0)
    transaction["playtime_short"] = playtime.gt(0) & playtime.le(60)
    transaction["playtime_medium"] = playtime.gt(60) & playtime.le(300)
    transaction["playtime_high"] = playtime.gt(300)

    owners = pd.to_numeric(data["owners_midpoint"], errors="raise")
    transaction["owners_low"] = owners.le(10_000)
    transaction["owners_medium"] = owners.gt(10_000) & owners.le(150_000)
    transaction["owners_high"] = owners.gt(150_000) & owners.le(1_500_000)
    transaction["owners_very_high"] = owners.gt(1_500_000)

    positive_ratio = pd.to_numeric(data["positive_ratio"], errors="raise")
    total_reviews = pd.to_numeric(data["total_reviews"], errors="raise")
    transaction["reception_high"] = (
        positive_ratio.ge(0.80) & total_reviews.ge(20)
    )

    transaction = transaction.astype(bool)
    if transaction.isna().any().any():
        raise ValueError("Boolean transaction matrix contains missing values")
    if not transaction.shape[1]:
        raise ValueError("No association-rule items were generated")
    return transaction


def remove_reversed_duplicates(rules):
    if rules.empty:
        return rules

    rules = rules.copy()
    rules["_rule_key"] = rules.apply(
        lambda row: frozenset(
            set(row["antecedents"]).union(set(row["consequents"]))
        ),
        axis=1,
    )
    rules = rules.drop_duplicates(subset="_rule_key", keep="first")
    rules = rules.drop(columns="_rule_key")
    return rules


def format_itemset(itemset):
    return " + ".join(sorted(itemset))


def save_summary_plot(rules):
    top_rules = rules.head(TOP_RULE_COUNT).sort_values("lift", ascending=True)
    labels = [
        f"{format_itemset(row.antecedents)} -> {format_itemset(row.consequents)}"
        for row in top_rules.itertuples(index=False)
    ]
    figure_height = max(6.0, 0.48 * len(top_rules))
    figure, axis = plt.subplots(figsize=(13, figure_height))
    bars = axis.barh(
        labels,
        top_rules["lift"],
        color="#4472C4",
        edgecolor="white",
    )
    axis.axvline(1.0, color="#C0504D", linestyle="--", linewidth=1.5, label="Lift = 1")
    axis.set(
        title=f"Top {len(top_rules)} Association Rules by Lift",
        xlabel="Lift",
        ylabel="Rule",
        xlim=(0, max(24.0, float(top_rules["lift"].max()) * 1.20)),
    )
    axis.grid(axis="x", alpha=0.25)
    for bar, row in zip(bars, top_rules.itertuples(index=False)):
        axis.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"conf={row.confidence:.3f}, supp={row.support:.3f}",
            va="center",
            fontsize=8,
        )
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(PLOT_OUTPUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)


def generate_association_rules():
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Clustered Steam data not found: {INPUT_PATH}")

    data = pd.read_csv(INPUT_PATH)
    required_columns = {
        "cluster_id",
        "platform_count",
        "price",
        "average_playtime",
        "owners_midpoint",
        "positive_ratio",
        "total_reviews",
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Clustered data is missing columns: {sorted(missing_columns)}")

    transaction = create_transaction_matrix(data)
    frequent_itemsets = apriori(
        transaction,
        min_support=MIN_SUPPORT,
        use_colnames=True,
        max_len=MAX_ITEMSET_SIZE,
        low_memory=True,
    )
    if frequent_itemsets.empty:
        raise ValueError(
            f"No frequent itemsets found with min_support={MIN_SUPPORT}"
        )

    rules = association_rules(
        frequent_itemsets,
        metric="lift",
        min_threshold=1.0,
        num_itemsets=len(frequent_itemsets),
    )
    rules = rules[
        (rules["support"] >= MIN_SUPPORT)
        & (rules["confidence"] >= MIN_CONFIDENCE)
        & (rules["lift"] >= MIN_LIFT)
    ].copy()
    rules = remove_reversed_duplicates(rules)
    rules = rules.sort_values(
        ["lift", "confidence", "support"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    rules["antecedents_text"] = rules["antecedents"].map(format_itemset)
    rules["consequents_text"] = rules["consequents"].map(format_itemset)
    rules["antecedent_count"] = rules["antecedents"].map(len)
    rules["consequent_count"] = rules["consequents"].map(len)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rules_output = rules[
        [
            "antecedents_text",
            "consequents_text",
            "antecedent_count",
            "consequent_count",
            "support",
            "confidence",
            "lift",
            "representativity",
            "leverage",
            "conviction",
            "zhangs_metric",
        ]
        if "representativity" in rules.columns
        else [
            "antecedents_text",
            "consequents_text",
            "antecedent_count",
            "consequent_count",
            "support",
            "confidence",
            "lift",
            "leverage",
            "conviction",
            "zhangs_metric",
        ]
    ].copy()
    rules_output.to_csv(RULES_OUTPUT_PATH, index=False)
    save_summary_plot(rules)

    print(f"Input rows: {len(data):,}")
    print(f"Boolean items: {transaction.shape[1]}")
    print(f"Frequent itemsets: {len(frequent_itemsets):,}")
    print(
        f"Filtered rules (support >= {MIN_SUPPORT:.3f}, "
        f"confidence >= {MIN_CONFIDENCE:.2f}, lift >= {MIN_LIFT:.2f}): "
        f"{len(rules):,}"
    )
    print(f"Saved rules table: {RULES_OUTPUT_PATH}")
    print(f"Saved summary plot: {PLOT_OUTPUT_PATH}")

    print("\nTop rules:")
    for rank, row in enumerate(rules.head(TOP_RULE_COUNT).itertuples(index=False), start=1):
        print(
            f"{rank}. {row.antecedents_text} -> {row.consequents_text} "
            f"| support={row.support:.4f}, confidence={row.confidence:.4f}, "
            f"lift={row.lift:.4f}"
        )


if __name__ == "__main__":
    generate_association_rules()
