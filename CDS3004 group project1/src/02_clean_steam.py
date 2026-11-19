import csv
import math
import re
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "steam.csv"
ROOT_CSV_FALLBACK = ROOT_DIR / "steam.csv"
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "steam_cleaned.csv"
OWNER_RANGE_PATTERN = re.compile(r"^(\d+)-(\d+)$")

ADDED_COLUMNS = [
    "release_year",
    "owners_midpoint",
    "total_reviews",
    "positive_ratio",
    "log_price",
    "log_avg_playtime",
    "log_median_playtime",
    "log_achievements",
    "log_owners_midpoint",
]


def parse_nonnegative_number(value, column, row_number):
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid number in column {column!r}, CSV row {row_number}: {value!r}"
        ) from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(
            f"Expected a finite nonnegative number in column {column!r}, "
            f"CSV row {row_number}: {value!r}"
        )
    return number


def parse_owner_midpoint(value, row_number):
    match = OWNER_RANGE_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(
            f"Invalid owners range in CSV row {row_number}: {value!r}"
        )
    lower, upper = (int(bound) for bound in match.groups())
    if lower > upper:
        raise ValueError(
            f"Owners range lower bound exceeds upper bound in CSV row "
            f"{row_number}: {value!r}"
        )
    return (lower + upper) / 2


def resolve_input_path():
    if RAW_DATA_PATH.is_file():
        return RAW_DATA_PATH
    if ROOT_CSV_FALLBACK.is_file():
        return ROOT_CSV_FALLBACK
    raise FileNotFoundError(
        f"Could not find input CSV at {RAW_DATA_PATH} or {ROOT_CSV_FALLBACK}"
    )


def clean_steam_data():
    input_path = resolve_input_path()
    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {input_path}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Input CSV has duplicate column names: {input_path}")
        required_columns = {
            "release_date",
            "publisher",
            "owners",
            "positive_ratings",
            "negative_ratings",
            "price",
            "average_playtime",
            "median_playtime",
            "achievements",
        }
        missing_columns = required_columns.difference(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                f"Input CSV is missing required columns: {sorted(missing_columns)}"
            )

        input_columns = list(reader.fieldnames)
        output_columns = input_columns + [
            column for column in ADDED_COLUMNS if column not in input_columns
        ]
        rows = []
        for row_number, row in enumerate(reader, start=2):
            release_date = row["release_date"]
            try:
                parsed_date = datetime.strptime(release_date, "%Y-%m-%d")
            except ValueError as exc:
                raise ValueError(
                    f"Invalid release_date in CSV row {row_number}: "
                    f"{release_date!r}"
                ) from exc
            row["release_date"] = parsed_date.date().isoformat()
            row["release_year"] = str(parsed_date.year)

            if not row["publisher"].strip():
                row["publisher"] = "Unknown"

            owners_midpoint = parse_owner_midpoint(row["owners"], row_number)
            row["owners_midpoint"] = str(owners_midpoint)

            try:
                positive_ratings = int(row["positive_ratings"])
                negative_ratings = int(row["negative_ratings"])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid rating count in CSV row {row_number}"
                ) from exc
            if positive_ratings < 0 or negative_ratings < 0:
                raise ValueError(
                    f"Rating counts must be nonnegative in CSV row {row_number}"
                )
            total_reviews = positive_ratings + negative_ratings
            positive_ratio = (
                positive_ratings / total_reviews if total_reviews > 0 else 0.0
            )
            row["total_reviews"] = str(total_reviews)
            row["positive_ratio"] = str(positive_ratio)

            log_inputs = {
                "log_price": ("price", "price"),
                "log_avg_playtime": ("average_playtime", "average_playtime"),
                "log_median_playtime": ("median_playtime", "median_playtime"),
                "log_achievements": ("achievements", "achievements"),
            }
            for output_column, (input_column, validation_name) in log_inputs.items():
                value = parse_nonnegative_number(
                    row[input_column], validation_name, row_number
                )
                row[output_column] = str(math.log1p(value))
            row["log_owners_midpoint"] = str(math.log1p(owners_midpoint))
            rows.append(row)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=output_columns)
        writer.writeheader()
        writer.writerows(rows)

    missing_values = {
        column: sum(not row[column].strip() for row in rows)
        for column in output_columns
    }
    eligible_rows = [
        row for row in rows if int(row["total_reviews"]) >= 20
    ]
    positive_target_count = sum(
        float(row["positive_ratio"]) >= 0.80 for row in eligible_rows
    )
    negative_target_count = len(eligible_rows) - positive_target_count
    positive_threshold_all_rows = sum(
        float(row["positive_ratio"]) >= 0.80 for row in rows
    )

    print(f"Input: {input_path}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Shape: ({len(rows)}, {len(output_columns)})")
    print(f"Remaining missing values by column: {missing_values}")
    print(f"Rows with total_reviews >= 20: {len(eligible_rows)}")
    print(
        "Target counts among rows with total_reviews >= 20 "
        f"(positive_ratio >= 0.80): positive={positive_target_count}, "
        f"negative={negative_target_count}"
    )
    print(
        "Rows with positive_ratio >= 0.80 (all rows, including low-review): "
        f"{positive_threshold_all_rows}"
    )


if __name__ == "__main__":
    clean_steam_data()
