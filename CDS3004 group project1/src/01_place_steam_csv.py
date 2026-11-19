import shutil
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SOURCE_PATH = ROOT_DIR / "steam.csv"
DESTINATION_PATH = ROOT_DIR / "data" / "raw" / "steam.csv"


def place_steam_csv():
    if not SOURCE_PATH.is_file():
        raise FileNotFoundError(f"Steam CSV source not found: {SOURCE_PATH}")

    DESTINATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_PATH, DESTINATION_PATH)
    print(f"Copied {SOURCE_PATH} to {DESTINATION_PATH}")


if __name__ == "__main__":
    place_steam_csv()
