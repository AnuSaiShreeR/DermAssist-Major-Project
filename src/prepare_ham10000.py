from pathlib import Path
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split


RAW_DIR = Path("data/raw/ham10000")
OUT_DIR = Path("data/processed/ham10000")

METADATA_FILE = RAW_DIR / "HAM10000_metadata.csv"
IMAGE_DIRS = [
    RAW_DIR / "HAM10000_images_part_1",
    RAW_DIR / "HAM10000_images_part_2",
]

RANDOM_STATE = 42


def find_image_path(image_id: str) -> Path:
    filename = f"{image_id}.jpg"
    for image_dir in IMAGE_DIRS:
        candidate = image_dir / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find image: {filename}")


def copy_split(df: pd.DataFrame, split_name: str) -> None:
    for _, row in df.iterrows():
        image_id = row["image_id"]
        label = row["dx"]

        src = find_image_path(image_id)
        dst_dir = OUT_DIR / split_name / label
        dst_dir.mkdir(parents=True, exist_ok=True)

        dst = dst_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)


def main() -> None:
    df = pd.read_csv(METADATA_FILE)

    # Keep only image_id and diagnosis label.
    df = df[["image_id", "dx"]].dropna()

    print("Class distribution:")
    print(df["dx"].value_counts())

    # Stratified split preserves class distribution across splits.
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["dx"],
        random_state=RANDOM_STATE,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["dx"],
        random_state=RANDOM_STATE,
    )

    print(f"Train: {len(train_df)}")
    print(f"Val:   {len(val_df)}")
    print(f"Test:  {len(test_df)}")

    copy_split(train_df, "train")
    copy_split(val_df, "val")
    copy_split(test_df, "test")

    print(f"Prepared dataset at: {OUT_DIR}")


if __name__ == "__main__":
    main()
