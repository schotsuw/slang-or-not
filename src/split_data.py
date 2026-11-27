"""
Create the fixed train / dev / test split (70 / 15 / 15).

Run once from the project root:
    python src/split_data.py
"""

import csv

from sklearn.model_selection import train_test_split

from load_data import PROJECT_ROOT, SPLIT_DIR, load_rows


def split_rows(rows):
    """Split 70 / 15 / 15, stratified by (target word, label), seed 42."""
    # One group per word + label, e.g. "Salty_slang", so every split
    # gets a similar mix of words and labels.
    strata = []
    for row in rows:
        strata.append(row["word"] + "_" + row["register"])

    # First split: 70% train, 30% left over.
    train_rows, remaining_rows, _, remaining_strata = train_test_split(
        rows, strata, test_size=0.30, random_state=42, stratify=strata
    )

    # Second split: the left-over 30% in half, giving 15% dev and 15% test.
    dev_rows, test_rows = train_test_split(
        remaining_rows, test_size=0.50, random_state=42, stratify=remaining_strata
    )
    return train_rows, dev_rows, test_rows


def save_split(name, rows):
    """Write one split to data/train_dev_test/<name>.csv."""
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    path = SPLIT_DIR / f"{name}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path.relative_to(PROJECT_ROOT)}")


def main():
    rows = load_rows()
    print("Rows with text:", len(rows))  # expect 4416
    train_rows, dev_rows, test_rows = split_rows(rows)
    save_split("train", train_rows)  # expect 3091
    save_split("dev", dev_rows)      # expect 662
    save_split("test", test_rows)    # expect 663


if __name__ == "__main__":
    main()
