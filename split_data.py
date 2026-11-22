import csv
from pathlib import Path

from sklearn.model_selection import train_test_split

from load_data import load_csv


raw_path = "data/raw/SlangTrack_ST_SlangDetection_public.csv"
output_folder = Path("data/train_dev_test")

rows = load_csv(raw_path)

# Group rows by target word and label so each split keeps a similar mix.
groups = []
for row in rows:
    word = row["word"]
    label = row["true_label"]
    group = word + "_" + label
    groups.append(group)

# Split into 80% training data and 20% remaining data.
train_rows, remaining_rows = train_test_split(
    rows,
    test_size=0.2,
    random_state=42,
    stratify=groups,
)

# Split the remaining data equally into development and test sets.
# Some word/label groups are too small to split again, so balance by label.
remaining_labels = []
for row in remaining_rows:
    label = row["true_label"]
    remaining_labels.append(label)

dev_rows, test_rows = train_test_split(
    remaining_rows,
    test_size=0.5,
    random_state=42,
    stratify=remaining_labels,
)

output_folder.mkdir(parents=True, exist_ok=True)
columns = rows[0].keys()

splits = [
    ("train.csv", train_rows),
    ("dev.csv", dev_rows),
    ("test.csv", test_rows),
]

for filename, split_rows in splits:
    output_path = output_folder / filename
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(split_rows)

    print("Wrote", len(split_rows), "rows to", output_path)
