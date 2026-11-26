import csv
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from load_data import load_csv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "SlangTrack_ST_SlangDetection_public.csv"
SPLIT_DIR = PROJECT_ROOT / "data" / "train_dev_test"

LABELS = ["non-slang", "slang"]


def create_splits():
    # Keep only CCOHA examples that have text.
    rows = []
    for row in load_csv(RAW_FILE):
        if row.get("text", "").strip():
            rows.append(row)

    # Stratify by the combination of target word and binary label.
    strata = []
    for row in rows:
        group = f"{row['word'].lower()}_{row['register'].lower()}"
        strata.append(group)

    # 70% train, 30% remaining.
    train_rows, remaining_rows, _, remaining_strata = train_test_split(
        rows,
        strata,
        test_size=0.30,
        random_state=42,
        stratify=strata,
    )

    # Split the remainder evenly into dev and test.
    # Balance by binary label because some word/label groups are too small.
    remaining_labels = []
    for row in remaining_rows:
        remaining_labels.append(row["register"])

    dev_rows, test_rows = train_test_split(
        remaining_rows,
        test_size=0.50,
        random_state=42,
        stratify=remaining_labels,
    )

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys())

    splits = [
        ("train.csv", train_rows),
        ("dev.csv", dev_rows),
        ("test.csv", test_rows),
    ]

    for filename, split_rows in splits:
        output_path = SPLIT_DIR / filename
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(split_rows)

        print(f"Wrote {len(split_rows)} rows to {output_path}")

    return train_rows, dev_rows, test_rows


def evaluate(split_name, rows, predictions):
    gold_labels = []
    for row in rows:
        gold_labels.append(row["register"])

    print(f"\n{split_name} results (n={len(rows)})")
    print(f"Accuracy: {accuracy_score(gold_labels, predictions):.3f}")
    print(classification_report(
        gold_labels,
        predictions,
        labels=LABELS,
        zero_division=0,
        digits=3,
    ))
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(gold_labels, predictions, labels=LABELS))


def run_b0():
    train_rows, dev_rows, test_rows = create_splits()

    # The feature values are placeholders; B0 ignores the text.
    model = DummyClassifier(strategy="most_frequent")

    train_x = []
    train_labels = []
    for row in train_rows:
        train_x.append([0])
        train_labels.append(row["register"])

    model.fit(train_x, train_labels)

    print(f"\nB0 always predicts: {model.classes_[0]}")

    dev_x = []
    for row in dev_rows:
        dev_x.append([0])
    evaluate("Dev", dev_rows, model.predict(dev_x))

    test_x = []
    for row in test_rows:
        test_x.append([0])
    evaluate("Test", test_rows, model.predict(test_x))


if __name__ == "__main__":
    run_b0()