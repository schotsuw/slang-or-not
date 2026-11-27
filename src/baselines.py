"""
Baselines. B0: always predict the most frequent label.

Run from the project root (after python src/split_data.py):
    python src/baselines.py
"""

from sklearn.dummy import DummyClassifier

from load_data import load_split
from evaluate import evaluate


def get_labels(rows):
    """Pull the gold label ("slang" / "non-slang") out of each row."""
    return [row["register"] for row in rows]


def run_b0():
    # Read the saved split. The split is made once by split_data.py, not here.
    train_rows = load_split("train")
    dev_rows = load_split("dev")
    test_rows = load_split("test")

    # B0 ignores the text. sklearn still expects one feature per row,
    # so we give every row a dummy feature [0].
    model = DummyClassifier(strategy="most_frequent")
    model.fit([[0]] * len(train_rows), get_labels(train_rows))
    print("B0 always predicts:", model.predict([[0]])[0])

    evaluate("B0 dev", get_labels(dev_rows), model.predict([[0]] * len(dev_rows)))
    evaluate("B0 test", get_labels(test_rows), model.predict([[0]] * len(test_rows)))


if __name__ == "__main__":
    run_b0()
