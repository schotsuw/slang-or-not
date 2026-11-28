"""
Baselines.
    B0: always predict the most frequent label.
    B1: TF-IDF (single words + word pairs) and logistic regression, tuned on dev.

Run from the project root (after python src/split_data.py):
    python src/baselines.py
"""

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer  # turns texts into TF-IDF numbers
from sklearn.linear_model import LogisticRegression          # the classifier
from sklearn.metrics import f1_score                         # to score each setting on dev

from load_data import load_split
from evaluate import evaluate


def get_labels(rows):
    """Pull the gold label ("slang" / "non-slang") out of each row."""
    return [row["register"] for row in rows]


def get_texts(rows):
    """Pull the passage text out of each row."""
    return [row["text"] for row in rows]  # a list with one string per row


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


def run_b1():
    """B1: TF-IDF (single words + word pairs) and logistic regression, tuned on dev."""

    # --- Load the data ---
    train_rows = load_split("train")  # 3,091 rows: the model learns from these
    dev_rows = load_split("dev")      # 662 rows: used to choose the settings
    test_rows = load_split("test")    # 663 rows: used once, for the final score

    train_texts = get_texts(train_rows)  # the passages, as a list of strings
    dev_texts = get_texts(dev_rows)
    test_texts = get_texts(test_rows)

    train_labels = get_labels(train_rows)  # "slang" / "non-slang" for each passage
    dev_labels = get_labels(dev_rows)
    test_labels = get_labels(test_rows)

    # --- TF-IDF: turn each passage into a vector of word weights ---
    # ngram_range=(1, 2): use single words ("salty") and word pairs ("so salty").
    # lowercase=True: "Salty" and "salty" count as the same word.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)

    # fit_transform: learn the vocabulary and word weights from TRAIN only,
    # then turn the training texts into vectors.
    train_x = vectorizer.fit_transform(train_texts)

    # transform: turn dev and test into vectors using the TRAIN vocabulary.
    # We never fit on dev or test, so the model can't peek at them.
    dev_x = vectorizer.transform(dev_texts)
    test_x = vectorizer.transform(test_texts)

    # --- Tune on dev: try every combination of settings, keep the best ---
    best_f1 = -1           # best dev macro-F1 so far (-1 so the first try always wins)
    best_model = None      # the model that got best_f1
    best_settings = None   # its settings, e.g. (1, "balanced")

    print("\nB1 tuning on dev:")
    for c in [0.1, 1, 10]:  # C: how strongly to regularize (smaller = simpler model)
        for class_weight in [None, "balanced"]:  # "balanced" gives the rarer slang class more weight
            model = LogisticRegression(
                C=c,
                class_weight=class_weight,
                max_iter=1000,    # allow enough steps for training to finish
                random_state=42,  # same result every run
            )
            model.fit(train_x, train_labels)  # train on the training set

            dev_predictions = model.predict(dev_x)  # predict the dev set
            dev_f1 = f1_score(dev_labels, dev_predictions,
                              average="macro", zero_division=0)  # our main metric
            print(f"C={c:<4} class_weight={str(class_weight):<8} dev macro-F1={dev_f1:.3f}")

            if dev_f1 > best_f1:  # better than the best so far? remember it
                best_f1 = dev_f1
                best_model = model
                best_settings = (c, class_weight)

    print(f"\nBest on dev: C={best_settings[0]}, class_weight={best_settings[1]} "
          f"(dev macro-F1 {best_f1:.3f})")

    # --- Final scores with the best settings ---
    evaluate("B1 dev", dev_labels, best_model.predict(dev_x))
    evaluate("B1 test", test_labels, best_model.predict(test_x))  # test is used only here, once


if __name__ == "__main__":
    run_b0()
    run_b1()
