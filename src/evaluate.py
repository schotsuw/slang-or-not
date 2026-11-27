"""Shared evaluation used by every model (B0, B1, M1, M2)."""

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

LABELS = ["non-slang", "slang"]


def evaluate(name, gold_labels, predicted_labels):
    """Print the scores for one model on one split, and return the main ones."""
    accuracy = accuracy_score(gold_labels, predicted_labels)
    macro_f1 = f1_score(gold_labels, predicted_labels, labels=LABELS,
                        average="macro", zero_division=0)

    print(f"\n=== {name} (n={len(gold_labels)}) ===")
    print(f"Macro-F1: {macro_f1:.3f}   (main metric)")
    print(f"Accuracy: {accuracy:.3f}")
    # Precision / recall / F1 for each class, slang and non-slang.
    print(classification_report(gold_labels, predicted_labels, labels=LABELS,
                                zero_division=0, digits=3))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(gold_labels, predicted_labels, labels=LABELS))

    return {"accuracy": accuracy, "macro_f1": macro_f1}
