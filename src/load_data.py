"""
Loading the SlangTrack data and the saved train / dev / test split.

Other scripts import from here, e.g.:
    from load_data import load_split
"""

import csv
from pathlib import Path

# Paths are built from this file's location, so they work from any folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "SlangTrack_ST_SlangDetection_public.csv"
SPLIT_DIR = PROJECT_ROOT / "data" / "train_dev_test"


def load_csv(path):
    """Read a CSV file into a list of dicts (one dict per row)."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_rows():
    """
    Return the rows of the raw SlangTrack CSV that we use: those with non-empty text.

    The public CSV has 12,712 rows, but the 8,296 Twitter rows have no text
    (the release gives only tweet IDs). Only the 4,416 CCOHA passages
    (books and magazines, 1980-2010) have text, so those are our dataset.

    Each row is a dict. The columns we use are:
        "text"      the passage
        "word"      the target word, e.g. "Salty"
        "register"  the label: "slang" or "non-slang"
    """
    rows = []
    for row in load_csv(RAW_FILE):
        if row["text"].strip():
            rows.append(row)
    return rows


def load_split(name):
    """Read a saved split: load_split("train"), load_split("dev") or load_split("test")."""
    return load_csv(SPLIT_DIR / f"{name}.csv")
