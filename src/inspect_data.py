"""
PLAN.md section 3.1: look at the SlangTrack data before writing any model code.

Run from the project root:  python src/inspect_data.py
"""

import csv
import random
import re
from collections import Counter
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "raw" / "SlangTrack_ST_SlangDetection_public.csv"


def main():
    random.seed(42)

    # Read the CSV into a list of dicts (one dict per row: column name -> value).
    with open(DATA_FILE, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    print("Columns:", list(rows[0].keys()))
    print("\nFirst 5 rows (text cut to 80 chars):")
    for row in rows[:5]:
        print({**row, "text": row["text"][:80]})

    print("\nTotal rows:", len(rows))

    # 'register' holds the slang / non-slang label, 'word' holds the target word.
    print("\nLabel counts:", Counter(row["register"] for row in rows))
    print("\nRows per (word, label):")
    for (word, label), n in sorted(Counter((r["word"], r["register"]) for r in rows).items()):
        print(f"  {word:<10} {label:<10} {n}")

    print("\nSplit column present:", any(c.lower() == "split" for c in rows[0]))

    # Many rows have no text (tweets are given only as a tweet_id).
    with_text = [row for row in rows if row["text"].strip()]
    print(f"\nRows with empty text: {len(rows) - len(with_text)}")
    print(f"Rows with text: {len(with_text)}, labels: {Counter(r['register'] for r in with_text)}")

    lengths = [len(row["text"].split()) for row in with_text]
    print(f"Text length in words (non-empty rows): min {min(lengths)}, "
          f"mean {sum(lengths) / len(lengths):.1f}, max {max(lengths)}")

    # 10 random rows per word. Only a short window around the target is shown,
    # to avoid printing whole offensive passages (see the content warning in PLAN.md).
    for word in sorted(set(row["word"] for row in rows)):
        print(f"\n--- {word} ---")
        word_rows = [row for row in with_text if row["word"] == word]
        for row in random.sample(word_rows, 10):
            text = row["text"]
            match = re.search(r"#?\b" + re.escape(word) + r"\w*", text, flags=re.IGNORECASE)
            if match:
                text = text[max(0, match.start() - 50):match.end() + 50]
            print(f"  {row['register']:<10} {text[:120]}")


if __name__ == "__main__":
    main()
