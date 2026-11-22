import csv

def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

rows = load_csv("dataset-slangtrack/SlangTrack_ST_SlangDetection_public.csv")
