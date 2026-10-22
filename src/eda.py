"""Exploratory analysis of the competition CSVs (labels, series metadata, report lengths).

Usage:  python src/eda.py /path/to/competition/data
Reads train.csv, train_series.csv, test_series.csv. Prints summaries only; writes nothing.
"""
import sys
import pandas as pd

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def main(root):
    train = pd.read_csv(f"{root}/train.csv")
    series = pd.read_csv(f"{root}/train_series.csv")

    labeled = train[LABELS].notna().all(axis=1)
    print(f"studies: {len(train)} | fully labeled: {labeled.sum()} "
          f"| unlabeled: {(~labeled).sum()}")

    print("\npositives / negatives among labeled studies:")
    for c in LABELS:
        vc = train.loc[labeled, c].value_counts()
        print(f"  {c:18s} pos={int(vc.get(1.0, 0)):3d}  neg={int(vc.get(0.0, 0)):3d}")

    print("\nlabel correlation (labeled studies):")
    print(train.loc[labeled, LABELS].corr().round(2).to_string())

    per_study = series.groupby("StudyInstanceUID").size()
    print(f"\nseries per study: min={per_study.min()} median={per_study.median():.0f} "
          f"max={per_study.max()}")
    print("\nseries by plane / fluid-sensitive / fat-suppressed:")
    print(series.groupby(["Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"]).size())

    words = train["Report"].str.split().str.len()
    print(f"\nreport length (words): median={words.median():.0f} max={words.max()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "/kaggle/input/competitions/rsna-knee-abnormality-detection")
