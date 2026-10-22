"""Plumbing-test submission: predicts 0.5 for every label (scores exactly 0.5 macro AUC).

Run inside a Kaggle notebook attached to the competition, with Internet off.
"""
import pandas as pd

DATA = "/kaggle/input/competitions/rsna-knee-abnormality-detection"
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

test = pd.read_csv(f"{DATA}/test.csv")
sub = test[["StudyInstanceUID"]].copy()
for c in LABELS:
    sub[c] = 0.5
sub.to_csv("submission.csv", index=False)
print(sub.shape)
