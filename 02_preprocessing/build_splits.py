"""
Build train/dev/test split CSVs from the detailed labels file.
Uses official DAIC-WOZ splits if split column present, else 70/15/15 stratified.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

LABELS_PATH = "labels/detailed_lables.csv"
SEED = 42

def main():
    df = pd.read_csv(LABELS_PATH)
    if "Participant_ID" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Participant_ID"})

    if "split" in df.columns:
        for split in ["train", "dev", "test"]:
            out = df[df["split"] == split][["Participant_ID"]].reset_index(drop=True)
            out.to_csv(f"labels/{split}_split.csv", index=False)
            print(f"  {split}: {len(out)} participants")
    else:
        dep_col = next(c for c in df.columns if "binary" in c.lower() or "label" in c.lower())
        tr, tmp = train_test_split(df, test_size=0.30, stratify=df[dep_col], random_state=SEED)
        dv, te  = train_test_split(tmp, test_size=0.50, stratify=tmp[dep_col], random_state=SEED)
        for split, data in [("train",tr),("dev",dv),("test",te)]:
            data[["Participant_ID"]].to_csv(f"labels/{split}_split.csv", index=False)
            print(f"  {split}: {len(data)} participants")

if __name__ == "__main__":
    main()
