import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def load_features(paths: dict, labels_path: str):
    df_labels = pd.read_csv(labels_path)
    if "Participant_ID" not in df_labels.columns:
        df_labels = df_labels.rename(columns={df_labels.columns[0]: "Participant_ID"})
    df_labels["Participant_ID"] = df_labels["Participant_ID"].astype(int)

    modality_dfs = {}
    for name, path in paths.items():
        df = pd.read_csv(path, index_col=0)
        df.index = df.index.astype(int)
        modality_dfs[name] = df

    common_ids = set(df_labels["Participant_ID"])
    for df in modality_dfs.values():
        common_ids &= set(df.index)
    common_ids = sorted(common_ids)

    df_labels = df_labels[df_labels["Participant_ID"].isin(common_ids)].set_index("Participant_ID")
    for name in modality_dfs:
        modality_dfs[name] = modality_dfs[name].loc[common_ids]

    return modality_dfs, df_labels, common_ids


def get_split_masks(df_labels, split_col="split"):
    splits = df_labels[split_col].values
    tr = np.array([s == "train" for s in splits])
    dv = np.array([s == "dev"   for s in splits])
    te = np.array([s == "test"  for s in splits])
    return tr, dv, te


def clean_features(X):
    col_means = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    return np.where(np.isfinite(X), X, col_means).astype(np.float32)
