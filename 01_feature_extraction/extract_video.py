"""
Extract OpenFace AU + Pose + Gaze features (mean+std+delta) from session archives.
Output: 03_processed_data/video/video_features_openface.csv
"""
import os, io, re, tarfile, glob
import numpy as np
import pandas as pd
from tqdm import tqdm

TAR_DIR  = "/content/drive/MyDrive/data"
OUT_PATH = "/content/drive/MyDrive/EDAIC_DISSERTATION/03_processed_data/video/video_features_openface.csv"

def get_pid(tar_path):
    m = re.match(r"(\d+)", os.path.basename(tar_path))
    return int(m.group(1)) if m else None

def read_csv_from_tar(tar_path, pattern):
    with tarfile.open(tar_path, "r:gz") as tf:
        match = next((m for m in tf.getnames()
                      if pattern.lower() in m.lower() and m.endswith(".csv")), None)
        if not match:
            return None
        content = tf.extractfile(match).read()
        return pd.read_csv(io.BytesIO(content), engine="python", on_bad_lines="skip")

def main():
    tar_files = sorted(glob.glob(f"{TAR_DIR}/*.tar.gz"))
    rows = []
    for tar_path in tqdm(tar_files):
        pid = get_pid(tar_path)
        if pid is None:
            continue
        df = read_csv_from_tar(tar_path, "bovw_openface")
        if df is None:
            df = read_csv_from_tar(tar_path, "openface")
        if df is None:
            continue
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] == 0:
            continue
        means  = numeric.mean().values
        stds   = numeric.std().fillna(0).values
        deltas = numeric.diff().abs().mean().fillna(0).values
        feat   = np.concatenate([means, stds, deltas])
        rows.append({"Participant_ID": pid, **{f"video_feat_{i}": v for i, v in enumerate(feat)}})
    pd.DataFrame(rows).set_index("Participant_ID").to_csv(OUT_PATH)
    print(f"Saved {len(rows)} rows to {OUT_PATH}")

if __name__ == "__main__":
    main()
