"""
Extract DistilBERT CLS embeddings from participant-only interview transcripts.
Reads {pid}_Transcript.csv from each .tar.gz session archive.
Output: 03_processed_data/text/text_features_distilbert.csv
"""
import os, io, re, tarfile, glob
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

TAR_DIR  = "/content/drive/MyDrive/data"
OUT_PATH = "/content/drive/MyDrive/EDAIC_DISSERTATION/03_processed_data/text/text_features_distilbert.csv"

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

def extract_participant_text(df):
    text_col = next((c for c in df.columns if "text" in c.lower()), df.columns[-1])
    conf_col = "Confidence" if "Confidence" in df.columns else None
    if conf_col:
        df = df[df[conf_col] > 0.5]
    texts = df[text_col].dropna().astype(str).tolist()
    texts = [t for t in texts if len(t.split()) > 2]
    return " ".join(texts).strip()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased").to(device).eval()
    tar_files = sorted(glob.glob(f"{TAR_DIR}/*.tar.gz"))
    rows = []
    for tar_path in tqdm(tar_files):
        pid = get_pid(tar_path)
        if pid is None:
            continue
        df_t = read_csv_from_tar(tar_path, "transcript")
        if df_t is None:
            continue
        text = extract_participant_text(df_t)
        if not text:
            continue
        enc = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            feat = model(**enc).last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        rows.append({"Participant_ID": pid, **{f"text_feat_{i}": v for i, v in enumerate(feat)}})
    pd.DataFrame(rows).set_index("Participant_ID").to_csv(OUT_PATH)
    print(f"Saved {len(rows)} rows to {OUT_PATH}")

if __name__ == "__main__":
    main()
