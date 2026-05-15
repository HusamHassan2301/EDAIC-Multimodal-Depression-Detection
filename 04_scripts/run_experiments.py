"""
Run all multimodal depression detection experiments.
Unimodal, bimodal, trimodal, and all-4 early/late fusion.
Results saved to results/experiment_results.csv
"""
import os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import get_models
from src.preprocessing import prepare
from src.evaluate import best_threshold, evaluate
from src.data import load_features, get_split_masks, clean_features

SEED = 42
np.random.seed(SEED)

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS   = os.path.join(BASE, "labels", "detailed_lables.csv")
RESULTS  = os.path.join(BASE, "results", "experiment_results.csv")

PATHS = {
    "text":  os.path.join(BASE, "03_processed_data/text/text_features_distilbert.csv"),
    "audio": os.path.join(BASE, "03_processed_data/audio/audio_features_egemaps.csv"),
    "image": os.path.join(BASE, "03_processed_data/image/image_features_densenet201.csv"),
    "video": os.path.join(BASE, "03_processed_data/video/video_features_openface.csv"),
}

def main():
    modality_dfs, df_labels, common_ids = load_features(PATHS, LABELS)
    tr_mask, dv_mask, te_mask = get_split_masks(df_labels)

    y_all = df_labels["Depression_label"].astype(int).values
    y_tr, y_dv, y_te = y_all[tr_mask], y_all[dv_mask], y_all[te_mask]

    Xtr, Xdv, Xte = {}, {}, {}
    for name in ["text","audio","image","video"]:
        X = clean_features(modality_dfs[name].values.astype("float32"))
        Xtr[name], Xdv[name], Xte[name] = X[tr_mask], X[dv_mask], X[te_mask]

    results = []

    def run_setting(setting, mods, fusion="early"):
        print(f"  [{setting}]")
        for model_name, clf in get_models().items():
            try:
                if fusion == "early":
                    Xr = np.hstack([Xtr[m] for m in mods])
                    Xd = np.hstack([Xdv[m] for m in mods])
                    Xe = np.hstack([Xte[m] for m in mods])
                    Xr, yr, Xd, Xe = prepare(Xr, y_tr, Xd, Xe)
                    clf.fit(Xr, yr)
                    dv_p = clf.predict_proba(Xd)[:, 1]
                    te_p = clf.predict_proba(Xe)[:, 1]
                else:
                    dv_list, te_list = [], []
                    for m in mods:
                        Xr, yr, Xd, Xe = prepare(Xtr[m], y_tr, Xdv[m], Xte[m])
                        cm = get_models()[model_name]
                        cm.fit(Xr, yr)
                        dv_list.append(cm.predict_proba(Xd)[:, 1])
                        te_list.append(cm.predict_proba(Xe)[:, 1])
                    dv_p = np.stack(dv_list).mean(axis=0)
                    te_p = np.stack(te_list).mean(axis=0)
                thr = best_threshold(y_dv, dv_p)
                row = evaluate(y_te, te_p, thr, setting, model_name)
                results.append(row)
                print(f"    {model_name:<5} F1={row['F1']:.4f}  AUC={row['AUC']:.4f}")
            except Exception as e:
                print(f"    {model_name:<5} ERROR: {e}")

    print("=" * 60)
    print("RUNNING EXPERIMENTS")
    print("=" * 60)

    run_setting("text_only",   ["text"])
    run_setting("audio_only",  ["audio"])
    run_setting("image_only",  ["image"])
    run_setting("video_only",  ["video"])
    run_setting("text+audio_early",  ["text","audio"])
    run_setting("text+image_early",  ["text","image"])
    run_setting("text+video_early",  ["text","video"])
    run_setting("audio+image_early", ["audio","image"])
    run_setting("audio+video_early", ["audio","video"])
    run_setting("image+video_early", ["image","video"])
    run_setting("text+audio+image_early", ["text","audio","image"])
    run_setting("text+audio+video_early", ["text","audio","video"])
    run_setting("text+image+video_early", ["text","image","video"])
    run_setting("audio+image+video_early",["audio","image","video"])
    run_setting("all_4_early_fusion", ["text","audio","image","video"], fusion="early")
    run_setting("all_4_late_fusion",  ["text","audio","image","video"], fusion="late")

    df_results = pd.DataFrame(results).sort_values("F1", ascending=False)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    df_results.to_csv(RESULTS, index=False)
    print(f"\nResults saved to {RESULTS}")
    print(df_results.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
