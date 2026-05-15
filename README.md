# EDAIC Multimodal Depression Detection

A 4-modality depression detection system built on the DAIC-WOZ dataset.

## Modalities
| Modality | Source | Features |
|---|---|---|
| Text | DistilBERT on interview transcripts | 768 dim |
| Audio | OpenSMILE eGeMAPS (mean+std+min+max) | 92 dim |
| Image | DenseNet-201 visual features | 1921 dim |
| Video | OpenFace AUs + Pose + Gaze (mean+std+delta) | 147 dim |

## Best Results (Test Split)
| Setting | Model | F1 | AUC |
|---|---|---|---|
| all_4_late_fusion | MLP | 0.6415 | 0.8250 |
| text+audio+video_early | RF | 0.6400 | 0.7081 |
| all_4_late_fusion | SVM | 0.6222 | 0.8009 |

## Key Findings
- Late fusion outperforms early fusion for 4-modality depression detection
- Video (OpenFace AUs) is the strongest single modality
- Best system: all_4_late_fusion MLP — F1=0.6415, AUC=0.8250

## Dataset
DAIC-WOZ (Distress Analysis Interview Corpus) — Gratch et al. (2014).
Access requires signing a data agreement at USC ICT.

## Project Structure
```
EDAIC-Multimodal-Depression-Detection/
├── 01_feature_extraction/    # Feature extraction scripts
├── 02_preprocessing/         # Split generation
├── 03_processed_data/        # Extracted feature CSVs
├── 04_scripts/               # Experiment runner
├── src/                      # Core package
├── labels/                   # Labels and splits
├── results/                  # Experiment results
└── requirements.txt
```

## Usage
Open EDAIC_Multimodal_Pipeline_v2.ipynb in Google Colab and run cells top to bottom.

## Citation
Gratch, J. et al. (2014). The Distress Analysis Interview Corpus of Human and
Computer Interviews. LREC 2014.
