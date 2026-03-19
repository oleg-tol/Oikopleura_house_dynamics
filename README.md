# DeepLabCut behavioural analysis OIkopleura dioica house related behavior
Pipeline for processing DLC tracking data, computing locomotion metrics, and fitting a Gaussian HMM to characterise behavioural states with emphasis on the manually labelled house expansion periods.

## Modules

| File | Description |
|------|-------------|
| `Loading_data.py` | Load `*filtered.csv` DLC outputs, drop likelihood columns, merge experiments, resample, assign condition labels |
| `Locomotion_metrics.py` | Centre-of-mass trajectories, deadbanded velocity, body curvature, tangent angles, quirkiness, tail-beat frequency & amplitude, omega (body rotation), SVD path complexity, windowed features (tortuosity, MSD slope, path complexity), RDQ state classification, house-expansion tail-beat metrics |
| `Feature_selection.py` | Assemble per-window feature table, CV filter, Spearman correlation filter, effect-size diagnostics, per-individual robust normalisation, build analysis matrices for PCA/HMM |
| `PCA.py` | Feature coverage check, preprocessing (clip → log1p → robust z-score → StandardScaler), PCA fit on control animals, project all conditions, save coordinates & loadings |
| `HMM.py` | BIC scan (control animals only), final Gaussian HMM fit (all animals), Viterbi decode, dwell smoothing, rare-state pruning, occupancy & transition tables, full ethograms, HE transient panels, chord diagrams |

## Pipeline

```
DLC *filtered.csv
  → Loading_data.py           →  x_final.csv, y_final.csv
  → Locomotion_metrics.py     →  wide_tables/  (velocity, curvature, tail-beat,
                                  omega, SVD complexity, windowed metrics,
                                  RDQ state time-series, bout_summary,
                                  per_animal_feature_matrix)
  → Feature_selection.py      →  feature_selection/  (reports, recommended_features.txt)
                                  matrices/  (normalised matrices, all/active windows)
  → PCA                       →  pca/  (pca_coordinates_all.csv, pca_loadings.csv,
                                  pca_summary_per_animal.csv, coverage_report.csv)
  → HMM.py                    →  hmm/  (decoded windows, occupancy, transitions,
                                  state means, ethograms, chord diagrams)
```

## Key parameters

| Parameter | Value | Location |
|-----------|-------|----------|
| FPS / resample rate | 30 fps / 1:5 → 6 Hz | `Locomotion_metrics.py` |
| Pixel to µm | 11.56 | `Locomotion_metrics.py` |
| Window / stride | 3 s / 1 s | `Locomotion_metrics.py` |
| Likelihood threshold | 0.99 | `Locomotion_metrics.py` |
| Deadband | 3 × MAD (hysteresis 1.2 / 0.8) | `Locomotion_metrics.py` |
| RDQ quiescence threshold | 50 µm/s | `Locomotion_metrics.py` |
| SVD embedding window | 9 frames (1.5 s) | `Locomotion_metrics.py` |
| CV filter / Spearman threshold | 0.05 / 0.70 | `Feature_selection.py` |
| Normalisation | per-individual robust z-score (median/IQR) + StandardScaler on control | `Feature_selection.py`, `PCA.py` |
| PCA fit condition | control only, projected to all | `PCA.py` |
| HMM K (BIC scan) | 2–10, pick elbow | `HMM.py` |
| HMM covariance / min_covar | full / 0.01 | `HMM.py` |
| Dwell smoothing | median filter 3 windows | `HMM.py` |
| HE transient context | ±60 s | `HMM.py` |

## Requirements

```
pandas numpy scipy scikit-learn shapely
hmmlearn pycirclize
matplotlib seaborn
```

Edit the path variables at the top of each file before running.
