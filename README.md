# DeepLabCut behavioural analysis of *Oikopleura dioica* house expansion (control vs 6-OHDA)
Pipeline for cleaning DLC skeletons, computing 1 s window features, decoding movement modes, analysing what precedes house expansion, and embedding whole-video posture dynamics with region-level quantification.

## Modules

| File | Description |
|------|-------------|
| `common.py` | Paths, constants, skeleton cleaning, window features, frame posture, skeleton normalisation, statistics and figure helpers |
| `01_inventory.py` | Find every DLC csv, deduplicate copies across sources, parse expansion / in-house / escape / abnormal-event annotations, assign groups, tracking QC |
| `02_window_features.py` | Skeleton cleaning (likelihood, anatomical attachment, jump removal), 16 kinematic and proximal-tail posture features per 1 s window, per-animal phase summaries, Mann-Whitney + BH |
| `03_movement_modes.py` | GMM on log speed × log tail-base angular velocity (fit on controls, K = 4), K selection, decoding (speed-only fallback for fast swimming), smoothing, bouts, time budgets, transitions, rasters, ethograms |
| `04_precursor.py` | Onset-aligned curves, look-back prediction of expansion (leave-one-animal-out, permutation null), still-bout outcomes and conversion, execution of the expansion, stillness panels |
| `05_posture_frames.py` | Frame-level segment angles around expansion onsets and still-bout ends, eigen-tail PCA on control frames, skeleton panels per annotated phase |
| `06_embedding_whole.py` | Whole-video 1 s posture-dynamics features (eigen-tail medians, Lomb-Scargle spectra, speed, rotation), balanced UMAP, behaviour-balanced watershed, merge to 8 regions |
| `07_embedding_quantification.py` | Condition densities, phases on the map, region composition (χ², batch-stratified permutation), posture features within regions (Cliff's delta, permutation + Mann-Whitney, BH), skeletons per region |
| `08_production_figures.py` | Publication versions of the maps from the saved embedding: fixed mm layout, no text over data, label overlays, vector points / filled regions |
| `docs/` | Feature descriptions (embedding inputs, region posture features) |

## Pipeline

```
DLC csv + annotation txt
  → 01_inventory               →  01_metadata_animals.csv, 01_annotation_events.csv
  → 02_window_features         →  02_features/  (windows_long, metadata_with_qc, wide/, phase summaries, stats)
  → 03_movement_modes          →  03_modes/, 03b_decoding/  (03b_windows_long_decoded.csv, bouts,
                                   time budgets, transitions, rasters, ethograms)
  → 04_precursor               →  04_precursor/, fig_panels/  (onset curves, lag prediction,
                                   still-bout outcomes, conversion, execution, stillness panels)
  → 05_posture_frames          →  05_posture/  (05a_frames.parquet, 05_eigentail_pca.pkl), fig_panels/
  → 06_embedding_whole         →  fig_embedding_whole/  (windows_whole_embedded.parquet, model_whole.pkl)
  → 07_embedding_quantification →  fig_embedding_whole/  (D, E, E3, F, G, H, STATS_*.csv)
  → 08_production_figures      →  fig_embedding_whole/production/
```

## Key parameters

| Parameter | Value | Location |
|-----------|-------|----------|
| FPS / window / stride | 30 fps / 1 s / 0.5 s | `common.py` |
| Likelihood threshold / gap fill | 0.80 / ≤ 3 frames | `common.py` |
| Anatomical attachment / jump removal | link ≤ 3.5 × median length / centroid > 1.5 trunk lengths from 2 s median | `common.py` |
| Length unit | Lref = proximal tail length (bp5→10→8→12) | `common.py` |
| Still window / still bout | median speed < 0.1 Lref/s / ≥ 5 s | `common.py` |
| Movement modes | GMM K = 4, full covariance, fit on control windows (≤ 300 per animal) | `03_movement_modes.py` |
| Decoding | speed-only posterior ≥ 0.80 (fast only), gap bridge 1 s, min bout 1.5 s | `03_movement_modes.py` |
| Look-back prediction | L = 5–30 s, 2 s gap, logistic regression, 200 permutations | `04_precursor.py` |
| Eigen-tail PCA | angles a0–a3, control frames B1 + B2 | `05_posture_frames.py` |
| Embedding features | 4 posture medians + 4 × 8 Lomb-Scargle powers (0.75–12 Hz) + log speed + log rotation | `06_embedding_whole.py` |
| UMAP | n_neighbors 15, min_dist 0.1, fit balanced over builders/non-builders × control/6-OHDA | `06_embedding_whole.py` |
| Regions | watershed σ = 6 on behaviour-balanced density, JSD merge to 8, ≥ 0.3 % windows and ≥ 10 animals | `06_embedding_whole.py` |
| Statistics | per-animal values; permutation (10 000, labels shuffled within batch), Mann-Whitney, BH | `07_embedding_quantification.py` |
| Exclusions | 20250818_135423 (condition conflict), 7 suspected-housed B2 controls, implausible Lref | `01`, `common.py`, `02` |

## Notes

- Batches: B1 = May–Jun 2025 (each animal recorded once), B2 = Aug–Sep 2025, B2026 = new recordings (6-OHDA re-recorded until building started, so onset latency is not used).
- Published maps and statistics use the saved `windows_whole_embedded.parquet` + `model_whole.pkl`; UMAP is not bit-reproducible across library versions, so `06` will not overwrite them unless `OVERWRITE = True`.
- Region names: fill `REGION_NAMES_W` in `common.py`.

## Requirements

```
numpy pandas scipy scikit-learn scikit-image umap-learn pyarrow
matplotlib seaborn
```

Edit `HD` in `common.py`, then run the steps in order (each reads the previous outputs from `OUT`).
