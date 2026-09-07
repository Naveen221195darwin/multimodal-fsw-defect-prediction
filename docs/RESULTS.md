# Results and metric provenance

## Final reported H=10 model

The supplied final overall-performance CSV reports the following ten-seed multiclass performance:

| Metric | Mean (%) | SD (%) | 95% CI lower (%) | 95% CI upper (%) |
|---|---:|---:|---:|---:|
| Accuracy | 83.57 | 3.74 | 81.248104 | 85.888665 |
| Macro precision | 84.24 | 3.02 | 82.372573 | 86.111704 |
| Macro recall | 85.86 | 3.32 | 83.807420 | 87.918930 |
| Macro F1 | 84.21 | 3.34 | 82.142061 | 86.281005 |
| Weighted F1 | 83.51 | 3.70 | 81.222120 | 85.807605 |

The corresponding final binary package reports 88.16 ± 3.77% accuracy,
83.77 ± 4.85% macro F1, and 88.09 ± 4.00% Defect recall. These values are reproducible
from `H10_binary_all_10_seed_OOF_predictions.csv`.

The aggregated ten-seed raw matrix is:

| Actual / predicted | Good | Defect |
|---|---:|---:|
| Good | 4,651 | 609 |
| Defect | 2,448 | 18,102 |

## Ablation artifact set

`final_ablation_summary.csv` and `out_of_fold_predictions.csv` form a separate,
self-consistent ablation artifact set. Recalculation from the 25,810 final-model prediction
rows (2,581 sequences × 10 seeds) gives 79.32 ± 6.35% multiclass accuracy and
79.40 ± 6.67% macro F1 for Fusion-LSTM with derivatives.

The repository deliberately labels these values as the **ablation benchmark**, rather than
silently replacing the separate final-analysis summary. The authoritative binary results are
validated from the dedicated binary OOF package, not reconstructed from the ablation CSV.

## Independent evaluation

The corrected independent evaluation contains 896 matched H=10 targets:

- Exp. 12: 298 samples
- Exp. 13: 349 samples
- Exp. 14: 249 samples

The pooled raw multiclass matrix is:

| True / predicted | Good | Burr | Flash-burr | Surface-groove/void |
|---|---:|---:|---:|---:|
| Good | 145 | 224 | 4 | 2 |
| Burr | 34 | 66 | 11 | 59 |
| Flash-burr | 7 | 5 | 11 | 64 |
| Surface-groove/void | 25 | 29 | 18 | 192 |

The pooled binary matrix is:

| True / predicted | Good | Defect |
|---|---:|---:|
| Good | 145 | 230 |
| Defect | 66 | 455 |

Run `python scripts/validate_results.py` to verify the ablation and independent-evaluation
artifacts.
