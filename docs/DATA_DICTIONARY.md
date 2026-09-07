# Data dictionary

## Processed development data

`outputs/master_fusion_dataset_clean.csv` contains one row per synchronized frame from
Experiments 1-11.

| Group | Representative columns | Description |
|---|---|---|
| Frame identity | `exp_id`, `frame_idx`, `video_time_s` | Experiment and temporal location |
| Synchronization | `pc2_frame_epoch_s`, `sensor_pc2_recv_epoch_s`, `sensor_delta_ms` | Camera-sensor timing information |
| Process signals | `sensor_force`, `sensor_rpm`, `sensor_torque`, `sensor_temp` | Synchronized sensor inputs |
| Vision geometry | `weld_width_mm`, `weld_area_mm2` | Weld-mask measurements |
| Defect geometry | `Burrs_area_mm2`, `flash_burr_area_mm2`, `surface_groove_void_area_mm2`, `total_defect_area_mm2` | Class-specific and total defect areas |
| Temporal derivatives | columns ending in `_diff` | Within-experiment first-order differences |
| Target | `target_id`, `defect_present` | Four-class future target and binary frame state |

The `video_path` and `experiment_dir` fields are sanitized relative references. Raw video
files are not included.

## Out-of-fold predictions

`out_of_fold_predictions.csv` contains predictions for four configurations, five outer
folds, and ten seeds. Important fields include `configuration`, `seed`, `fold`,
`sequence_id`, `true_id`, and `pred_id`.

## Independent matched data

`FINAL_H10_ZERO_SHOT_MATCHED_896.csv` combines predictions generated at frame `t` with
independently annotated future targets at frame `t+10`. It contains predicted probabilities,
multimodal inputs, annotation identifiers, the four-class target, and the collapsed binary
target.

## Final Fusion-LSTM summaries

`Final_H10_Fusion_PerClass_Analysis/H10_overall_performance_summary.csv` stores the final
ten-seed overall values shown in the README. `H10_per_class_summary_10seeds.csv` stores
precision, recall, specificity, and F1 mean, standard deviation, and confidence interval for
each class. `H10_consensus_per_class_metrics.csv` stores the consensus prediction metrics.

## Binary future-state package

`Final_H10_Binary_Future_Weld_State/H10_binary_all_10_seed_OOF_predictions.csv` contains
25,810 rows: 2,581 held-out sequences for each of ten seeds. `true_binary` and `pred_binary`
contain the Good/Defect collapse. The accompanying per-seed metrics, ten-seed summary,
class-performance table, raw and normalized confusion matrices, audit, manuscript text, and
figure provide the complete final binary result package.
