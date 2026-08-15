# Sensor-Aligned VLM Labeling v2

## Current Status

- Pair table: 484 pairs, all videos present.
- Split: 365 train / 60 val / 59 test.
- The current test groups have already been inspected during development.
- Labels produced for this table are suitable for development and ablation, not a final untouched-test claim.
- The labeling script resumes from its existing CSV and does not relabel completed pair ids unless `--force` is used.

## 1. Label A Small Pilot

Set the API key in the current PowerShell session without writing it into this project:

```powershell
$env:MIMO_API_KEY = "<YOUR_KEY>"
```

Run 12 pairs first:

```powershell
& "C:\Users\User\PycharmProjects\pythonProject2\venv\Scripts\python.exe" `
  "D:\Users\User\Desktop\双目深度reward\dataset_generation\11_mimo_batch_pairwise_labels.py" `
  --queues "D:\Users\User\Desktop\reward_model_dataset\training_tables\stackcube_sensor_pairs_v2_20260728\stackcube_pairs_v2.csv" `
  --out "D:\Users\User\Desktop\reward_model_dataset\vlm_labels\stackcube_sensor_pairs_v2_20260729" `
  --model mimo-v2.5 `
  --need-filter true `
  --frames-per-clip 4 `
  --batch-size 3 `
  --limit 12
```

Inspect the 12 labels and raw responses, then rerun the same command without `--limit`. It will resume from the remaining pair ids.

## 2. Validate All Labels

```powershell
& "C:\Users\User\PycharmProjects\pythonProject2\venv\Scripts\python.exe" `
  "D:\Users\User\Desktop\双目深度reward\dataset_generation\47_validate_sensor_aligned_labels_v2.py" `
  --pairs "D:\Users\User\Desktop\reward_model_dataset\training_tables\stackcube_sensor_pairs_v2_20260728\stackcube_pairs_v2.csv" `
  --labels "D:\Users\User\Desktop\reward_model_dataset\vlm_labels\stackcube_sensor_pairs_v2_20260729\mimo_pairwise_labels.csv"
```

Proceed only when `valid_for_fusion` is `true`. Retry API-error pair ids before fusion.

## 3. Build Development Fusion Labels

```powershell
& "C:\Users\User\PycharmProjects\pythonProject2\venv\Scripts\python.exe" `
  "D:\Users\User\Desktop\双目深度reward\dataset_generation\40_build_fusion_labels_v2.py" `
  --pairs "D:\Users\User\Desktop\reward_model_dataset\training_tables\stackcube_sensor_pairs_v2_20260728\stackcube_pairs_v2.csv" `
  --physical-scores "D:\Users\User\Desktop\reward_model_dataset\physical_pair_scores\physical_pair_scores_v2_stackcube_20260728\physical_pair_scores_v2.csv" `
  --semantic-labels "D:\Users\User\Desktop\reward_model_dataset\vlm_labels\stackcube_sensor_pairs_v2_20260729\mimo_pairwise_labels.csv" `
  --semantic-source sensor_aligned `
  --test-split-status touched `
  --out-dir "D:\Users\User\Desktop\reward_model_dataset\fusion_labels\fusion_labels_v2_sensor_aligned_dev_20260729"
```

Do not add `--evaluate-test` while selecting fusion thresholds. A final test requires newly collected source groups and a newly frozen untouched split.
