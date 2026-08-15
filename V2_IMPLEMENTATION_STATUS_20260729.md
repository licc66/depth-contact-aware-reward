# Reward v2 Implementation Status - 2026-07-29

## Completed

1. Collected the StackCube sensor-consistent dataset through the online SGBM, segmentation, gripper-geometry, and contact adapters.
2. Removed 10 label-conflict near-miss trajectories that actually reached simulator success.
3. Trained and cross-validated physical progress v2.2 on observable sensor features.
4. Rebuilt 484 video pairs and exported physical preferences plus 128-D physical clip embeddings.
5. Precomputed frozen OpenCLIP ViT-B/32 features for 229 sensor-aligned clips.
6. Built physical-only audit fusion labels and trained RGB-only, physical-only, and fusion reward-model v2 variants.
7. Added the frozen online reward v2 scorer without modifying the frozen v1 wrapper.
8. Added scheduled sensor acquisition: contact is read every step; stereo/RGB is refreshed every K steps and at termination.
9. Added a terminal consistency constraint below the 0.75 completion boundary.
10. Added a v2 SAC launcher and a resumable multi-seed launcher.
11. Ran all four SAC conditions end to end and ran a three-seed sparse/physical integration smoke.
12. Added sensor-label validation and an API-key-free operation document.

## Verified Results

### Physical v2.2 fixed test

- stage macro-F1: 0.724
- potential Spearman: 0.914
- pair accuracy: 1.000
- potential MAE: 0.011
- success-trajectory Spearman: 0.969
- terminal false-positive rate: 0.000

### Physical v2.2 eight-fold source-group CV

- stage macro-F1: rule 0.638 +/- 0.154; fusion 0.743 +/- 0.118
- potential Spearman: rule 0.766 +/- 0.318; fusion 0.839 +/- 0.093
- pair accuracy: rule 0.929 +/- 0.122; fusion 0.997 +/- 0.008
- potential MAE: rule 0.044 +/- 0.060; fusion 0.030 +/- 0.021
- The learned model is better on average, but it is not uniformly safer on every fold.

### Online wrapper

- Unit tests: 62/62 passed.
- Frozen v1 SHA-256 remains `5400B110069DE8B48EACB67FE9378892B793EF19E20FBD255B901BD3D9323222`.
- Physical-only and OpenCLIP-fusion dry runs changed the sparse reward zero times.
- Interval 1 adapter mean: about 162 ms/step over 24 steps.
- Interval 4 adapter mean: about 92 ms/step over 24 steps.
- Interval 4 non-refresh step: about 0.77 ms; SGBM refresh remains the bottleneck.

### RL integration

- sparse-only, RGB-only, physical-only, and fusion each completed 64 training steps, saved SAC policies, and ran an independent evaluation episode.
- Three seeds (3, 7, 11) completed and aggregated through the resumable launcher.
- These runs are integration smoke tests. Zero success at 64 steps has no performance meaning.

## Current Blockers

### 1. Missing sensor-aligned semantic labels

The 484 new pairs use newly rendered sensor-consistent videos. Their retained legacy VLM labels came from different videos and cannot train the primary semantic branch. Current reward checkpoints therefore contain:

```text
semantic_source = none
primary_scientific_result = false
```

The labeling operation is documented in `SENSOR_ALIGNED_VLM_LABELING_V2.md`. No API key was read or used during this continuation.

### 2. No untouched final test set

The current data has only eight independent success source groups. The existing test group has already been inspected during physical-model and reward-model development. It must remain a development/audit test, even after adding VLM labels.

A final claim needs newly collected independent source groups, a newly frozen group-level split, and a test set that is not viewed until the method and hyperparameters are fixed. Fusion v2 now marks a run primary only when both conditions hold:

```text
semantic_source = sensor_aligned
test_split_status = untouched
```

### 3. Completion calibration is underdetermined

The current reward-model test contains only one completion clip. Physical-only and fusion achieved zero near-miss false positives but also zero test completion recall at the validation-selected threshold. This is not evidence of safety; it is evidence that the terminal calibration sample is too small.

## Deliberately Not Started

- No 200k-step multi-seed RL run was launched.
- No result from the current audit-only reward model is labeled as a scientific comparison.
- No commercial VLM API was called.

## Recommended Next Gate

1. Run a 12-pair VLM-label pilot and inspect the raw responses.
2. If label quality is acceptable, finish all 484 development labels and retrain the sensor-aligned development fusion model.
3. Collect additional independent StackCube success source groups and derived failures.
4. Freeze a new untouched group-level holdout before further tuning.
5. Only then start full multi-seed RL.
