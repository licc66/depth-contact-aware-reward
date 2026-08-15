# Step 4 Review - 2026-07-28

## Status

Steps 1-4 are complete. Work is paused before the online wrapper, WSL ManiSkill,
and RL stages.

## Corrections Made

- Merged and audited the Fable5 Phase 2-4 handoff.
- Fixed canonical source-group ids. All 1,782 full-table pairs now map to the
  script-17 train/val/test groups; previously all would have been unassigned.
- Added canonical `--split-map-dir` support to scripts 24 and 25 and retrained
  the physical branch without auxiliary train/val/test contamination.
- Exposed the frozen physical clip embedding without changing checkpoint-v5
  parameters.
- Made missing-modality fusion deterministic: missing physical selects RGB;
  missing RGB selects physical.
- Added full-pair metadata support to scripts 20, 28, and 29 so conflict rows
  are not silently dropped again.
- Corrected PAV calibration for tied scores and chance-corrected the agreement
  confidence fusion.
- Made `rgb_only` and `physical_only` true single-branch ablations, rather than
  carrying unused fusion parameters.
- Split hard-pair error from strict success-vs-near-miss pairwise false
  positives in the evaluation report.
- Added deny-list, split-map, checkpoint, missing-modality, and exporter
  contract tests. Final result: 30/30 tests passed.

## Physical Branch Retraining

Checkpoint:
`D:\Users\User\Desktop\reward_model_dataset\reward_model_runs\physical_progress_branch_v1_splitfixed\best_model.pt`

- Canonical auxiliary trajectories: train 316, val 57, test 65.
- Pair accuracy: val 0.9783, test 0.9852.
- Test hard-negative accuracy: 1.0000.
- Parameters: 149,709.

Independent trajectory audit:

| Task | Success Spearman | Success stage-4 | Near-miss stage-4 |
| --- | ---: | ---: | ---: |
| StackCube | 0.772 | 1.000 | 0.000 |
| StackPyramid | 0.816 | 1.000 | 0.000 |
| PegInsertion | 0.920 | 1.000 | 0.045 |

PegInsertion still has physical near-miss failures that reach stage 4. This is
not resolved by the current physical model.

## Fusion Labels

Directory:
`D:\Users\User\Desktop\reward_model_dataset\fusion_labels\fusion_labels_v1_splitfixed_full`

- Full rows: 1,782; unassigned rows: 0.
- Labels: 1,094 `A>B`, 548 `B>A`, 140 abstain.
- Val reference accuracy: 0.9695; val coverage: 0.8678.
- Source-group leakage: 0; base-success leakage: 0.
- The test split was not evaluated during threshold selection.

The reference is still the constructed `candidate_label`; this accuracy is a
label-reconstruction result, not independent task-success ground truth.

## Reward Model Results

Directory:
`D:\Users\User\Desktop\reward_model_dataset\reward_model_runs\reward_model_v1_splitfixed_full`

| Variant | Params | Val pair acc. | Test pair acc. | Hard-pair error | Strict near-miss pair FP | Completion FP | Success recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RGB only | 217,351 | 0.8934 | 0.8132 | 0.1639 | 0.1633 | 0.0204 | 1.000 |
| Physical only | 19,231 | 0.9645 | 0.8794 | 0.1421 | 0.1633 | 0.0000 | 1.000 |
| Fusion | 219,428 | 0.9695 | 0.8911 | 0.1421 | 0.1633 | 0.0000 | 1.000 |

Fusion improves test pair accuracy over physical-only by 0.0117, but it does
not improve strict near-miss pairwise FP in this run. Fusion test accuracy by
task is PegInsertion 0.9844, StackCube 0.7652, and StackPyramid 1.0000. The
`success_vs_offset_hard_negative` accuracy is only 0.6667.

The learned fusion gate has mean RGB weight 0.381 and median 0.378 on test
clips, so it primarily uses the physical branch while retaining nonzero RGB
input. Test ECE is 0.1261 / 0.1400 / 0.1472 for RGB / physical / fusion; fusion
is not better calibrated.

Reported latency is reward-head-only. It excludes OpenCLIP encoding and the
frozen 149,709-parameter physical branch, so it must not be presented as total
online latency.

## Artifacts

- Physical scores and embeddings:
  `D:\Users\User\Desktop\reward_model_dataset\physical_pair_scores\physical_pair_scores_v1_splitfixed_full`
- Full OpenCLIP cache: 847 clips, 5,029 frame references, 512 dimensions:
  `D:\Users\User\Desktop\reward_model_dataset\reward_model_features\openclip_vit_b32_v1_full`
- Full metrics:
  `D:\Users\User\Desktop\reward_model_dataset\reward_model_runs\reward_model_v1_splitfixed_full\metrics_full.json`

## Boundaries Before Continuing

- Current stereo features are simulator-state projections, not image-derived
  stereo estimates. No claim about real stereo-depth robustness is justified.
- Pair labels and several evaluation subsets remain construction-entangled.
- Reward-model trajectory Spearman on test has only four qualifying success
  trajectories; it is a small diagnostic, not a strong generalization result.
- The online RGB feature contract, actual stereo/contact observation adapters,
  wrapper behavior, and RL training have not been implemented or validated yet.
