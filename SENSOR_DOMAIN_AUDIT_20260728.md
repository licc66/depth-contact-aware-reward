# StackCube Sensor-Domain Audit - 2026-07-28

## Scope

- Dataset: `stackcube_bootstrap_v2_settled_20260728`
- Trajectories: 144
- Frames: 7,996
- Independent source-success groups: 8
- Group leakage across train/val/test: 0
- Online input table and offline simulator-supervision table are separate.

## Stereo Geometry

| Quantity | Error |
| --- | ---: |
| Object depth MAE | 18.09 mm |
| Goal depth MAE | 18.58 mm |
| Object-goal 3D distance MAE | 8.87 mm |
| Object-goal 3D distance P90 | 19.80 mm |
| Object-goal XY error MAE | 10.58 mm |
| Object-goal height error MAE | 7.17 mm |

Geometry is useful for coarse manipulation progress, but the P90 distance error
is too large to treat stereo alone as a precise insertion/contact-success test.

## Missingness

- Valid object/goal stereo geometry: 90.17% of all frames.
- Missing geometry: 786 frames (9.83%).
- All 786 missing frames belong to train groups `SC-SUCC-0003` and
  `SC-SUCC-0004`.
- Validation and test geometry are both 100% valid in this bootstrap dataset.

Therefore aggregate validation/test results can be optimistic. V2 must append
feature-validity indicators, use modality dropout, and report performance by
source group. This dataset cannot establish robustness to arbitrary sensor
failure.

## Frozen V1 On Sensor Inputs

| Metric | Result |
| --- | ---: |
| Frame potential vs diagnostic GT progress Spearman | 0.1722 |
| Mean clean-success trajectory Spearman | 0.7119 |
| Terminal preference accuracy | 0.7074 |
| Raw SGBM terminal-distance preference accuracy | 0.7500 |
| Failure stage-4 false-positive rate | 0.0000 |
| Completion recall at potential >= 0.8 | 0.5333 |

V1 does not transfer cleanly from simulator-projected geometry to SGBM input.
Its terminal ordering is worse than the simple observable distance baseline.
V1 remains a baseline and must not be presented as the final physical branch.

## Contract Gap Found

The v2 dataset contains object and goal geometry but no observable gripper/TCP
geometry. Consequently, depth cannot determine pre-grasp approach progress; a
model can only infer that phase indirectly from action/contact history. A new
stereo adapter therefore adds an image-derived gripper-center proxy. Pilot
results on a complete success replay show 102/102 valid frames and 14.46 mm MAE
against simulator TCP-object distance. Full v3 collection is required before
training the next model.

Entity localization currently uses ManiSkill renderer segmentation IDs. These
IDs do not expose object poses, but they are a simulation-specific perception
aid. A real-world claim would require replacing them with an image detector or
segmenter and repeating the sensor-domain audit.

## Decision

1. Keep the frozen v1 checkpoint and raw-distance rule as explicit baselines.
2. Train v2 only on the gripper-augmented sensor dataset.
3. Compare depth-only, contact-only, and gated depth-contact fusion variants.
4. Use simulator fields only as offline stage/progress supervision.
5. Treat eight source groups as a bootstrap study, not broad generalization.
6. Do not claim RL improvement until the frozen reward wrapper is tested in
   policy learning with multiple seeds.
