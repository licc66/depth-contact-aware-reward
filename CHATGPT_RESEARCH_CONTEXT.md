# Research Context: Depth-Contact-Aware Reward Learning

## Project goal

This project studies a lightweight multimodal reward/progress model for robotic manipulation in ManiSkill. The intended online inputs are RGB or frozen visual features, stereo-derived geometry, contact/gripper signals, actions, and short history. Commercial VLM calls are offline only.

## Current task and data status

- Main pilot task: ManiSkill `StackCube-v1`.
- The current sensor-aligned StackCube v2 table has 484 pair candidates: 365 train, 60 validation, and 59 test.
- The sensor collection contains 144 derived samples from 8 source groups: 6 train, 1 validation, and 1 test. Most derived samples are truncations or terminal spatial perturbations of the original successful trajectories, so they are not 144 independent behaviors.
- The frozen common benchmark contains 95 pairs across three tasks and is used only to audit teacher labels.
- The frozen same-stage hard benchmark contains 91 StackCube pairs. It is a test set, and all pairs come from one held-out source group; it must not be used for training or threshold tuning.
- Existing MiMo labels are legacy labels on the old pair/video table. They are retained for audit and baseline comparison, not treated as primary supervision for the sensor-aligned v2 videos.

## Current research question

How should the dataset be expanded so that a reward model learns useful semantic and physical progress rather than memorizing trajectory time, endpoint offsets, or simulator-derived shortcuts?

The next study should prioritize new independent source groups and qualitatively different behaviors: successful policies, failed approaches, failed grasps, dropped objects, wrong placement, near-miss alignment, and recovery attempts. Pair construction should include cross-stage, same-stage progress, near-ties, and regressions while keeping train/validation/test separated by source group.

## Important evaluation rules

1. Do not use `reference_label_v2`, success flags, object pose, or other privileged simulator truth as online model inputs.
2. Do not use the frozen 95-pair or 91-pair benchmarks for training, fusion threshold selection, or prompt tuning.
3. Treat truncations and perturbations derived from one successful rollout as correlated views, not independent trajectories.
4. Report both pair count and independent source-group count.
5. Preserve raw teacher labels and create separate filtered/fused labels with provenance.

## Desired research output

Recommend a concrete ManiSkill data expansion protocol, including trajectory types, perturbation mechanisms, pair sampling ratios, source-group split rules, teacher-label budget, hard-negative mining, and ablations that test whether added data improves near-miss and physical-consistency judgments.
