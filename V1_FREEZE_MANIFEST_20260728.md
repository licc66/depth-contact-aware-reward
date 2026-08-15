# V1 Freeze Manifest - 2026-07-28

This file freezes the baseline before sensor-consistent v2 data collection.
V2 work must use new script, dataset, run, and checkpoint names. It must not
overwrite the artifacts below.

## Frozen Artifacts

| Artifact | SHA-256 |
| --- | --- |
| `dataset_generation/30_maniskill_reward_wrapper_v1.py` | `5400B110069DE8B48EACB67FE9378892B793EF19E20FBD255B901BD3D9323222` |
| `physical_progress_branch_v1_splitfixed/best_model.pt` | `3D32A2A9037A33CB7FA1FEBA8DE6203D1524B02124B448297C3F7F4A967A45AF` |
| `reward_model_v1_rgb_only.pt` | `FB85FB6219D9F6243BD0E028FB7CF603869DE07EA08448D32C4E86D109D48C21` |
| `reward_model_v1_physical_only.pt` | `779804DBBA5A8B7CB02A8EE72176F98D1019F0A841DA4C285F792FBCDEF3E5CB` |
| `reward_model_v1_fusion.pt` | `852235AD61367B84785AFFEA2C46EF02EC23EF269162C66C70E07850340CCF3D` |
| `bootstrap_v1_fusion_stereo_v1_clean/split_summary.json` | `3E86E98AF143126CB69E8FF2AC3E9AF4F8C3F9018DD860DD857E93F142E4211F` |

## Frozen Baseline Scope

- Physical input: simulator-projected geometry plus contact features.
- Reward variants: RGB-only, physical-only, and gated fusion.
- Online engineering smoke: passed; not a scientific RL comparison.
- Known sensor-domain gap: v1 was not trained on online SGBM geometry.
- Known PegInsertion gap: no observable hole locator.

## V2 Rules

1. Model inputs come only from `sensor_features.csv` and an explicit whitelist.
2. Simulator poses and evaluate outputs remain in `offline_supervision.csv`.
3. Source-group splits are inherited and checked before training.
4. All v2 results are reported separately from bootstrap v1 metrics.
