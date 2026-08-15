# Stereo Depth Truth Prototype

This folder is the first prototype for adding fixed-view stereo depth to the reward project.

There are two levels in this folder:

1. `01_extract_truth_depth_features.py` uses ManiSkill environment truth as a stand-in for what a future fixed stereo camera pair would recover:

- `tcp_3d`
- `object_3d`
- `goal_3d`
- object-goal 3D distance
- fixed-camera depth error
- fixed-camera lateral alignment
- pseudo disparity from `Z = fB / disparity`

2. `02_stereo_sgbm_compare.py` renders real fixed left/right images, computes disparity with OpenCV StereoSGBM, and compares the recovered disparity/depth with ManiSkill truth.

The point is to validate whether the geometry signal is useful, then measure how much noise appears when replacing truth with image-based stereo matching.

## Run

From the project root:

```powershell
wsl -d Ubuntu-22.04 -- bash -lc "source paper_style_tasks/wsl_env.sh && python stereo_depth_truth/01_extract_truth_depth_features.py"
```

Run real stereo image matching:

```powershell
wsl -d Ubuntu-22.04 -- bash -lc "source paper_style_tasks/wsl_env.sh && python stereo_depth_truth/02_stereo_sgbm_compare.py"
```

## Outputs

```text
stereo_depth_truth/outputs/three_task_truth_depth/truth_depth_features.csv
stereo_depth_truth/outputs/three_task_truth_depth/truth_depth_summary.csv
stereo_depth_truth/outputs/three_task_truth_depth/truth_depth_progress_curves.png
stereo_depth_truth/outputs/three_task_truth_depth/fixed_camera_geometry_paths.png
stereo_depth_truth/outputs/stereo_sgbm_compare/stereo_sgbm_summary.csv
stereo_depth_truth/outputs/stereo_sgbm_compare/stereo_sgbm_frame_metrics.csv
stereo_depth_truth/outputs/stereo_sgbm_compare/stereo_sgbm_entity_metrics.csv
stereo_depth_truth/outputs/stereo_sgbm_compare/stereo_sgbm_entity_summary.csv
stereo_depth_truth/outputs/stereo_sgbm_compare/previews/*.png
```

Use these outputs to decide how to combine depth geometry with the process reward.
