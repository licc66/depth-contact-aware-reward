# RoboCLIP-style Baseline

This folder contains a weak, practical RoboCLIP-style reproduction for the current project.

It does not train a policy or fine-tune CLIP. Instead, it uses a pretrained CLIP/open-clip model to score the three ManiSkill videos with:

- image-to-task-text similarity
- image-to-final-success-frame similarity
- image-to-ordered-stage-text similarity

This is useful as a zero-shot trajectory-similarity baseline. It is not yet a full progress-reward evaluation because the three tasks still need unified frame-level environment-truth `stage_id` and `true_progress` labels.

## Run

From the project root in PowerShell:

```powershell
wsl -d Ubuntu-22.04 -- bash -lc "source paper_style_tasks/wsl_env.sh && python RoboCLIP-style/01_run_roboclip_baseline.py"
```

## Outputs

```text
RoboCLIP-style/outputs/three_task_roboclip/roboclip_summary.csv
RoboCLIP-style/outputs/three_task_roboclip/roboclip_samples.csv
RoboCLIP-style/outputs/three_task_roboclip/roboclip_score_curves.png
RoboCLIP-style/outputs/three_task_roboclip/stage_similarity_heatmaps.png
RoboCLIP-style/outputs/three_task_roboclip/contact_sheets/
```

Interpretation:

- `roboclip_score_norm`: combined task-text and final-frame visual similarity.
- `done_probability`: CLIP softmax between task-complete text and not-complete text.
- `weighted_stage_progress`: progress-like value from ordered stage-text similarities.
- Current order metrics are compared to time-normalized proxy progress, not true environment progress.
