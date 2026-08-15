# TOPReward-style

This folder keeps the TOPReward-style baseline.

Current local script:

- `01_run_topreward_style_local.py`

This is a local weak reproduction, not a commercial VLM/API run. It uses OpenCLIP to score each sampled trajectory prefix against two text prompts: task complete vs not complete. The prefix score is treated as a reward-like progress signal.

Run from the project root:

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/d/Users/User/Desktop/双目深度reward && source paper_style_tasks/wsl_env.sh && python 'TOPReward-style/01_run_topreward_style_local.py'"
```

Outputs:

- `outputs/three_task_topreward_local/topreward_local_summary.csv`
- `outputs/three_task_topreward_local/topreward_local_samples.csv`
- `outputs/three_task_topreward_local/topreward_local_progress_curves.png`
- `outputs/three_task_topreward_local/RESULTS.txt`

Important: metrics are currently compared with time-normalized proxy progress, not environment-truth progress or stage labels.
