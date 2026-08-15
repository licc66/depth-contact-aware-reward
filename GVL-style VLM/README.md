# GVL-style VLM

This folder keeps the GVL-style baseline.

Current local script:

- `01_run_gvl_style_local.py`
- `02_run_mimo_gvl_baseline.py`

`01_run_gvl_style_local.py` is a local weak reproduction, not a commercial VLM/API run. It uses OpenCLIP to compare sampled video frames with ordered, goal-conditioned stage prompts, then converts the stage probabilities into a progress-like score.

`02_run_mimo_gvl_baseline.py` calls MiMo as the commercial VLM baseline. The current default model is `mimo-v2.5`, chosen for multimodal image understanding. The script reads the API key from `MIMO_API_KEY`, `XIAOMI_MIMO_API_KEY`, or the local ignored file `mimo apikey.txt`; the key is not written to outputs.

Run from the project root:

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/d/Users/User/Desktop/双目深度reward && source paper_style_tasks/wsl_env.sh && python 'GVL-style VLM/01_run_gvl_style_local.py'"
```

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/d/Users/User/Desktop/双目深度reward && source paper_style_tasks/wsl_env.sh && python 'GVL-style VLM/02_run_mimo_gvl_baseline.py' --num-samples 6"
```

Outputs:

- `outputs/three_task_gvl_local/gvl_local_summary.csv`
- `outputs/three_task_gvl_local/gvl_local_samples.csv`
- `outputs/three_task_gvl_local/gvl_local_progress_curves.png`
- `outputs/three_task_gvl_local/gvl_local_stage_heatmaps.png`
- `outputs/three_task_gvl_local/RESULTS.txt`
- `outputs/three_task_mimo_gvl/mimo_gvl_summary.csv`
- `outputs/three_task_mimo_gvl/mimo_gvl_samples.csv`
- `outputs/three_task_mimo_gvl/mimo_gvl_progress_curves.png`
- `outputs/three_task_mimo_gvl/raw_responses/*.json`

Important: metrics are currently compared with time-normalized proxy progress, not environment-truth progress or stage labels.
