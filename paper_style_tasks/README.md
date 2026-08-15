# Paper-style ManiSkill Tasks

This folder is now the recommended task-generation entry point.

The active demos use ManiSkill official Panda motion-planning solutions through `mplib` in WSL2. This is preferred over the earlier guided demos because the arm actually plans and executes the manipulation instead of directly moving the object pose.

## Active Files

- `wsl_env.sh`: activates the WSL virtual environment and local Vulkan renderer.
- `wsl_verify_maniskill.py`: checks ManiSkill, SAPIEN, mplib, and rendering.
- `wsl_run_official_motionplanning.py`: generates official motion-planning trajectories and videos.
- `wsl_local_vulkan_setup.sh`: rebuilds the user-local Mesa/Vulkan fallback if WSL rendering breaks.

## Outputs

Recommended videos are saved under:

```text
paper_style_tasks/outputs/wsl_motionplanning/<EnvId>/motionplanning/
```

Current recommended tasks:

- `StackCube-v1`
- `StackPyramid-v1`
- `PegInsertionSide-v1`

Use the root `运行命令说明.txt` for exact commands.
