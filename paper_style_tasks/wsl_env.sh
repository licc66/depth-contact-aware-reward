#!/usr/bin/env bash
# Source this file inside WSL before running ManiSkill scripts.

set -euo pipefail

export MANISKILL_WSL_VENV="${MANISKILL_WSL_VENV:-/home/jbda/codex_envs/maniskill_mplib}"
export VULKAN_LOCAL_BASE="${VULKAN_LOCAL_BASE:-/home/jbda/vulkan_local}"
export VULKAN_LOCAL_LIB_DIR="$VULKAN_LOCAL_BASE/root/usr/lib/x86_64-linux-gnu"
export VULKAN_LOCAL_ICD="$VULKAN_LOCAL_BASE/icd/lvp_icd.local.json"

if [[ ! -f "$MANISKILL_WSL_VENV/bin/activate" ]]; then
  echo "Missing WSL virtual environment: $MANISKILL_WSL_VENV" >&2
  echo "Create it first, or set MANISKILL_WSL_VENV to the correct path." >&2
  return 1 2>/dev/null || exit 1
fi

source "$MANISKILL_WSL_VENV/bin/activate"
export PYTHON="$MANISKILL_WSL_VENV/bin/python"

if [[ -d "$VULKAN_LOCAL_LIB_DIR" ]]; then
  export LD_LIBRARY_PATH="$VULKAN_LOCAL_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [[ -f "$VULKAN_LOCAL_ICD" ]]; then
  export VK_ICD_FILENAMES="$VULKAN_LOCAL_ICD"
fi

export MANISKILL_RENDER_BACKEND=cpu
