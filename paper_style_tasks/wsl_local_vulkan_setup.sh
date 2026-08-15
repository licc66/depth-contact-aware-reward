#!/usr/bin/env bash
# Install a user-local Mesa Vulkan software renderer inside WSL.
# This avoids needing sudo when SAPIEN cannot find a Vulkan ICD.

set -euo pipefail

BASE=/home/jbda/vulkan_local
PKGS="$BASE/pkgs"
ROOT="$BASE/root"
ICD_DIR="$BASE/icd"
LOCAL_ICD="$ICD_DIR/lvp_icd.local.json"

mkdir -p "$PKGS" "$ROOT" "$ICD_DIR"
cd "$PKGS"

apt-get download libvulkan1 mesa-vulkan-drivers vulkan-tools
rm -rf "$ROOT"
mkdir -p "$ROOT"

for deb in "$PKGS"/*.deb; do
  echo "extracting $deb"
  dpkg-deb -x "$deb" "$ROOT"
done

python - "$ROOT" "$LOCAL_ICD" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
target = Path(sys.argv[2])
icd_candidates = sorted(root.glob("usr/share/vulkan/icd.d/*lvp*.json"))
if not icd_candidates:
    raise SystemExit("Could not find lavapipe ICD JSON in extracted Mesa packages.")

data = json.loads(icd_candidates[0].read_text(encoding="utf-8"))
data["ICD"]["library_path"] = str(root / "usr/lib/x86_64-linux-gnu/libvulkan_lvp.so")
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(data, indent=2), encoding="utf-8")
print(f"Wrote {target}")
PY

echo "Use this before running ManiSkill:"
echo "source paper_style_tasks/wsl_env.sh"
