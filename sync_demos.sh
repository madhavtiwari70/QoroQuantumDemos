#!/usr/bin/env bash
#
# sync_demos.sh — refresh vendor/divi-demos/ from the real QoroQuantum/divi-demos repo.
#
# This is the ONLY thing that should ever touch vendor/divi-demos/. Never
# hand-edit files inside that folder — if you need different values, edit
# the data/*.yaml files instead. If you need different logic, that belongs
# in demos/<name>/<name>_wrapper.py.
#
# Usage:
#   ./sync_demos.sh                  # sync every demo folder
#   ./sync_demos.sh spin_dynamics    # sync just one
#
set -euo pipefail

UPSTREAM_REPO="https://github.com/QoroQuantum/divi-demos.git"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR_DIR="$SCRIPT_DIR/vendor/divi-demos"

ALL_DEMOS=(
  economic_load_dispatch
  minimum_birkhoff_decomposition
  portfolio_optimization
  quantum_guided_cluster
  spin_dynamics
  travelling_salesman
)

DEMOS_TO_SYNC=("$@")
if [ ${#DEMOS_TO_SYNC[@]} -eq 0 ]; then
  DEMOS_TO_SYNC=("${ALL_DEMOS[@]}")
fi

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

echo "Cloning $UPSTREAM_REPO ..."
git clone --depth 1 "$UPSTREAM_REPO" "$TMP_DIR" --quiet

UPSTREAM_COMMIT=$(git -C "$TMP_DIR" rev-parse --short HEAD)
UPSTREAM_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

mkdir -p "$VENDOR_DIR"

for demo in "${DEMOS_TO_SYNC[@]}"; do
  if [ ! -d "$TMP_DIR/$demo" ]; then
    echo "  ⚠️  '$demo' not found upstream (may have been renamed/removed) — skipping"
    continue
  fi
  rm -rf "$VENDOR_DIR/$demo"
  cp -r "$TMP_DIR/$demo" "$VENDOR_DIR/$demo"
  echo "  ✅ synced $demo"
done

cat > "$VENDOR_DIR/SYNC_INFO.md" << EOF
# Sync info

Last synced: $UPSTREAM_DATE
Upstream commit: $UPSTREAM_COMMIT
Upstream repo: $UPSTREAM_REPO

Everything under vendor/divi-demos/ is copied verbatim from the commit
above. Do not hand-edit these files — re-run sync_demos.sh instead.
EOF

echo ""
echo "Done. Upstream commit: $UPSTREAM_COMMIT"
echo "Review the changes with: git status vendor/"
echo "Then commit them: git add vendor/ && git commit -m \"Sync divi-demos @ $UPSTREAM_COMMIT\""
