#!/usr/bin/env bash
# Build all APG capability packages as Python wheels.
#
# Usage:
#   ./scripts/build_all_packages.sh                   # build all
#   ./scripts/build_all_packages.sh intel fintech fin # build selected domains
#   JOBS=8 ./scripts/build_all_packages.sh           # parallel builds
#
# Output: dist/wheels/
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAPS_DIR="$REPO_ROOT/capabilities"
OUT_DIR="$REPO_ROOT/dist/wheels"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
JOBS="${JOBS:-4}"

mkdir -p "$OUT_DIR"

echo "APG Capability Package Builder"
echo "  Capabilities: $CAPS_DIR"
echo "  Output:       $OUT_DIR"
echo "  Workers:      $JOBS"
echo ""

# Collect capability directories to build
mapfile -t DIRS < <(
  find "$CAPS_DIR" -name pyproject.toml \
    -not -path "*/__pycache__/*" \
    -exec dirname {} \; | sort
)

# Filter by domain if args provided
if [[ $# -gt 0 ]]; then
  FILTERED=()
  for dir in "${DIRS[@]}"; do
    for domain in "$@"; do
      if [[ "$dir" == *"/capabilities/$domain"* ]]; then
        FILTERED+=("$dir")
        break
      fi
    done
  done
  DIRS=("${FILTERED[@]}")
fi

echo "Building ${#DIRS[@]} packages..."
echo ""

PASS=0; FAIL=0
FAILED_CAPS=()

build_one() {
  local cap_dir="$1"
  local rel="${cap_dir#$CAPS_DIR/}"
  if "$PYTHON" -m build --wheel --outdir "$OUT_DIR" "$cap_dir" \
       >"$OUT_DIR/.build_${rel//\//_}.log" 2>&1; then
    echo "  ✓  $rel"
    return 0
  else
    echo "  ✗  $rel  (see $OUT_DIR/.build_${rel//\//_}.log)"
    return 1
  fi
}

export -f build_one
export PYTHON OUT_DIR CAPS_DIR

if command -v parallel &>/dev/null; then
  # GNU parallel for concurrent builds
  printf '%s\n' "${DIRS[@]}" | parallel --jobs "$JOBS" build_one
else
  # Sequential fallback
  for dir in "${DIRS[@]}"; do
    if build_one "$dir"; then
      ((PASS++)) || true
    else
      ((FAIL++)) || true
      FAILED_CAPS+=("$dir")
    fi
  done
fi

echo ""
echo "Build complete."
echo "  Wheels:  $(find "$OUT_DIR" -name '*.whl' | wc -l | tr -d ' ')"
echo "  Output:  $OUT_DIR"

if [[ ${#FAILED_CAPS[@]} -gt 0 ]]; then
  echo ""
  echo "FAILURES:"
  printf '  %s\n' "${FAILED_CAPS[@]}"
  exit 1
fi
