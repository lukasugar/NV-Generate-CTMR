#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="configs/config_generate_mr_brain_recommended.json"
SBATCH_SCRIPT="scripts/generate_recommended_mr_brain_images.sbatch"
MAX_CONCURRENT=8
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --sbatch-script)
            SBATCH_SCRIPT="$2"
            shift 2
            ;;
        --max-concurrent)
            MAX_CONCURRENT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            cat <<'EOF'
Usage: scripts/submit_recommended_mr_brain_images.sh [--dry-run] [--config PATH] [--sbatch-script PATH] [--max-concurrent N]

Submits one Slurm array job for all recommended MR brain targets.
EOF
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ ! -f "${CONFIG}" ]]; then
    echo "Config file not found: ${CONFIG}" >&2
    exit 1
fi

if [[ ! -f "${SBATCH_SCRIPT}" ]]; then
    echo "Sbatch script not found: ${SBATCH_SCRIPT}" >&2
    exit 1
fi

if ! [[ "${MAX_CONCURRENT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--max-concurrent must be a positive integer, got: ${MAX_CONCURRENT}" >&2
    exit 1
fi

TARGET_COUNT="$(
    uv run --frozen python - "${CONFIG}" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    config = json.load(f)

targets = config.get("targets", [])
if not targets:
    raise ValueError("config must define at least one target")

print(len(targets))
PY
)"

ARRAY_MAX="$((TARGET_COUNT - 1))"
ARRAY_SPEC="0-${ARRAY_MAX}%${MAX_CONCURRENT}"
SBATCH_CMD=(
    sbatch
    "--array=${ARRAY_SPEC}"
    "${SBATCH_SCRIPT}"
    "${CONFIG}"
)

echo "Target count: ${TARGET_COUNT}"
echo "Array range: 0-${ARRAY_MAX}"
echo "Max concurrent array tasks: ${MAX_CONCURRENT}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf 'Dry run command:'
    printf ' %q' "${SBATCH_CMD[@]}"
    printf '\n'
    exit 0
fi

"${SBATCH_CMD[@]}"
