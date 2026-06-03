#!/usr/bin/env bash
#SBATCH --job-name=mr-brain-recommended
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#
# Array worker for the recommended MR brain generation matrix. Submit via:
#   scripts/submit_recommended_mr_brain_images.sh

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

CONFIG="${1:-configs/config_generate_mr_brain_recommended.json}"
ENV_TEMPLATE="configs/environment_maisi_diff_model_rflow-mr-brain.json"
MODEL_DEF="configs/config_network_rflow.json"
GENERATE_VERSION="rflow-mr-brain"
NUM_GPUS="${NUM_GPUS:-1}"
TARGET_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "Config file not found: ${CONFIG}" >&2
    exit 1
fi

if [[ ! -f "${ENV_TEMPLATE}" ]]; then
    echo "Environment template not found: ${ENV_TEMPLATE}" >&2
    exit 1
fi

mapfile -t TARGET_VALUES < <(
    uv run --frozen python - "${CONFIG}" "${TARGET_INDEX}" <<'PY'
import json
import sys
from pathlib import PurePosixPath

config_path, target_index_text = sys.argv[1:3]
target_index = int(target_index_text)

with open(config_path) as f:
    config = json.load(f)

targets = config.get("targets", [])
if target_index < 0 or target_index >= len(targets):
    raise ValueError(f"target_index {target_index} is outside target range 0..{len(targets) - 1}")

num_images = int(config.get("num_images", 1))
if num_images < 1:
    raise ValueError("num_images must be >= 1")

base_output_dir = config.get("output_dir")
if not base_output_dir:
    raise ValueError("output_dir must be set in the config")

target = targets[target_index]
required_keys = [
    "condition",
    "modality_name",
    "modality",
    "plane",
    "dim",
    "spacing",
    "fov",
    "output_subdir",
]
missing = [key for key in required_keys if key not in target]
if missing:
    raise ValueError(f"target {target_index} is missing keys: {missing}")

base_seed = int(config.get("diffusion_unet_inference", {}).get("random_seed", 0))
target_seed = base_seed + target_index * 100000
output_dir = str(PurePosixPath(base_output_dir) / target["output_subdir"])

print(num_images)
print(output_dir)
print(target_seed)
print(target["condition"])
print(target["modality_name"])
print(target["modality"])
print(target["plane"])
print("x".join(str(v) for v in target["dim"]))
print("x".join(str(v) for v in target["spacing"]))
print("x".join(str(v) for v in target["fov"]))
PY
)

NUM_IMAGES="${TARGET_VALUES[0]}"
OUTPUT_DIR="${TARGET_VALUES[1]}"
TARGET_RANDOM_SEED="${TARGET_VALUES[2]}"
CONDITION="${TARGET_VALUES[3]}"
MODALITY_NAME="${TARGET_VALUES[4]}"
MODALITY="${TARGET_VALUES[5]}"
PLANE="${TARGET_VALUES[6]}"
DIM_LABEL="${TARGET_VALUES[7]}"
SPACING_LABEL="${TARGET_VALUES[8]}"
FOV_LABEL="${TARGET_VALUES[9]}"

mkdir -p "${OUTPUT_DIR}"

WORK_DIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/mr-brain-recommended.XXXXXX")"
trap 'rm -rf "${WORK_DIR}"' EXIT

RUNTIME_CONFIG="${WORK_DIR}/config_target_${TARGET_INDEX}.json"
RUNTIME_ENV_CONFIG="${WORK_DIR}/environment_target_${TARGET_INDEX}.json"

uv run --frozen python - "${CONFIG}" "${RUNTIME_CONFIG}" "${TARGET_INDEX}" "${TARGET_RANDOM_SEED}" <<'PY'
import json
import sys

config_path, runtime_config_path, target_index_text, target_seed_text = sys.argv[1:5]
target_index = int(target_index_text)
target_seed = int(target_seed_text)

with open(config_path) as f:
    config = json.load(f)

target = config["targets"][target_index]
runtime_config = {
    "diffusion_unet_train": config.get("diffusion_unet_train", {}),
    "diffusion_unet_inference": dict(config.get("diffusion_unet_inference", {})),
}
runtime_config["diffusion_unet_inference"]["dim"] = target["dim"]
runtime_config["diffusion_unet_inference"]["spacing"] = target["spacing"]
runtime_config["diffusion_unet_inference"]["modality"] = int(target["modality"])
runtime_config["diffusion_unet_inference"]["random_seed"] = target_seed

with open(runtime_config_path, "w") as f:
    json.dump(runtime_config, f, indent=4)
    f.write("\n")
PY

uv run --frozen python - "${ENV_TEMPLATE}" "${RUNTIME_ENV_CONFIG}" "${OUTPUT_DIR}" <<'PY'
import json
import sys

template_path, runtime_path, output_dir = sys.argv[1:4]
with open(template_path) as f:
    env_config = json.load(f)

env_config["output_dir"] = output_dir

with open(runtime_path, "w") as f:
    json.dump(env_config, f, indent=4)
    f.write("\n")
PY

echo "Config: ${CONFIG}"
echo "Array target: ${TARGET_INDEX}"
echo "Condition: ${CONDITION}"
echo "Modality: ${MODALITY_NAME} (${MODALITY})"
echo "Plane: ${PLANE}"
echo "Dim: ${DIM_LABEL}"
echo "Spacing: ${SPACING_LABEL}"
echo "FOV: ${FOV_LABEL}"
echo "Target base random seed: ${TARGET_RANDOM_SEED}"
echo "Output folder: ${OUTPUT_DIR}"
echo "Number of images: ${NUM_IMAGES}"

uv run --frozen python -m scripts.download_model_data \
    --version "${GENERATE_VERSION}" \
    --root_dir "./" \
    --model_only

uv run --frozen python -m scripts.diff_model_infer_MANY \
    -t "${MODEL_DEF}" \
    -e "${RUNTIME_ENV_CONFIG}" \
    -c "${RUNTIME_CONFIG}" \
    -g "${NUM_GPUS}" \
    --num_images "${NUM_IMAGES}"

echo "Done. Generated files are in: ${OUTPUT_DIR}"
