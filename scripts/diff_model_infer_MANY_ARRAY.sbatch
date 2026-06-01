#!/usr/bin/env bash
#SBATCH --job-name=mr-brain-array
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --array=0-3
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#
# Submit one array job to generate all skull-stripped MR brain modalities:
#   0 -> mri_t1_skull_stripped    modality 29
#   1 -> mri_t2_skull_stripped    modality 30
#   2 -> mri_flair_skull_stripped modality 31
#   3 -> mri_swi_skull_stripped   modality 32
#
# Cluster-specific Slurm settings to review before submitting:
#   - Edit #SBATCH --partition=<partition_name> if your cluster requires another partition.
#   - Edit #SBATCH --account=<account_name> if your cluster requires another account.
#   - Increase --time, --mem, --cpus-per-task, or --gpus-per-task for larger volumes.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# BASE CONFIG INPUT:
# Pass a base config as the first sbatch argument, or use the default.
# This script creates a temporary per-array-task config that overrides:
#   - diffusion_unet_inference.modality
#   - output_dir
CONFIG="${1:-configs/config_generate_mr_brain_images.json}"

# Fixed MR brain model files. Change these only if you are intentionally using
# another model definition, environment file, or generation version.
ENV_TEMPLATE="configs/environment_maisi_diff_model_rflow-mr-brain.json"
MODEL_DEF="configs/config_network_rflow.json"
GENERATE_VERSION="rflow-mr-brain"

# Number of GPUs used by scripts.diff_model_infer_MANY. Keep this aligned with
# the Slurm GPU request above.
NUM_GPUS="${NUM_GPUS:-1}"

MODALITIES=(29 30 31 32)
MODALITY_NAMES=(
    "mri_t1_skull_stripped"
    "mri_t2_skull_stripped"
    "mri_flair_skull_stripped"
    "mri_swi_skull_stripped"
)

ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( ARRAY_TASK_ID < 0 || ARRAY_TASK_ID >= ${#MODALITIES[@]} )); then
    echo "Unsupported SLURM_ARRAY_TASK_ID=${ARRAY_TASK_ID}; expected 0-3." >&2
    exit 1
fi

MODALITY="${MODALITIES[$ARRAY_TASK_ID]}"
MODALITY_NAME="${MODALITY_NAMES[$ARRAY_TASK_ID]}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "Config file not found: ${CONFIG}" >&2
    exit 1
fi

if [[ ! -f "${ENV_TEMPLATE}" ]]; then
    echo "Environment template not found: ${ENV_TEMPLATE}" >&2
    exit 1
fi

# Parameter notes for the base config:
#   num_images:
#     Number of images to generate per array task/modality.
#   output_dir:
#     Base output folder. This script writes each modality under:
#       <output_dir>/<modality_name>
#   diffusion_unet_inference.modality:
#     Ignored from the base config for this array script; selected from
#     MODALITIES using SLURM_ARRAY_TASK_ID.
#   diffusion_unet_inference.random_seed:
#     Base seed. The Python entrypoint uses base_seed + image_index + local_rank.
#   diffusion_unet_inference.dim and spacing:
#     Together define field of view: FOV[i] = dim[i] * spacing[i].
#   diffusion_unet_inference.num_inference_steps:
#     Keep 30 for rflow-mr-brain.
#   diffusion_unet_inference.cfg_guidance_scale:
#     Keep 10 for MR contrast guidance unless intentionally tuning it.

mapfile -t CONFIG_VALUES < <(
    uv run python - "${CONFIG}" <<'PY'
import json
import sys

config_path = sys.argv[1]
with open(config_path) as f:
    config = json.load(f)

num_images = int(config.get("num_images", 1))
if num_images < 1:
    raise ValueError("num_images must be >= 1")

base_output_dir = config.get("output_dir")
if not base_output_dir:
    raise ValueError("output_dir must be set in the config")

print(num_images)
print(base_output_dir)
PY
)
NUM_IMAGES="${CONFIG_VALUES[0]}"
BASE_OUTPUT_DIR="${CONFIG_VALUES[1]}"
OUTPUT_DIR="${BASE_OUTPUT_DIR%/}/${MODALITY_NAME}"

# OUTPUT FOLDER:
# Generated .nii.gz files for this array task are written here.
mkdir -p "${OUTPUT_DIR}"

WORK_DIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/mr-brain-array.XXXXXX")"
trap 'rm -rf "${WORK_DIR}"' EXIT

RUNTIME_CONFIG="${WORK_DIR}/config_${MODALITY_NAME}.json"
RUNTIME_ENV_CONFIG="${WORK_DIR}/environment_${MODALITY_NAME}.json"

uv run python - "${CONFIG}" "${RUNTIME_CONFIG}" "${MODALITY}" "${OUTPUT_DIR}" <<'PY'
import json
import sys

base_config_path, runtime_config_path, modality, output_dir = sys.argv[1:5]
with open(base_config_path) as f:
    config = json.load(f)

config["output_dir"] = output_dir
config.setdefault("diffusion_unet_inference", {})
config["diffusion_unet_inference"]["modality"] = int(modality)

with open(runtime_config_path, "w") as f:
    json.dump(config, f, indent=4)
    f.write("\n")
PY

uv run python - "${ENV_TEMPLATE}" "${RUNTIME_ENV_CONFIG}" "${OUTPUT_DIR}" <<'PY'
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

echo "Base config: ${CONFIG}"
echo "Runtime config: ${RUNTIME_CONFIG}"
echo "Array task: ${ARRAY_TASK_ID}"
echo "Modality: ${MODALITY_NAME} (${MODALITY})"
echo "Output folder: ${OUTPUT_DIR}"
echo "Number of images: ${NUM_IMAGES}"

uv run python -m scripts.download_model_data \
    --version "${GENERATE_VERSION}" \
    --root_dir "./" \
    --model_only

uv run python -m scripts.diff_model_infer_MANY \
    -t "${MODEL_DEF}" \
    -e "${RUNTIME_ENV_CONFIG}" \
    -c "${RUNTIME_CONFIG}" \
    -g "${NUM_GPUS}" \
    --num_images "${NUM_IMAGES}"

echo "Done. Generated files are in: ${OUTPUT_DIR}"
