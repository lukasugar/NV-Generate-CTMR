#!/usr/bin/env bash
#SBATCH --job-name=mr-brain-generation
#SBATCH --partition=main
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#
# Cluster-specific Slurm settings to review before submitting:
#   - Add or edit #SBATCH --partition=<partition_name> if your cluster requires it.
#   - Add or edit #SBATCH --account=<account_name> if your cluster requires it.
#   - Increase --time, --mem, --cpus-per-task, or --gres=gpu:N for larger volumes.
#
# This script calls scripts.diff_model_infer_MANY, which loads the model once
# and generates multiple images sequentially via --num_images.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# CONFIG INPUT:
# Set the config here, or pass it as the first sbatch argument.
# This config controls num_images, output_dir, and diffusion_unet_inference.
CONFIG="${1:-configs/config_generate_mr_brain_images.json}"

# Fixed MR brain model files. Change these only if you are intentionally using
# another model definition, environment file, or generation version.
ENV_TEMPLATE="configs/environment_maisi_diff_model_rflow-mr-brain.json"
MODEL_DEF="configs/config_network_rflow.json"
GENERATE_VERSION="rflow-mr-brain"

# Number of GPUs used by scripts.diff_model_infer_MANY. Keep this aligned with
# the Slurm GPU request above, for example --gres=gpu:2 and NUM_GPUS=2.
NUM_GPUS="${NUM_GPUS:-1}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "Config file not found: ${CONFIG}" >&2
    exit 1
fi

if [[ ! -f "${ENV_TEMPLATE}" ]]; then
    echo "Environment template not found: ${ENV_TEMPLATE}" >&2
    exit 1
fi

# Parameter notes for the config:
#   num_images:
#     Number of images to generate. This value is passed to
#     scripts.diff_model_infer_MANY as --num_images.
#   output_dir:
#     Output folder for generated .nii.gz files. This script creates the folder
#     explicitly and injects it into a temporary runtime environment config.
#   diffusion_unet_inference.random_seed:
#     Base seed. Image i uses base_seed + i + local_rank.
#   diffusion_unet_inference.modality:
#     Whole-brain: 9=T1, 10=T2, 11=FLAIR, 20=SWI.
#     Skull-stripped: 29=T1, 30=T2, 31=FLAIR, 32=SWI.
#   diffusion_unet_inference.dim and spacing:
#     Together define field of view: FOV[i] = dim[i] * spacing[i].
#     For good MR brain quality, stay close to the recommended FOV table in
#     docs/inference.md for the selected modality and acquisition plane.
#   diffusion_unet_inference.num_inference_steps:
#     Keep 30 for rflow-mr-brain.
#   diffusion_unet_inference.cfg_guidance_scale:
#     Keep 10 for MR contrast guidance unless you are intentionally tuning it.

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

output_dir = config.get("output_dir")
if not output_dir:
    raise ValueError("output_dir must be set in the config")

print(num_images)
print(output_dir)
PY
)
NUM_IMAGES="${CONFIG_VALUES[0]}"
OUTPUT_DIR="${CONFIG_VALUES[1]}"

# OUTPUT FOLDER:
# Generated .nii.gz files are written here. Keep this explicit so Slurm jobs do
# not silently write to the default ./output folder.
mkdir -p "${OUTPUT_DIR}"

WORK_DIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/mr-brain-many.XXXXXX")"
trap 'rm -rf "${WORK_DIR}"' EXIT

RUNTIME_ENV_CONFIG="${WORK_DIR}/environment_mr_brain_runtime.json"

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

echo "Config: ${CONFIG}"
echo "Output folder: ${OUTPUT_DIR}"
echo "Number of images: ${NUM_IMAGES}"

uv run python -m scripts.download_model_data \
    --version "${GENERATE_VERSION}" \
    --root_dir "./" \
    --model_only

uv run python -m scripts.diff_model_infer_MANY \
    -t "${MODEL_DEF}" \
    -e "${RUNTIME_ENV_CONFIG}" \
    -c "${CONFIG}" \
    -g "${NUM_GPUS}" \
    --num_images "${NUM_IMAGES}"

echo "Done. Generated files are in: ${OUTPUT_DIR}"
