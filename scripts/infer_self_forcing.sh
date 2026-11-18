#!/usr/bin/env bash
# FLEX inference with the Self Forcing configuration.
# Lengths used with this codebase:
#   21  latent frames ~ 5s  (training horizon, S = 1)
#   126 latent frames ~ 30s
#   240 latent frames ~ 60s
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

WAN_MODEL_ROOT="${WAN_MODEL_ROOT:-$ROOT/wan_models/Wan2.1-T2V-1.3B}"
CHECKPOINT="${CHECKPOINT:-$ROOT/checkpoints/self_forcing_dmd.pt}"
PROMPTS="${PROMPTS:-$ROOT/prompts/example.txt}"
NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-126}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-$ROOT/outputs/self_forcing_flex_${NUM_OUTPUT_FRAMES}}"
SEED="${SEED:-0}"

python inference.py \
  --config_path configs/self_forcing_flex.yaml \
  --wan_model_root "$WAN_MODEL_ROOT" \
  --checkpoint_path "$CHECKPOINT" \
  --data_path "$PROMPTS" \
  --output_folder "$OUTPUT_FOLDER" \
  --num_output_frames "$NUM_OUTPUT_FRAMES" \
  --seed "$SEED"
