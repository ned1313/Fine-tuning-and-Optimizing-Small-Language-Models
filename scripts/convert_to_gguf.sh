#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
llama_cpp_path="${LLAMA_CPP_PATH:-$repo_root/../llama.cpp}"
model_path="${MODEL_PATH:-$repo_root/artifacts/sft_full_run/final_model}"
model_path="$(cd -- "$model_path" && pwd)"

cd -- "$llama_cpp_path"
uv venv
uv sync
uv pip install --python .venv/bin/python --upgrade "transformers>=5"
source .venv/bin/activate

python convert_hf_to_gguf.py "$model_path" \
    --outfile "$model_path/taco_alley_sft.gguf" --outtype bf16

llama quantize "$model_path/taco_alley_sft.gguf" \
    "$model_path/taco_alley_sft_Q4_K_S.gguf" Q4_K_S
