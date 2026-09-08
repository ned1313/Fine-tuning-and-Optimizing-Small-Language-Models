#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
llama_cpp_path="${LLAMA_CPP_PATH:-$repo_root/../llama.cpp}"
model_path="$repo_root/artifacts/Qwen3-8B"

printf '%s\n' 'Downloading Qwen3-8B from Hugging Face...'
hf download "Qwen/Qwen3-8B" --local-dir "$model_path"

printf '%s\n' 'Converting Qwen3-8B to GGUF...'
cd -- "$llama_cpp_path"
source .venv/bin/activate

python convert_hf_to_gguf.py "$model_path" \
    --outfile "$model_path/Qwen3-8B-f16.gguf" --outtype f16

printf '%s\n' 'Quantizing Qwen3-8B to 4-bit GGUF...'
llama quantize "$model_path/Qwen3-8B-f16.gguf" \
    "$model_path/Qwen3-8B-Q4_K_M.gguf" Q4_K_M
