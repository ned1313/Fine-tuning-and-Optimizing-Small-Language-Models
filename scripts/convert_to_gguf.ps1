# Run these commands from the root of the llama.cpp repository.
uv venv

uv sync

uv pip install --upgrade "transformers>=5"

.\venv\Scripts\activate

# Update the model path to point to your fine-tuned model directory
$model_path="C:\gh\Fine-tuning-and-Optimizing-Small-Language-Models\artifacts\sft_full_run\final_model\"

python convert_hf_to_gguf.py $model_path --outfile "$model_path\taco_alley_sft.gguf" --outtype bf16

llama quantize "$model_path\taco_alley_sft.gguf" "$model_path\taco_alley_sft_Q4_K_S.gguf" Q4_K_S