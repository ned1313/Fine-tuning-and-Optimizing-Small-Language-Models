# Pull the Qwen-3.8B model from Hugging Face and convert it to GGUF format for use with GGUF-compatible tools.
# Save repo root path to a variable
$repoRoot = "C:\gh\Fine-tuning-and-Optimizing-Small-Language-Models"
# Save llama.cpp path to a variable
$llamaCppPath = "C:\gh\llama.cpp"

# Download the model from Hugging Face to the artifacts directory
Write-Host "Downloading Qwen-3.8B model from Hugging Face..."
hf download "Qwen/Qwen3-8B" --local-dir ./artifacts/Qwen3-8B

# Convert the model to GGUF format using the convert script
Write-Host "Converting Qwen-3.8B model to GGUF format..."
cd $llamaCppPath
.venv\Scripts\activate

python .\convert_hf_to_gguf.py "$repoRoot\artifacts\Qwen3-8B\"  `
    --outfile "$repoRoot\artifacts\Qwen3-8B\Qwen3-8B-f16.gguf" --outtype f16

# Quantize the GGUF model to 4-bit using the quantize script
Write-Host "Quantizing Qwen-3.8B model to 4-bit GGUF format..."
llama quantize "$repoRoot\artifacts\Qwen3-8B\Qwen3-8B-f16.gguf"  `
    "$repoRoot\artifacts\Qwen3-8B\Qwen3-8B-Q4_K_M.gguf" Q4_K_M
