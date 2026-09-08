# Fine-tuning and Optimizing Small Language Models

Welcome to the companion repository for my Pluralsight course, Fine-tuning and Optimizing Small Language Models!

Contained within this repository are the datasets, notebooks, and scripts used throughout the course. I invite you to follow along and experiment with these resources. But first a few notes on the structure of the repository, hardware and software requirements, and expectations for time/effort.

## Repository Contents

Most demonstrations use Taco Alley, a fictional restaurant business, to explore complaint processing, response preferences, and SQL generation.

| Notebook | What it covers | Default model |
| --- | --- | --- |
| [Basic SFT example](notebooks/basic_sft_example.ipynb) | An introductory supervised fine-tuning workflow | Qwen3-0.6B-Base |
| [Dataset exploration](notebooks/datasets.ipynb) | Loading datasets from Hugging Face and local files | None |
| [Synthetic data generation](notebooks/data_generation.ipynb) | Generating and validating synthetic records with OpenAI | Hosted OpenAI model |
| [Full SFT](notebooks/sft_full_run.ipynb) | Fine-tuning all model weights to structure complaints as JSON | Qwen3-1.7B |
| [LoRA SFT](notebooks/sft_lora_run.ipynb) | Parameter-efficient fine-tuning with LoRA and PiSSA initialization | Qwen3-1.7B |
| [QLoRA SFT](notebooks/sft_qlora_run.ipynb) | Training adapters on a 4-bit base model | Qwen3-8B |
| [DPO with QLoRA](notebooks/dpo_qlora_run.ipynb) | Direct Preference Optimization using chosen/rejected responses | Qwen3-8B |
| [GRPO with LoRA](notebooks/grpo_lora_run.ipynb) | Reward-based training for SQL generation against SQLite | Qwen2.5-Coder-1.5B |

The [datasets](datasets) directory includes complaint records, seed and synthetic data, preference pairs, sample sentiment data, and a WikiText sample. The [SQL data directory](datasets/data) contains the SQLite database, schema, training prompts, and evaluation inputs used for SQL experiments.

The [scripts](scripts) directory contains utilities for merging adapters, converting models and adapters to GGUF, evaluating complaint-structuring models with llama.cpp, and running short GRPO trials. The YAML files in [scripts/tests](scripts/tests) are trial configurations, not an automated test suite. Training notebooks write generated checkpoints, adapters, and metrics beneath the ignored `artifacts/` directory.

The notebooks are ordered to match how they are introduced in the course. Some notebooks build on the work of previous notebooks, so be aware running them out of order may cause issues. Run cells in order within each notebook. You can use the supplied datasets without regenerating them.

These are hands-on training experiments, not instant demos. Downloads and initial evaluation can take substantial time before training begins; training duration depends on your GPU, model, batch size, sequence length, and number of steps. Start with a small run, inspect its metrics, and budget additional time for evaluation and export. GRPO generates completions during training, so even a small model can take considerable time. No fixed runtime or identical result across devices is guaranteed.

## Hardware and Software Requirements

### Hardware

- **GPU:** An Nvidia GPU with a compatible driver is the default local path. A 16 GB-class GPU is a useful planning target for these exercises, not a guarantee that every notebook fits unchanged. Full fine-tuning, 8B QLoRA, DPO, and GRPO have different memory requirements. Start with the 0.6B example when resources are limited.
- **System memory:** 32 GB RAM is a practical recommendation, with more useful when loading or merging larger models. GPU VRAM and system RAM are separate constraints.
- **Storage:** Reserve tens of gigabytes of free SSD space for model downloads, the Python environment, checkpoints, and exported models. Multiple runs or 8B model exports can require substantially more.
- **Internet:** Required for initial package, model, and online dataset downloads, and for any hosted API calls.

If you run out of GPU memory, first lower the per-device batch size or sequence length. Gradient accumulation can help preserve the effective batch size, and gradient checkpointing trades computation for memory. Choose a smaller model or an adapter-based method when necessary. CPU-only execution is useful for data preparation and small checks, but is not a practical substitute for the larger training runs. A rented GPU or hosted notebook is another option; check pricing and shut down paid instances when finished.

Each of the notebooks includes a link to run the notebook on Google Colab. Please note that while the free tier of Google Colab does offer a 16GB VRAM GPU instance, it is from the Tesla generation of GPUs and will be pretty slow on the training runs. The free tier also limits how long you can run an instance before it times out. You may find that some notebooks cannot complete their training before that time limit. Adjusting the training run or dataset slice can lower the time required. You can also opt for a paid instance that comes with a better GPU and no time restrictions. I've done my best to limit the cost required to follow along in the course, but alas technology is not entirely free.

### Software

- Git and [uv](https://docs.astral.sh/uv/getting-started/installation/) for cloning the repository and managing the environment.
- Python: [.python-version](.python-version) selects **3.14**, while [pyproject.toml](pyproject.toml) declares **3.10 or newer**. That lower bound does not guarantee compatible wheels for every dependency on every platform. Start with the repository's selected version and check backend compatibility before changing it.
- VS Code with the Python and Jupyter extensions, or JupyterLab in a browser.
- PyTorch, Transformers, Datasets, TRL, PEFT, bitsandbytes, and the other dependencies declared in [pyproject.toml](pyproject.toml). [uv.lock](uv.lock) records the resolved environment.
- For Nvidia: a driver compatible with the configured **CUDA 13.2** PyTorch build. Check `nvidia-smi` and the [PyTorch installation guidance](https://pytorch.org/get-started/locally/). Prebuilt PyTorch wheels generally do not require a separately installed CUDA toolkit unless you build extensions.

The dependency configuration explicitly routes PyTorch to the `cu132` wheel index. The default setup below is therefore intended for Nvidia/CUDA, not a universal installation command for every GPU. Check out the sections dealing with other GPU types for guidance. If dependency resolution fails, check the selected Python version, available wheels, and backend compatibility before changing package versions.

Some workflows have additional requirements:

- **Synthetic data generation:** The notebook reads `OPENAI_TOKEN` from a `.env` file in its working directory. Supply your own OpenAI API credential locally; API usage can incur charges. Do not commit secrets. Generating new data is optional because sample data is included.
- **Hugging Face uploads:** Publishing a trained model requires an account and a token with appropriate permissions. Upload cells are optional; public model downloads generally do not require a token unless access is restricted.
- **GGUF conversion and evaluation:** Install [llama.cpp](https://github.com/ggml-org/llama.cpp) separately, including the converters and binaries required by the script you use. The PowerShell examples contain machine-specific Windows paths and must be adjusted. The optional `gguf` dependency extra installs `llama-cpp-python`, not the separate llama.cpp checkout or command-line tools.
- **GRPO:** Review the notebook's extra package installation cell, including `sqlglot`; optional vLLM acceleration has its own platform requirements. Before running, change the notebook's `DATASETS_DIR` to `REPO_ROOT / "datasets" / "data"` so it can find both the database and JSONL inputs in [datasets/data](datasets/data). When launching [scripts/trial.py](scripts/trial.py) from the repository root, supply `--db datasets/data/taco_alley.db --data-dir datasets/data` and select a configuration from [scripts/tests](scripts/tests). Treat these as experiments that may need local adjustments.

## Setting up the `ipykernel`

The kernel connects your notebook to the project's Python environment. Run the following from a terminal for the default Nvidia setup:

```powershell
git clone https://github.com/ned1313/Fine-tuning-and-Optimizing-Small-Language-Models.git
cd Fine-tuning-and-Optimizing-Small-Language-Models
uv sync --locked
uv run --locked python -m ipykernel install --user --name fto-slm --display-name "Python (fto-slm)"
```

If you already cloned the repository, start in its root directory and skip the first two commands. `uv sync --locked` creates the local `.venv` and installs the locked dependencies without updating the lockfile. You do not need to activate the environment when using `uv run`.

In VS Code, open a notebook, choose **Select Kernel**, and select **Python (fto-slm)**. If it is not listed, select the project's `.venv` interpreter through **Python Environments**. For a browser-based session, run:

```powershell
uv run --locked jupyter lab
```

Before loading a model, run this in a notebook cell to verify the interpreter and Nvidia GPU access:

```python
import sys
import torch

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

The interpreter should be inside this repository's `.venv`. On the default Nvidia setup, GPU availability should be `True`. If it is not, check the selected kernel, driver, and installed PyTorch build before starting training. Restart the kernel after changing packages. When moving between training notebooks, shut down unused kernels so their models do not continue occupying GPU memory.

## Running on non-Nvidia GPUs

The notebooks primarily use CUDA device checks, CUDA memory reporting, mixed precision, and, for QLoRA, bitsandbytes quantization. Changing the PyTorch installation alone may not be enough. The guidance below describes adaptation paths, not verified drop-in replacements for every exercise.

Use a separate environment for a different backend. Select a Python version and PyTorch build supported by your hardware and operating system using the vendor documentation. Install compatible versions of the remaining dependencies, then register that environment's kernel with `python -m ipykernel install`. Do not run the unchanged project's `uv sync` over a manually configured backend environment: its explicit CUDA index can fail or replace your chosen build. For a uv-managed alternative, update the PyTorch source configuration and resolve a backend-specific lockfile in your own copy.

Begin with a small, non-quantized SFT or LoRA run. Review device placement, precision flags, optimizers, and memory-reporting calls before attempting QLoRA, DPO, or GRPO. bitsandbytes support depends on its version, backend, and exact hardware; confirm support for the required 4-bit training operations, not just package installation.

### AMD

Use a supported [ROCm PyTorch installation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html). Consult AMD's current GPU and operating-system compatibility matrices; Linux, WSL, and native Windows support are not interchangeable.

ROCm PyTorch deliberately reuses much of the `torch.cuda` API, so `torch.cuda.is_available()` can be `True` on an AMD GPU. Check `torch.version.hip` to identify a ROCm build. This makes many CUDA-style device calls reusable, but does not establish compatibility with CUDA-only extensions or quantized training kernels. Verify precision support and start with full SFT or LoRA on a small model before adapting the bitsandbytes-based notebooks.

### Intel GPUs

Use the [PyTorch Intel GPU guidance](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html) to install an XPU-capable build for a supported GPU and operating system. Verify availability with `torch.xpu.is_available()`.

The notebooks' `torch.cuda.is_available()` branches will not detect an Intel GPU. Adapt explicit device placement and memory APIs to `xpu`, and review model loading, trainer device selection, and mixed-precision settings for the installed PyTorch/Transformers stack. Confirm that bitsandbytes supports the required XPU operations before attempting QLoRA; otherwise use non-quantized LoRA with a model that fits available memory.

### Mac Metal

On a supported Apple Silicon Mac, use a macOS PyTorch build with the [MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html). Verify availability with `torch.backends.mps.is_available()` and use the `mps` device for explicit model and tensor placement.

The CUDA checks in these notebooks will not recognize Metal. Review those branches and trainer settings rather than assuming that a CUDA-unavailable result means the Mac has no usable GPU. Start with float32 and disabled CUDA-oriented `fp16`/`bf16` training flags, then enable other precision modes only where supported by your installed stack. Some operations may be unsupported or fall back to the CPU, affecting performance.

Apple Silicon uses unified memory shared with the operating system, so its memory capacity is not equivalent to the same amount of dedicated GPU VRAM. Start with the smallest model and conservative batch sizes. Do not assume that the bitsandbytes 4-bit workflows will run unchanged on MPS. [MLX examples](https://github.com/ml-explore/mlx-examples/tree/main/llms) offer an Apple-native alternative for supported models, but require a different training workflow rather than simply switching this repository's kernel.
