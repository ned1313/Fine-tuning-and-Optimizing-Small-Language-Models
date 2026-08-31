#!/usr/bin/env python
"""
adapter_to_gguf.py

Convert a standard Hugging Face PEFT LoRA adapter to an adapter GGUF by
invoking llama.cpp's convert_lora_to_gguf.py.

Example:
    python adapter_to_gguf.py \
        --llama-cpp C:\\gh\\llama.cpp \
        --adapter .\\dpo_lora \
        --base C:\\models\\Qwen3-8B \
        --output .\\dpo_lora-f16.gguf \
        --outtype f16

IMPORTANT FOR PiSSA:
Convert the trained PiSSA adapter to a standard LoRA delta first. The
llama.cpp converter serializes LoRA tensors and metadata; it does not perform
PiSSA decomposition/reconstruction for you.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a PEFT LoRA adapter to GGUF using llama.cpp."
    )
    parser.add_argument(
        "--llama-cpp",
        required=True,
        help="Path to the llama.cpp repository.",
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="Path to the PEFT LoRA adapter directory.",
    )
    base_group = parser.add_mutually_exclusive_group(required=True)
    base_group.add_argument(
        "--base",
        help=(
            "Path to the original Hugging Face base model directory. "
            "llama.cpp needs its config/tokenizer metadata; base weights are "
            "not required by convert_lora_to_gguf.py."
        ),
    )
    base_group.add_argument(
        "--base-model-id",
        help=(
            "Hugging Face model ID for the original base model. "
            "Passed through to llama.cpp's --base-model-id option."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass --trust-remote-code to llama.cpp when using a model that requires it.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output GGUF filename.",
    )
    parser.add_argument(
        "--outtype",
        choices=["f32", "f16", "bf16", "q8_0", "auto"],
        default="f16",
        help="Adapter tensor output type. Default: f16.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help=(
            "Python executable used to run llama.cpp's converter. "
            "Defaults to the interpreter running this wrapper."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Pass --verbose to convert_lora_to_gguf.py.",
    )
    return parser.parse_args()


def validate_adapter(adapter_dir: Path) -> None:
    config_path = adapter_dir / "adapter_config.json"
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing adapter_config.json: {config_path}")

    if not safetensors_path.exists() and not bin_path.exists():
        raise FileNotFoundError(
            "Expected adapter_model.safetensors or adapter_model.bin in "
            f"{adapter_dir}"
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    init_method = config.get("init_lora_weights")

    # This catches the obvious/raw PiSSA case. A PiSSA adapter produced through
    # the residual-model workflow may have init_lora_weights=True, so this check
    # cannot prove an adapter is portable.
    if isinstance(init_method, str) and init_method.startswith("pissa"):
        raise RuntimeError(
            "This adapter is marked as raw PiSSA "
            f"(init_lora_weights={init_method!r}). Convert it to a standard "
            "LoRA delta with PEFT's path_initial_model_for_weight_conversion "
            "before converting it to GGUF."
        )

    print(f"Adapter init_lora_weights: {init_method!r}")
    print(
        "Note: for PiSSA adapters trained from a preprocessed residual model, "
        "make sure this directory is the PEFT-converted portable LoRA adapter, "
        "not the raw trained PiSSA adapter."
    )


def main() -> None:
    args = parse_args()

    llama_cpp = Path(args.llama_cpp).resolve()
    converter = llama_cpp / "convert_lora_to_gguf.py"
    adapter = Path(args.adapter).resolve()
    output = Path(args.output).resolve()

    if not converter.exists():
        raise FileNotFoundError(f"llama.cpp converter not found: {converter}")

    validate_adapter(adapter)

    output.parent.mkdir(parents=True, exist_ok=True)

    command = [
        args.python,
        str(converter),
        str(adapter),
    ]

    if args.base:
        base = Path(args.base).resolve()
        if not base.exists():
            raise FileNotFoundError(f"Base model directory not found: {base}")
        command.extend(["--base", str(base)])
    else:
        command.extend(["--base-model-id", args.base_model_id])

    command.extend(
        [
            "--outfile",
            str(output),
            "--outtype",
            args.outtype,
        ]
    )

    if args.trust_remote_code:
        command.append("--trust-remote-code")

    if args.verbose:
        command.append("--verbose")

    print("Running:")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))

    subprocess.run(
        command,
        cwd=llama_cpp,
        check=True,
    )

    print(f"GGUF adapter written to: {output}")


if __name__ == "__main__":
    main()
