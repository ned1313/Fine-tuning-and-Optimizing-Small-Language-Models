#!/usr/bin/env python
"""
merge_adapter.py

Merge a PEFT LoRA/PiSSA adapter into a Hugging Face base model.

For a standard LoRA adapter:
    python merge_adapter.py \
        --base-model Qwen/Qwen3-8B \
        --adapter ./dpo_adapter \
        --output ./dpo_merged

For a PiSSA adapter trained using a preprocessed residual base:
    python merge_adapter.py \
        --base-model Qwen/Qwen3-8B \
        --adapter ./dpo_adapter \
        --output ./dpo_merged \
        --pissa-init ./pissa_init \
        --pissa-residual-base ./pissa_residual_model \
        --portable-adapter-output ./dpo_lora

The PiSSA path first converts the trained PiSSA adapter into an equivalent
standard LoRA delta, then reloads a fresh copy of the ORIGINAL base model and
merges that portable adapter into it.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a PEFT LoRA/PiSSA adapter into a Hugging Face model."
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Original Hugging Face base model ID or local path.",
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="Path to the trained PEFT adapter.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory for the merged Hugging Face model.",
    )
    parser.add_argument(
        "--dtype",
        choices=DTYPES,
        default="bf16",
        help="Precision used while merging. Default: bf16.",
    )
    parser.add_argument(
        "--device-map",
        default="cpu",
        help='Transformers device_map value. Default: "cpu". Use "auto" if desired.',
    )
    parser.add_argument(
        "--tokenizer-source",
        default=None,
        help="Tokenizer source. Defaults to --base-model.",
    )

    # PiSSA-specific arguments.
    parser.add_argument(
        "--pissa-init",
        default=None,
        help=(
            "Path to the UNTRAINED PiSSA initialization adapter snapshot. "
            "When supplied, the trained PiSSA adapter is converted to an "
            "equivalent standard LoRA adapter before merging."
        ),
    )
    parser.add_argument(
        "--pissa-residual-base",
        default=None,
        help=(
            "Residual/decomposed base model used during PiSSA training. "
            "Required when the PiSSA workflow preprocessed the model and "
            "trained the adapter against that residual model."
        ),
    )
    parser.add_argument(
        "--portable-adapter-output",
        default=None,
        help=(
            "Directory in which to save the PiSSA adapter converted to ordinary "
            "LoRA. Strongly recommended if the adapter will be distributed or "
            "converted to GGUF."
        ),
    )
    return parser.parse_args()


def read_adapter_config(adapter_path: str | Path) -> dict:
    config_path = Path(adapter_path) / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing PEFT adapter config: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_base(model_id_or_path: str, dtype: torch.dtype, device_map: str):
    print(f"Loading model: {model_id_or_path}")
    resolved_device_map = None if device_map == "cpu" else device_map
    return AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        dtype=dtype,
        device_map=resolved_device_map,
        low_cpu_mem_usage=True,
    )


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    dtype = DTYPES[args.dtype]

    adapter_config = read_adapter_config(args.adapter)
    init_method = adapter_config.get("init_lora_weights")

    print(f"Adapter: {args.adapter}")
    print(f"Adapter init_lora_weights: {init_method!r}")

    adapter_to_merge = args.adapter

    if args.pissa_init:
        portable_output = Path(
            args.portable_adapter_output
            or (str(Path(args.output).with_name(Path(args.output).name + "_adapter")))
        )
        portable_output.mkdir(parents=True, exist_ok=True)

        # In the preprocessed PiSSA + QLoRA workflow, the trained adapter is
        # attached to the residual model, not directly to the original base.
        conversion_base_path = args.pissa_residual_base or args.base_model

        if args.pissa_residual_base:
            print(
                "PiSSA mode: reconstructing the trained adapter on the residual "
                "base used during training."
            )
        else:
            print(
                "PiSSA mode: no residual base supplied; assuming the trained "
                "adapter can be reconstructed directly on the original base."
            )

        conversion_base = load_base(
            conversion_base_path,
            dtype=dtype,
            device_map=args.device_map,
        )

        pissa_model = PeftModel.from_pretrained(
            conversion_base,
            args.adapter,
            is_trainable=False,
        )

        print("Converting trained PiSSA weights to a portable standard-LoRA delta...")
        pissa_model.save_pretrained(
            portable_output,
            safe_serialization=True,
            path_initial_model_for_weight_conversion=args.pissa_init,
        )

        print(f"Portable LoRA adapter saved to: {portable_output}")
        adapter_to_merge = str(portable_output)

        # Drop both references before loading the fresh original base model.
        del pissa_model
        del conversion_base
        cleanup()
    else:
        if isinstance(init_method, str) and init_method.startswith("pissa"):
            print(
                "WARNING: This appears to be a raw PiSSA adapter. PEFT can "
                "reconstruct raw PiSSA from the original base by performing the "
                "PiSSA initialization again, but for distribution/GGUF conversion "
                "you should prefer the portable LoRA conversion and supply "
                "--pissa-init."
            )
        else:
            print("Using adapter as a standard LoRA adapter.")

    # Always merge into a fresh copy of the ORIGINAL base model.
    base_model = load_base(
        args.base_model,
        dtype=dtype,
        device_map=args.device_map,
    )

    peft_model = PeftModel.from_pretrained(
        base_model,
        adapter_to_merge,
        is_trainable=False,
    )

    print("Merging adapter into base model...")
    merged_model = peft_model.merge_and_unload(safe_merge=True)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Saving merged model to: {output}")
    merged_model.save_pretrained(
        output,
        safe_serialization=True,
    )

    tokenizer_source = args.tokenizer_source or args.base_model
    print(f"Saving tokenizer from: {tokenizer_source}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    tokenizer.save_pretrained(output)

    print("Done.")


if __name__ == "__main__":
    main()
