#!/usr/bin/env python3
"""
Export MentalRoBERTa-Caps to ONNX and optionally quantize.

Usage:
    python -m mentalroberta.tools.export_onnx \\
        --checkpoint checkpoints/best_model.pt \\
        --output checkpoints/model.onnx \\
        --opset 13 \\
        --quantize
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch

from mentalroberta.model import MentalRoBERTaCaps


def load_model(checkpoint: Path, model_name: str, num_classes: int, device: str) -> MentalRoBERTaCaps:
    model = MentalRoBERTaCaps(num_classes=num_classes, model_name=model_name)
    state = torch.load(checkpoint, map_location=device)

    # Support both plain state_dict and full training checkpoint
    state_dict = state.get("model_state_dict", state)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def export_onnx(
    checkpoint: Path,
    output: Path,
    model_name: str,
    num_classes: int,
    opset: int,
    max_length: int,
) -> None:
    device = "cpu"
    model = load_model(checkpoint, model_name, num_classes, device)

    # Dummy inputs
    batch = 1
    input_ids = torch.randint(low=0, high=100, size=(batch, max_length), device=device)
    attention_mask = torch.ones((batch, max_length), device=device, dtype=torch.long)

    output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        output.as_posix(),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits", "capsule_outputs"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
            "capsule_outputs": {0: "batch"},
        },
        opset_version=opset,
    )
    print(f"✅ ONNX export complete: {output}")


def quantize_onnx(onnx_path: Path, quantized_path: Path) -> Optional[Path]:
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("⚠️ onnxruntime.quantization not installed; skipping quantization.")
        return None

    quantized_path.parent.mkdir(parents=True, exist_ok=True)
    # Older onnxruntime versions do not support optimize_model/reduce_range args.
    quantize_kwargs = {
        "model_input": onnx_path.as_posix(),
        "model_output": quantized_path.as_posix(),
        "weight_type": QuantType.QInt8,
        "per_channel": False,
    }
    quantize_dynamic(**quantize_kwargs)
    print(f"✅ Quantized model written to: {quantized_path}")
    return quantized_path


def parse_args():
    parser = argparse.ArgumentParser(description="Export MentalRoBERTa-Caps to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--output", type=Path, required=True, help="Output ONNX path")
    parser.add_argument("--model_name", type=str, default="deepset/gbert-base", help="Base HF model name")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of target classes")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (14+ recommended for SDPA)")
    parser.add_argument("--max_length", type=int, default=256, help="Sequence length for dynamic axes")
    parser.add_argument("--quantize", action="store_true", help="Also write an int8 quantized ONNX model")
    return parser.parse_args()


def main():
    args = parse_args()
    export_onnx(
        checkpoint=args.checkpoint,
        output=args.output,
        model_name=args.model_name,
        num_classes=args.num_classes,
        opset=args.opset,
        max_length=args.max_length,
    )

    if args.quantize:
        quant_out = args.output.with_suffix(".int8.onnx")
        quantize_onnx(args.output, quant_out)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"❌ Export failed: {exc}", file=sys.stderr)
        sys.exit(1)
