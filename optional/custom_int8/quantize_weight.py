from __future__ import annotations

import argparse
import json
from pathlib import Path


def quantize_per_channel(weight):
    import torch

    scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127
    quantized = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
    return quantized, scale.to(torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline weight-only INT8 experiment")
    parser.add_argument("--input", type=Path, required=True, help="torch state_dict file")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    state = torch.load(args.input, map_location="cpu")
    output = {}
    summary = {"input": str(args.input), "tensors": 0, "bytes_before": 0, "bytes_after": 0}
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or not tensor.is_floating_point():
            output[name] = tensor
            continue
        quantized, scale = quantize_per_channel(tensor.float())
        output[name + ".int8"] = quantized
        output[name + ".scale"] = scale
        summary["tensors"] += 1
        summary["bytes_before"] += tensor.numel() * tensor.element_size()
        summary["bytes_after"] += quantized.numel() + scale.numel() * 4
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    Path(str(args.output) + ".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

