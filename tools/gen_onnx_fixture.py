#!/usr/bin/env python3
"""Generate the ONNX fixture used by `cv-dnn`'s `from_onnx` test.

Writes a small two-layer convolutional network exported from PyTorch, its
input, and the output PyTorch itself computed. The Rust test then asserts that
loading the `.onnx` through RLX reproduces that output.

    python3 tools/gen_onnx_fixture.py [outdir]        # default /tmp/onnxtest

Requires torch. The fixture is deliberately tiny and seeded, so the values are
reproducible on any machine.
"""
import sys
import pathlib

import numpy as np
import torch
import torch.nn as nn


def main():
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/onnxtest")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    class Small(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(3, 4, 3, padding=1)
            self.c2 = nn.Conv2d(4, 2, 3, padding=1)

        def forward(self, x):
            return self.c2(torch.relu(self.c1(x)))

    model = Small().eval()
    x = torch.arange(1 * 3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8) / 100.0
    torch.onnx.export(
        model,
        (x,),
        str(out / "small.onnx"),
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        dynamo=False,
    )
    with torch.no_grad():
        ref = model(x).numpy()
    np.save(out / "input.npy", x.numpy())
    np.save(out / "expected.npy", ref)
    print(f"wrote small.onnx, input.npy, expected.npy to {out}")
    print(f"  output shape {ref.shape}, first values {ref.ravel()[:4]}")

    # Second opinion: OpenCV's own dnn module on the same file.
    try:
        import cv2

        net = cv2.dnn.readNetFromONNX(str(out / "small.onnx"))
        net.setInput(x.numpy())
        got = net.forward()
        print(f"  cv2.dnn agrees with torch to {np.abs(got - ref).max():.3e}")
    except Exception as e:  # cv2 is optional here
        print(f"  (cv2 cross-check skipped: {e})")


if __name__ == "__main__":
    main()
