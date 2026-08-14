#!/usr/bin/env python3
"""Generate the baseline-JPEG fixtures used by `cv-imgcodecs`' decoder tests.

Writes a greyscale and a colour JPEG plus the raw pixels cv2 decodes them to,
so the Rust test can assert its own decode lands close to OpenCV's.

    python3 tools/gen_jpeg_fixture.py [outdir]      # default /tmp/jpegtest
"""
import pathlib
import sys

import cv2
import numpy as np


def main():
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/jpegtest")
    out.mkdir(parents=True, exist_ok=True)

    # Smooth content: quantisation hurts sharp edges far more, and the point is
    # to test the decoder, not to measure JPEG's loss.
    yy, xx = np.mgrid[0:32, 0:32]
    gray = (128 + 100 * np.sin(xx * 0.2) * np.cos(yy * 0.15)).clip(0, 255).astype(np.uint8)
    color = np.dstack([
        (128 + 100 * np.sin(xx * 0.2)).clip(0, 255),
        (128 + 100 * np.cos(yy * 0.15)).clip(0, 255),
        (128 + 80 * np.sin((xx + yy) * 0.1)).clip(0, 255),
    ]).astype(np.uint8)  # BGR, cv2's order

    q95 = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    cv2.imwrite(str(out / "gray.jpg"), gray, q95)
    cv2.imwrite(str(out / "color.jpg"), color, q95)
    # 4:2:0 chroma subsampling, the common camera default.
    cv2.imwrite(
        str(out / "subsampled.jpg"),
        color,
        [int(cv2.IMWRITE_JPEG_QUALITY), 90, int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR), cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420],
    )

    # What cv2 itself decodes them back to — the comparison target.
    (out / "gray.raw").write_bytes(cv2.imread(str(out / "gray.jpg"), cv2.IMREAD_GRAYSCALE).tobytes())
    (out / "color.raw").write_bytes(cv2.imread(str(out / "color.jpg"), cv2.IMREAD_COLOR).tobytes())
    print(f"wrote gray.jpg, color.jpg, subsampled.jpg and their decoded pixels to {out}")


if __name__ == "__main__":
    main()
