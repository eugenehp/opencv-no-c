#!/usr/bin/env python3
"""Generate the video fixtures `cv-videoio`'s tests read.

Writes an MJPEG AVI, an uncompressed AVI and a PNG sequence, plus the pixels
cv2 decodes the MJPEG frames back to.

    python3 tools/gen_video_fixture.py [outdir]     # default /tmp/viotest

A note on the reference, because it is not obvious: the decoded pixels come
from `cv2.imdecode` on each extracted frame, **not** from `cv2.VideoCapture`.
Those two disagree — VideoCapture goes through FFmpeg's MJPEG decoder and
imdecode goes through libjpeg, and on this fixture they differ by up to 29
levels on identical bytes. This port's decoder is libjpeg-shaped, so libjpeg is
the meaningful comparison; measuring against FFmpeg would be measuring the gap
between two of OpenCV's own decoders.
"""
import pathlib
import struct
import sys

import cv2
import numpy as np


def extract_frames(path):
    """Pull the raw `00dc` payloads out of an AVI's `movi` list.

    The search has to stop at the end of `movi`: the `idx1` index that follows
    also contains `00dc` as its per-entry chunk id, and walking into it yields
    garbage "frames" whose sizes come from the index's flags field.
    """
    d = path.read_bytes()
    movi = d.find(b"movi")
    list_start = movi - 8  # back up over 'LIST' + size
    list_size = struct.unpack_from("<I", d, list_start + 4)[0]
    end = list_start + 8 + list_size

    off, out = movi + 4, []
    while off + 8 <= end:
        cid = d[off : off + 4]
        size = struct.unpack_from("<I", d, off + 4)[0]
        if cid[2:4] in (b"dc", b"db") and size:
            out.append(d[off + 8 : off + 8 + size])
        off += 8 + size + (size & 1)
    return out


def main():
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/viotest")
    out.mkdir(parents=True, exist_ok=True)

    frames = []
    for t in range(6):
        f = np.zeros((32, 48, 3), np.uint8)
        f[:, :, 0] = 40 + t * 30
        f[:, :, 1] = 100
        f[:, :, 2] = 200 - t * 20
        cv2.rectangle(f, (4 + t * 4, 4), (12 + t * 4, 12), (255, 255, 255), -1)
        frames.append(f)

    for name, cc in [("mjpg.avi", "MJPG"), ("rgba.avi", "RGBA")]:
        w = cv2.VideoWriter(str(out / name), cv2.VideoWriter_fourcc(*cc), 10.0, (48, 32))
        for f in frames:
            w.write(f)
        w.release()

    for i, f in enumerate(frames):
        cv2.imwrite(str(out / f"seq_{i:03d}.png"), f)

    # The comparison target: libjpeg's decode of each extracted payload.
    payloads = extract_frames(out / "mjpg.avi")
    decoded = [cv2.imdecode(np.frombuffer(p, np.uint8), cv2.IMREAD_COLOR) for p in payloads]
    (out / "frames_libjpeg.raw").write_bytes(np.array(decoded).tobytes())

    # How far apart OpenCV's own two decoders are, for the record.
    cap, ff = cv2.VideoCapture(str(out / "mjpg.avi")), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        ff.append(f)
    cap.release()
    gap = np.abs(np.array(decoded).astype(int) - np.array(ff).astype(int))
    print(f"wrote mjpg.avi, rgba.avi and a png sequence to {out}")
    print(f"  cv2 libjpeg vs cv2 FFMPEG on identical bytes: worst={gap.max()} mean={gap.mean():.3f}")


if __name__ == "__main__":
    main()
