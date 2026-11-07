"""Check every deliverable against its spec by decoding the actual files. Also builds the review previews."""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

out = Path(sys.argv[1])
prev = out / "previews"
prev.mkdir(exist_ok=True)


def probe(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                        "stream=codec_name,profile,width,height,pix_fmt,r_frame_rate,nb_read_packets", "-of", "json", str(p)],
                       capture_output=True, text=True, check=True)
    return json.loads(r.stdout)["streams"][0]


def mov_frame_alpha(p, n):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-vf", f"select=eq(n\\,{n}),alphaextract", "-vframes", "1", "-f", "rawvideo",
                        "-pix_fmt", "gray", "-"], capture_output=True, check=True)
    a = np.frombuffer(r.stdout, np.uint8)
    return int(a.min()), int(a.max())


print("== STINGER")
fs = sorted((out / "stinger" / "png_60fps").glob("*.png"))
print(len(fs), "frames", fs[0].name, "..", fs[-1].name)
mins, maxs, bad = [], [], 0
for f in fs:
    im = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
    if im.shape != (1080, 1920, 4):
        bad += 1
    mins.append(int(im[..., 3].min()))
    maxs.append(int(im[..., 3].max()))
solid = [i for i, m in enumerate(mins) if m == 255]
print("wrong-shape frames:", bad, "| frame 0 max alpha", maxs[0], "| last frame max alpha", maxs[-1])
print("fully opaque:", solid[0], "..", solid[-1], "contiguous:", solid == list(range(solid[0], solid[-1] + 1)))
print(probe(out / "stinger" / "BAPS_Diwali_Stinger_ProRes4444.mov"))

print("== OVERLAY")
for mode in ("IN", "LOOP", "OUT"):
    d = out / "diya_overlay" / f"png_60fps_{mode}"
    fl = sorted(d.glob("*.png"))
    print(mode, len(fl), probe(out / "diya_overlay" / f"Diwali_Diya_Overlay_{mode}_ProRes4444.mov"))
L = sorted((out / "diya_overlay" / "png_60fps_LOOP").glob("*.png"))
rd = lambda p: cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float32)
a0, a1, a2, last = rd(L[0]), rd(L[1]), rd(L[100]), rd(L[-1])
print("loop seam diff (last->first):", round(float(np.abs(last - a0).mean()), 4), "| normal step (0->1):", round(float(np.abs(a1 - a0).mean()), 4))
I = sorted((out / "diya_overlay" / "png_60fps_IN").glob("*.png"))
O = sorted((out / "diya_overlay" / "png_60fps_OUT").glob("*.png"))
print("IN->LOOP join diff:", round(float(np.abs(rd(I[-1]) - a0).mean()), 4), "| LOOP->OUT join diff:", round(float(np.abs(last - rd(O[0])).mean()), 4))
print("IN first frame max alpha", int(rd(I[0])[..., 3].max()), "| OUT last frame max alpha", int(rd(O[-1])[..., 3].max()))

print("== LOWER THIRDS")
movs = sorted((out / "lower_thirds").glob("*.mov"))
print(len(movs), "files,", round(sum(m.stat().st_size for m in movs) / 1e9, 2), "GB")
for m in movs:
    p = probe(m)
    print(m.name, p["codec_name"], p["profile"], p["pix_fmt"], p["r_frame_rate"], p["nb_read_packets"],
          "first", mov_frame_alpha(m, 0)[1], "last", mov_frame_alpha(m, int(p["nb_read_packets"]) - 1)[1], "hold", mov_frame_alpha(m, 300)[1])

print("== BACKGROUNDS / SUPERSOURCE")
for f in sorted((out / "backgrounds").glob("*")):
    if f.suffix == ".png":
        print(f.name, cv2.imread(str(f)).shape)
    else:
        print(f.name, probe(f))
for f in sorted((out / "supersource").glob("*.png")):
    print(f.name, cv2.imread(str(f), cv2.IMREAD_UNCHANGED).shape)
