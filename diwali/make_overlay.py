"""Animated diya overlay: a deepmala (row of lights) along the foot of the frame, plus corner marigolds.

Deepavali literally means a row of lights, so that is the whole idea. Three clips, all 1920x1080p60
with alpha, sharing one flame timeline so they join without a jump:

  IN    lamps rise into place one by one from the centre, and each flame takes (90 frames)
  LOOP  flames only; the last frame runs straight back into the first (200 frames)
  OUT   flames dip and go out, lamps sink away from the outside in (90 frames)

    python make_overlay.py <output_dir> [--test] [--workers N]
"""
import argparse
import math
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from art import c, ease_back, ease_io, ease_out, over, prog, to_png
from scene import Lamps, scatter_petals, strand_sprite

W, H = 1920, 1080
FPS = 60
LOOP_N, IN_N, OUT_N = 200, 90, 90
LOOP_S = LOOP_N / FPS
COUNT = 11                # odd, so one lamp stands exactly on the centre line
BASE_Y = 1062
STEP = 172


def places():
    out = []
    mid = COUNT // 2
    for i in range(COUNT):
        d = i - mid
        size = 118 - abs(d) * 2 + (6 if d == 0 else 0)
        out.append((W / 2 + d * STEP, BASE_Y - (mid - abs(d)) * 0.8, size, d > 0))
    return out


class Scene:
    def __init__(self):
        self.lamps = Lamps(places(), 1.0, loop=LOOP_S, seed=5)
        self.floor = np.zeros((H, W, 4), np.float32)
        # low warm pool of lamplight along the foot of the frame, only as much as a key can hold cleanly
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        p = np.exp(-((yy - 1080) / 84) ** 2) * 0.24
        self.floor[..., :3] = c(190, 62, 14) * p[..., None]
        self.floor[..., 3] = p
        self.petals = np.zeros((H, W, 4), np.float32)
        scatter_petals(self.petals, 1.0, [(W / 2 + (i - 5) * STEP + STEP / 2, BASE_Y + 4, 40, 3) for i in range(10)], seed=9)
        strands = np.zeros((H, W, 4), np.float32)
        for x, length, s in ((46, 300, 11), (98, 196, 12), (W - 46, 300, 13), (W - 98, 196, 14)):
            st = strand_sprite(length, 0.86, s)
            over(strands, st, x - st.shape[1] / 2, 0)
        self.strands = strands


_S = None


def _init():
    global _S
    _S = Scene()


def render(s, mode, i):
    """mode in, loop, out. Flame time is continuous: IN ends where LOOP begins, OUT begins where LOOP ends."""
    mid = COUNT // 2
    n = COUNT
    img = np.zeros((H, W, 4), np.float32)
    if mode == "loop":
        t = (i / FPS) % LOOP_S
        rise = [0.0] * n
        power = [1.0] * n
        vis = 1.0
    elif mode == "in":
        t = ((i - IN_N) / FPS) % LOOP_S
        k = i / (IN_N - 1)
        rise, power = [], []
        for j in range(n):
            d = abs(j - mid) / mid
            a = 0.46 * d
            q = ease_back(prog(k, a, a + 0.34), 1.3)
            rise.append((1 - q) * 170)
            f = (prog(k, a + 0.22, a + 0.50))
            power.append(ease_back(f, 2.4) if f > 0 else 0.0)
        vis = ease_out(prog(k, 0, 0.5), 2)
    else:
        t = (i / FPS) % LOOP_S
        k = i / (OUT_N - 1)
        rise, power = [], []
        for j in range(n):
            d = 1 - abs(j - mid) / mid
            a = 0.46 * d
            f = 1 - ease_io(prog(k, a * 0.6, a * 0.6 + 0.26), 2)
            power.append(f)
            q = ease_io(prog(k, a + 0.16, a + 0.50), 2.6)
            rise.append(q * 170)
        vis = 1 - ease_io(prog(k, 0.55, 1.0), 2)
    over(img, s.floor * vis, 0, 0)
    over(img, s.petals * vis, 0, 0)
    drop = (1 - vis) * -360
    img2 = np.zeros((H, W, 4), np.float32)
    over(img2, s.strands, 0, drop)
    img = img2 + img * (1 - img2[..., 3:])
    for j, it in enumerate(s.lamps.items):
        over(img, it["spr"], it["x"], it["y"] + rise[j])
        if power[j] > 0:
            f = it["flame"].render(t, power[j])
            over(img, f, it["wx"] - f.shape[1] * 0.5, it["wy"] + rise[j] - it["flame"].ay)
    return img


def _job(a):
    mode, i, path = a
    to_png(render(_S, mode, i), path, level=4)
    return i


def encode(folder, stem, out):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(folder / f"{stem}_%03d.png"),
           "-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le", "-alpha_bits", "16", "-vendor", "apl0",
           "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", str(out)]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    root = Path(a.out) / "diya_overlay"
    root.mkdir(parents=True, exist_ok=True)
    if a.test:
        s = Scene()
        tdir = Path(a.out) / "_test"
        tdir.mkdir(exist_ok=True)
        for mode, i in (("in", 40), ("loop", 30), ("out", 50)):
            to_png(render(s, mode, i), tdir / f"ov_{mode}.png")
        return
    jobs_all = []
    for mode, n in (("in", IN_N), ("loop", LOOP_N), ("out", OUT_N)):
        folder = root / f"png_60fps_{mode.upper()}"
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        stem = f"Diwali_Diya_Overlay_{mode.upper()}"
        jobs_all += [(mode, i, folder / f"{stem}_{i:03d}.png") for i in range(n)]
    with ProcessPoolExecutor(a.workers, initializer=_init) as ex:
        for _ in ex.map(_job, jobs_all, chunksize=8):
            pass
    for mode in ("in", "loop", "out"):
        stem = f"Diwali_Diya_Overlay_{mode.upper()}"
        encode(root / f"png_60fps_{mode.upper()}", stem, root / f"{stem}_ProRes4444.mov")
        print(stem, flush=True)


if __name__ == "__main__":
    main()
