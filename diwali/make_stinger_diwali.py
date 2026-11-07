"""Diwali stinger: temple curtains closing and opening for darshan. 1920x1080, 60 fps, 180 frames.

The curtains are painted pichwai cloths. Two panels sweep in from the sides with marigold ladis
on their leading edges and fully cover the picture. While closed, the BAPS mark settles on a lotus
medallion and a row of lamps lights from the centre outward. Then the cloth is drawn aside from the foot
upward, so the opening rises like a mandir arch.

    python make_stinger_diwali.py <output_dir> [--test] [--workers N]
"""
import argparse
import math
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from art import (CRIMSON, GOLD, PLUM, Flame, c, cloth, diya, draw_mask, ease_back, ease_io, ease_out, fbm, gold_paint, jaal,
                 load_logo, lotus, over, prog, rangoli_mask, smooth, to_png, WINE, xform, _petal, ROSE, ROSE_DK, MARI_LT)
from scene import bank, paste_c, strand_sprite, garland_row

W, H = 1920, 1080
FPS = 60
N = 180
TW, TH = 1400, H + 80                  # one painted curtain panel; leading edge at x = PAD
PAD = 110                              # room beyond the edge for the hanging marigolds
SUB = 8                                # motion blur samples while the cloth is moving
SHELF = 176                            # dark painted border along the foot of the cloth, where the lamps stand
COVER_FROM, COVER_TO = 54, 118         # frames between which the picture must be fully hidden


def paint_curtain(seed, front=True):
    """One curtain panel as premultiplied RGBA, leading edge on the LEFT at x=PAD. Mirrored for the other side."""
    rng = np.random.default_rng(seed)
    w, h = TW, TH
    if front:
        base = cloth(h, w, seed, scale=1.0)
    else:
        base = cloth(h, w, seed, base=c(120, 8, 62), lift=c(176, 22, 66), scale=1.0)
    img = np.concatenate((base, np.ones((h, w, 1), np.float32)), -1)
    img[:, :PAD] = 0
    if front:
        # gold jaal holding a lotus in every other cell, the kamal talai of a pichwai
        cell = 300
        m = jaal(h, w, cell, 2.0, seed + 1) * 0.34
        m[:, :PAD] = 0
        g = gold_paint(m, seed + 2)
        img = g + img * (1 - g[..., 3:])
        blooms = bank(96, seed + 3, 6, "lotus")
        small = bank(50, seed + 4, 4, "lotus")
        cw = int(cell * 0.66)
        for row in range(-1, h // cell + 2):
            for col in range(-1, w // cw + 2):
                x = PAD + 150 + col * cw + cw * 0.5 + rng.normal(0, 3)
                y = row * cell + cell * 0.5 + rng.normal(0, 3)
                if x < PAD + 150:
                    continue
                paste_c(img, xform(blooms[rng.integers(0, 6)], 1.0 + rng.normal(0, 0.05), rng.random() * 360), x, y)
                paste_c(img, xform(small[rng.integers(0, 4)], 0.9, rng.random() * 360), x + cw * 0.5, y + cell * 0.5)
    else:
        m = jaal(h, w, 240, 1.8, seed + 1) * 0.22
        m[:, :PAD] = 0
        g = gold_paint(m, seed + 2)
        img = g + img * (1 - g[..., 3:])

    if front:
        # pichwai foot border and a slim head border (texture y = frame y + 16)
        fy0 = H + 16 - SHELF
        img[fy0:, PAD:, :3] = c(26, 0, 30) * (0.9 + 0.2 * fbm(h - fy0, w - PAD, 120, seed + 11, 3)[..., None])
        img[16:16 + 40, PAD:, :3] = c(26, 0, 30)
        rule = np.zeros((h, w), np.float32)
        rule[fy0 - 10:fy0, PAD:] = 1
        rule[fy0 + 5:fy0 + 8, PAD:] = 0.9
        rule[16 + 40:16 + 48, PAD:] = 1
        rule[16 + 53:16 + 56, PAD:] = 0.9
        g = gold_paint(rule, seed + 12)
        img = g + img * (1 - g[..., 3:])

    # border along the leading edge: gold bands either side of a run of blooms on dark silk
    bw = 118 if front else 46
    img[:, PAD:PAD + bw, :3] = c(30, 0, 34) if front else c(70, 2, 44)
    band = np.zeros((h, w), np.float32)
    band[:, PAD:PAD + 12] = 1
    band[:, PAD + 16:PAD + 19] = 0.9
    band[:, PAD + bw - 12:PAD + bw] = 1
    band[:, PAD + bw - 19:PAD + bw - 16] = 0.9
    g = gold_paint(np.clip(band, 0, 1), seed + 5)
    img = g + img * (1 - g[..., 3:])
    if front:
        blooms = bank(64, seed + 6, 5, "lotus")
        y = 40
        while y < h:
            paste_c(img, xform(blooms[rng.integers(0, 5)], 1.0, rng.random() * 360), PAD + bw / 2, y)
            y += 72
        # marigold strands hung just past the edge, so the silhouette isn't a clean wipe
        st = strand_sprite(h + 40, 0.95, seed + 8, tassel=False)
        over(img, st, PAD + 2 - st.shape[1] / 2, -10)
        st2 = strand_sprite(h + 40, 0.72, seed + 9, tassel=False)
        over(img, st2, PAD - 32 - st2.shape[1] / 2, -24)
    return img.astype(np.float32)


class Assets:
    def __init__(self):
        self.front = paint_curtain(41, True)
        self.back = paint_curtain(52, False)
        self.medal = self._medallion()
        self.lamps = self._lamps()
        self.fold_a = fbm(H, 2, 620, 7, 2)[:, 0]
        self.fold_b = fbm(H, 2, 300, 8, 2)[:, 0]
        self.xs = np.arange(W, dtype=np.float32) + 0.5
        self.ys = np.arange(H, dtype=np.float32) + 0.5
        self.warm = np.exp(-((self.ys[:, None] - H) / 330) ** 2) * np.ones((H, W), np.float32)

    def _medallion(self):
        """White disc for the logo, in a gold band and a double lotus rim."""
        Rr = 318
        S = (Rr + 120) * 2
        rim = np.zeros((S * 2, S * 2, 4), np.float32)
        cx = S
        rng = np.random.default_rng(3)
        for count, length, width, phase in ((30, 92, 40, 0.5), (30, 70, 36, 0.0)):
            for k in rng.permutation(count):
                a = 2 * math.pi * (k + phase) / count + rng.normal(0, 0.008)
                _petal(rim, cx + (Rr - 12) * 2 * math.cos(a), cx + (Rr - 12) * 2 * math.sin(a), a, length * (1 + rng.normal(0, 0.03)) * 2, width * 2,
                       ROSE_DK, ROSE, c(255, 200, 216), c(150, 16, 84), rng, pointed=0.25)
        yy, xx = np.mgrid[0:S * 2, 0:S * 2].astype(np.float32)
        r = np.hypot(xx - cx, yy - cx) / 2
        band = np.clip(Rr + 1 - r, 0, 1) * np.clip(r - (Rr - 22), 0, 1)
        g = gold_paint(band, 4)
        rim = g + rim * (1 - g[..., 3:])
        disc = np.clip(Rr - 21 - r + 0.5, 0, 1)[..., None]
        col = c(255, 251, 240) + (c(250, 228, 196) - c(255, 251, 240)) * smooth(0.6, 1.0, r / Rr)[..., None]
        rim = np.concatenate((col * disc, disc), -1) + rim * (1 - disc)
        rim = cv2.resize(rim, (S, S), interpolation=cv2.INTER_AREA)
        logo = load_logo(506)
        paste_c(rim, logo, S / 2, S / 2 + 2)
        return rim.astype(np.float32)

    def _lamps(self):
        items = []
        # six each side of the centre seam, standing on the painted foot border
        step = 150
        for i in range(12):
            sign = -1 if i < 6 else 1
            n = (i % 6)
            d = n + 0.9
            size = 130 - n * 3
            spr, (wx, wy) = diya(int(size), 2 + i % 4, sign > 0)
            h, w = spr.shape[:2]
            x = W / 2 + sign * d * step - w * (0.56 if sign > 0 else 0.44)
            y = H - 22 - h
            items.append(dict(spr=spr, x=x, y=y, wx=x + wx, wy=y + wy, flame=Flame(size * 0.36, i + 20, 3.0), d=n))
        return items


_A = None


def _init():
    global _A
    _A = Assets()


def close_pos(t, start, dur, lag):
    """Fraction of the half-frame the leading edge has reached, per row: the foot of the cloth trails the top."""
    v = _A.ys / H
    tt = (t - start - lag * v * dur) / dur
    tt = np.clip(tt, 0, 1)
    # settle: a quick throw, then the cloth eases into place
    return 1 - (1 - tt) ** 3.4


def open_pos(t, start, dur, arch):
    """Drawn aside from the foot upward: the top of the cloth leaves last, so the opening rises as an arch."""
    v = _A.ys / H
    delay = arch * (1 - v) ** 1.5
    tt = np.clip((t - start - delay * dur) / dur, 0, 1)
    return tt * tt * tt * (tt * (6 * tt - 15) + 10)


def curtain_layer(tex, edge, side, phase, gather):
    """Warp one painted cloth so its leading edge follows `edge`, folds deepening where it is gathered.

    edge: per-row distance of the leading edge from the panel's own side of frame, in px.
    side: 0 for the panel entering from the left of frame, 1 from the right.
    """
    xs = _A.xs
    d = edge[:, None] - (xs[None, :] if side == 0 else (W - xs)[None, :])          # px behind the leading edge
    stretch = 1.0 + gather[:, None] * 1.1
    # fold ripple: a cloth is never flat, and rows don't all fold in step
    wav = _A.fold_a[:, None] * 3.2 + _A.fold_b[:, None] * 1.0
    k = 2 * math.pi / 118
    fold = np.sin(d * k * stretch + wav + phase)
    u = d * stretch + (9 + 20 * gather[:, None]) * fold + PAD
    map_x = u.astype(np.float32)
    map_y = (_A.ys[:, None] + 16 + 5 * np.sin(d * 0.012 + phase)).astype(np.float32) * np.ones_like(map_x)
    out = cv2.remap(tex, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # shade the folds: light on the crest, deep in the valley, stronger where gathered
    slope = np.cos(d * k * stretch + wav + phase)
    body = np.clip((d - 8) / 40, 0, 1)
    shade = 1 + (0.10 + 0.34 * gather[:, None]) * slope * body
    sharp = np.clip(-slope, 0, 1) ** 3 * (0.14 + 0.32 * gather[:, None]) * body
    out[..., :3] = np.clip(out[..., :3] * (shade - sharp)[..., None], 0, 1)
    out[..., :3] = np.minimum(out[..., :3], out[..., 3:])
    return out


def frame_at(t):
    """t in frames, may be fractional. Returns premultiplied RGBA."""
    img = np.zeros((H, W, 4), np.float32)
    half = W / 2 + 6      # panels overlap slightly at the centre, so no hairline shows when closed

    for layer, tex, cl_s, cl_d, op_s, op_d, ph in (
        ("back", _A.back, 2, 48, 126, 44, 1.0),
        ("front", _A.front, 8, 46, 116, 50, 2.6),
    ):
        pc = close_pos(t, cl_s, cl_d, 0.22)
        po = open_pos(t, op_s, op_d, 0.42)
        travel_extra = PAD + 70                       # the strands must clear the frame edge too
        closed = pc * (half + travel_extra) - travel_extra
        edge = closed - po * (half + travel_extra + 60)
        gather = np.clip(1 - np.maximum(edge, 0) / half, 0, 1)
        if edge.max() < -PAD - 40:
            continue
        for side in (0, 1):
            ph2 = ph + side * 1.7 + (t / FPS) * 0.8
            cur = curtain_layer(tex, edge, side, ph2, gather)
            img = cur + img * (1 - cur[..., 3:])
            # the top cloth throws a soft shadow onto the one beneath at the centre seam
    # glow of the lamps on the cloth, and the lamps themselves
    light = 0.0
    if 40 <= t <= 140:
        lamp_rise = ease_back(prog(t, 42, 62), 1.2) * (1 - ease_io(prog(t, 112, 128), 2.2))
        out_y = (1 - lamp_rise) * 210
        lit = 0.0
        for it in _A.lamps:
            a = 54 + it["d"] * 5
            power = ease_back(prog(t, a, a + 12), 2.0) * (1 - ease_io(prog(t, 110, 120), 2))
            lit += max(power, 0)
        light = lit / len(_A.lamps)
        if light > 0:
            warm = _A.warm[..., None] * 0.28 * light
            img[..., :3] = np.clip(img[..., :3] + warm * img[..., 3:] * c(255, 130, 34) * (1 - img[..., :3]), 0, 1)
        if lamp_rise > 0:
            for it in _A.lamps:
                over(img, it["spr"], it["x"], it["y"] + out_y)
                a = 54 + it["d"] * 5
                power = ease_back(prog(t, a, a + 12), 2.0) * (1 - ease_io(prog(t, 110, 120), 2))
                if power > 0:
                    f = it["flame"].render(t / FPS, power)
                    over(img, f, it["wx"] - f.shape[1] * 0.5, it["wy"] + out_y - it["flame"].ay)
    # medallion: settles with a slow push, and is gone before the cloth starts to part
    q_in = prog(t, 46, 68)
    q_out = prog(t, 110, 122)
    if 0 < q_in and q_out < 1:
        scale = (0.72 + 0.28 * ease_back(q_in, 1.5)) * (1 + 0.025 * prog(t, 68, 112)) * (1 + 0.10 * ease_io(q_out))
        al = ease_out(q_in * 1.8, 2) * (1 - ease_io(q_out, 2))
        s = xform(_A.medal, scale * 0.74, (1 - ease_out(q_in, 3)) * -24) * al
        paste_c(img, s, W / 2, (H - SHELF) / 2 + 22)
    return img


def render_frame(i):
    moving = i <= 58 or 112 <= i <= 176
    if not moving:
        return frame_at(float(i))
    acc = np.zeros((H, W, 4), np.float32)
    for s in range(SUB):
        acc += frame_at(i + (s / SUB - 0.5) * 0.5)          # 180 degree shutter
    return acc / SUB


def _job(a):
    i, path = a
    im = render_frame(i)
    to_png(im, path, level=5)
    al = im[..., 3]
    return i, float(al.min()), float(al.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    root = Path(a.out) / "stinger"
    folder = root / "png_60fps"
    if a.test:
        _init()
        tdir = Path(a.out) / "_test"
        tdir.mkdir(parents=True, exist_ok=True)
        for i in (20, 38, 66, 96, 134, 150, 164):
            im = render_frame(i)
            to_png(im, tdir / f"st_{i:03d}.png", level=2)
            print(i, im[..., 3].min(), im[..., 3].max(), flush=True)
        return
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    jobs = [(i, folder / f"BAPS_Diwali_Stinger_{i:03d}.png") for i in range(N)]
    stats = {}
    with ProcessPoolExecutor(a.workers, initializer=_init) as ex:
        for i, lo, hi in ex.map(_job, jobs, chunksize=4):
            stats[i] = (lo, hi)
    solid = [i for i, (lo, _) in stats.items() if lo > 0.999]
    print("first/last clear:", stats[0], stats[N - 1])
    print(f"fully opaque frames: {solid[0]}..{solid[-1]} ({len(solid)})")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(folder / "BAPS_Diwali_Stinger_%03d.png"),
                    "-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le", "-alpha_bits", "16", "-vendor", "apl0",
                    str(root / "BAPS_Diwali_Stinger_ProRes4444.mov")], check=True)


if __name__ == "__main__":
    main()
