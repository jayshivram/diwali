"""Name-only animated lower thirds, one ProRes 4444 with alpha per name, for Mitti into the ATEM.

Keeps the geometry of the approved v2 design: a circle overhanging a thin rounded bar that
ends at 45% of the frame, with a gold line under it, and 3s in / 3s on / 3s out where the exit plays
the entry backwards. The diya in the circle stays lit throughout.

    python make_lowerthirds.py <output_dir> <names.txt> [--test] [--workers N] [--fps 60]
"""
import argparse
import math
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from art import (BLUSH, GOLD, PLUM, Flame, c, cloth, diya, draw_mask, ease_back, ease_io, ease_out, fbm, font, gold_paint,
                 jaal, lotus, marigold, over, prog, smooth, to_rgba8, xform, MARI_ORANGE, MARI_YELLOW)
from scene import bank, paste_c

W, H = 1920, 1080
FX0, FY0, FW, FH = 0, 760, 1000, 320         # working rectangle: everything happens inside this
CX, CY = 156, 868                            # circle centre, in frame px; clears the diya overlay beneath
R = 80
BAR_H = 78
BAR_END = 864                                # 45% of the frame width, as approved
BAR_TOP = CY - BAR_H // 2
IN_S, HOLD_S, OUT_S = 3.0, 3.0, 3.0
TEXT_X0 = CX + R + 30
TEXT_X1 = BAR_END - 66
BASE_SIZE = 46
MIN_SIZE = 26


def fx(x):
    return x - FX0


def fy(y):
    return y - FY0


class Parts:
    """Everything that doesn't depend on the name; painted once per worker."""

    def __init__(self):
        self.bar = self._bar()
        self.gold_line = self._gold_line()
        self.ring, self.face = self._medallion()
        self.flame = Flame(50, 3, loop=9.0)
        spr, (wx, wy) = diya(92, 2)
        self.diya, self.wick = spr, (wx, wy)
        self.end_bloom = cv2.resize(lotus(200, 77), (86, 86), interpolation=cv2.INTER_AREA)

    def _bar(self):
        """The thin ribbon: dark silk with a jaal, gold selvedge, rounded end."""
        h, w = BAR_H + 8, BAR_END - CX + 8
        base = cloth(h, w, 31, base=c(58, 2, 58), lift=c(104, 10, 88))
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ny = (yy - h / 2) / (h / 2)
        base = base * (1.14 - 0.38 * np.abs(ny) ** 1.6)[..., None]              # slight roll, like a stitched band
        j = jaal(h, w, 100, 1.1, 5) * 0.22
        g = gold_paint(j, 7)
        rgba = np.concatenate((base, np.ones((h, w, 1), np.float32)), -1)
        rgba = g + rgba * (1 - g[..., 3:])
        sel = np.zeros((h, w), np.float32)
        sel[4:7] = 0.95
        sel[h - 7:h - 4] = 0.95
        g = gold_paint(sel, 9)
        rgba = g + rgba * (1 - g[..., 3:])
        return rgba.astype(np.float32)

    def _gold_line(self):
        """The line that stays under the bar, as on the approved v2: here a marigold thread."""
        w = BAR_END - CX
        m = np.ones((5, w), np.float32)
        col = np.zeros((5, w, 4), np.float32)
        col[..., :3] = c(255, 164, 28)
        col[..., 3] = 1
        col[:, :, :3] *= 0.86 + 0.14 * fbm(5, w, 60, 3, 2)[..., None]
        return col

    def _medallion(self):
        """A petal ring that can spin as it rolls in, and the still face behind the lamp."""
        S = R * 2 + 90
        ss = 2
        n = S * ss
        ring = np.zeros((n, n, 4), np.float32)
        cx = n / 2
        from art import _petal, ROSE, ROSE_DK, c as col
        rng = np.random.default_rng(12)
        # two overlapping tiers of petals, close together, so the rim reads as an open lotus
        for count, length, width, phase in ((20, 30, 14.5, 0.5), (20, 24, 13.0, 0.0)):
            for k in rng.permutation(count):
                a = 2 * math.pi * (k + phase) / count + rng.normal(0, 0.015)
                _petal(ring, cx + (R - 6) * ss * math.cos(a), cx + (R - 6) * ss * math.sin(a), a, length * ss * (1 + rng.normal(0, 0.04)), width * ss,
                       ROSE_DK, ROSE, col(255, 196, 214), col(150, 16, 84), rng, pointed=0.25, veins=True)
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        r = np.hypot(xx - cx, yy - cx) / ss
        face = np.clip(R - r + 0.5, 0, 1)
        # gold band with a bead run, the disc sits on it
        band = np.clip(R + 1 - r, 0, 1) * np.clip(r - (R - 7), 0, 1)
        g = gold_paint(band, 3)
        ring = g + ring * (1 - g[..., 3:])
        ring = cv2.resize(ring, (S, S), interpolation=cv2.INTER_AREA)
        fy_ = (yy - cx) / ss / R
        tone = np.clip(0.3 + 0.8 * fy_, 0, 1)[..., None]
        dark = c(36, 0, 40) * (1 - tone) + c(110, 8, 70) * tone
        inner = np.clip(R - 6 - r + 0.5, 0, 1)[..., None]
        fc = np.concatenate((dark * inner, inner), -1)
        m = jaal(n, n, 90 * ss, 1.0 * ss, 3) * 0.16
        g = gold_paint(m * inner[..., 0], 2)
        fc = g + fc * (1 - g[..., 3:])
        fc = cv2.resize(fc, (S, S), interpolation=cv2.INTER_AREA)
        return ring.astype(np.float32), fc.astype(np.float32)


def layout_name(name):
    """Choose a size so the name fits the fixed bar (the bar never changes), centred in the text area."""
    room = TEXT_X1 - TEXT_X0
    size = BASE_SIZE
    while size > MIN_SIZE and font(size).getlength(name) > room:
        size -= 1
    f = font(size)
    total = f.getlength(name)
    x0 = TEXT_X0 + (room - total) / 2
    ss = 3
    fss = font(size * ss)
    letters = []
    asc, desc = fss.getmetrics()
    hh = asc + desc + 8 * ss
    for i, ch in enumerate(name):
        if ch == " ":
            continue
        adv = fss.getlength(name[:i + 1]) - fss.getlength(ch)      # kerned position of this glyph
        ww = int(fss.getlength(ch)) + 10 * ss
        m = Image.new("L", (ww, hh), 0)
        ImageDraw.Draw(m).text((4 * ss, 2 * ss), ch, font=fss, fill=255)
        arr = np.asarray(m.resize((ww // ss, hh // ss), Image.LANCZOS), np.float32) / 255.0
        lh, lw = arr.shape
        shade = np.linspace(0, 1, lh, dtype=np.float32)[:, None]
        col = (c(255, 240, 222) * (1 - shade[..., None]) + c(246, 206, 158) * shade[..., None]) * arr[..., None]
        spr = np.concatenate((col, arr[..., None]), -1).astype(np.float32)
        sh = cv2.GaussianBlur(arr, (0, 0), 1.6) * 0.7
        letters.append(dict(spr=spr, shadow=sh, x=x0 + adv / ss - 4, i=i))
    # vertical centre uses the name's real ink so names with and without descenders still sit right
    box = font(size).getbbox("Hxg")
    y0 = BAR_TOP + BAR_H / 2 - (box[1] + box[3]) / 2 - 2
    return dict(size=size, letters=letters, y0=y0, n=len(name))


def timeline(t):
    """Progress of the choreography, 0 to 1: forward on the way in, mirrored on the way out."""
    if t < IN_S:
        return t
    if t < IN_S + HOLD_S:
        return IN_S
    return max(IN_S - (t - IN_S - HOLD_S), 0.0)


def draw_frame(p, name_lay, t):
    img = np.zeros((FH, FW, 4), np.float32)
    k = timeline(t)                         # seconds along the entry, 0..3
    if k <= 0:
        return img

    # 1. The bar pushes out from behind the circle
    e = ease_io(prog(k, 0.85, 1.85), 3.2)
    if e > 0:
        bar_w = (BAR_END - CX) * e
        bar = p.bar
        h = bar.shape[0]
        xe = CX + bar_w
        rnd = BAR_H / 2
        top = BAR_TOP - 4
        mask = np.zeros((FH, FW), np.float32)
        yy = (np.arange(h, dtype=np.float32) + 0.5 - h / 2)
        half = np.sqrt(np.clip(rnd ** 2 - yy ** 2, 0, None))
        limit = xe - rnd + half                                      # rounded end
        xs = np.arange(FW, dtype=np.float32) + 0.5 + FX0
        cov = np.clip(limit[:, None] - xs[None, :] + 0.5, 0, 1) * (np.abs(yy) <= rnd + 0.5)[:, None]
        cov = cov * (xs[None, :] > CX - 2)
        # the pattern is fixed on the cloth and uncovered by the moving end
        y0, x0 = fy(top), fx(CX)
        region = img[y0:y0 + h, x0:x0 + bar.shape[1]]
        c_ = cov[:, x0:x0 + bar.shape[1]][..., None]
        spr = bar[:, :region.shape[1]] * c_
        # soft drop shadow on the picture beneath
        sh_full = np.zeros((FH, FW), np.float32)
        shy0 = y0 + 7
        sh_full[shy0:shy0 + h] = cov
        sh_full = cv2.GaussianBlur(sh_full, (0, 0), 9) * 0.40
        img[..., 3] = sh_full + img[..., 3] * (1 - sh_full)
        img[..., :3] = c(10, 0, 12) * sh_full[..., None] + img[..., :3] * (1 - sh_full[..., None])
        region = img[y0:y0 + h, x0:x0 + bar.shape[1]]
        region[:] = spr + region * (1 - spr[..., 3:])
        # the thread under the bar draws on just after the bar
        g = ease_io(prog(k, 1.05, 2.0), 2.2)
        if g > 0:
            lw = int(((BAR_END - rnd * 0.55) - CX) * g)
            yl = fy(BAR_TOP + BAR_H + 3)
            line = p.gold_line[:, :lw].copy()
            fade = np.clip((lw - np.arange(lw, dtype=np.float32)) / 18, 0, 1)
            line = line * fade[None, :, None]
            img[yl:yl + 5, x0:x0 + lw] = line + img[yl:yl + 5, x0:x0 + lw] * (1 - line[..., 3:])

    # 2. Name, letter by letter, sliding from the left behind a soft mask (keeps inside the bar)
    n = name_lay["n"]
    for ly in name_lay["letters"]:
        start = 1.62 + 0.70 * ly["i"] / max(n - 1, 1)
        q = prog(k, start, start + 0.52)
        if q <= 0:
            continue
        o = ease_out(q, 3.2)
        dx = (1 - o) * -26
        x = ly["x"] + dx
        al = min(q * 2.2, 1.0)
        xi = int(math.floor(x))
        frac = x - xi
        M = np.float32([[1, 0, frac], [0, 1, 0]])
        s = cv2.warpAffine(ly["spr"], M, (ly["spr"].shape[1] + 1, ly["spr"].shape[0]), flags=cv2.INTER_LINEAR) * al
        y = name_lay["y0"]
        over(img, s, fx(xi), fy(y))

    # 3. Shimmer across the name once it has landed
    sweep = prog(k, 2.45, 2.98)
    if 0 < sweep < 1:
        xc = TEXT_X0 + (TEXT_X1 - TEXT_X0 + 120) * ease_io(sweep, 1.8) - 60
        y0, y1 = fy(BAR_TOP + 8), fy(BAR_TOP + BAR_H - 8)
        xs = np.arange(FW, dtype=np.float32) + FX0 + 0.5
        gl = np.exp(-((xs - xc) / 46) ** 2) * math.sin(sweep * math.pi)
        reg = img[y0:y1, :]
        reg[..., :3] += (gl * 0.24)[None, :, None] * reg[..., 3:] * c(255, 220, 160) * (reg[..., :3].sum(-1, keepdims=True) > 1.2)

    # 4. A bloom set on the bar's end, opening as the bar arrives
    b = ease_back(prog(k, 1.62, 2.1), 1.9)
    if b > 0:
        xe = BAR_END - 22
        s = xform(p.end_bloom, max(b, 0.02), (1 - b) * -110)
        paste_c(img, s, fx(xe), fy(CY))

    # 5. Circle: rolls in from the left like a wheel, with a small overshoot
    r = ease_back(prog(k, 0.0, 0.85), 1.35)
    if r > 0:
        x = -R * 1.8 + (CX + R * 1.8) * r
        turn = (1 - r) * -300
        ring = xform(p.ring, 1.0, turn)
        sh = cv2.GaussianBlur(ring[..., 3], (0, 0), 8) * 0.42
        cx, cy = fx(x), fy(CY) + 6
        sh_rgba = np.concatenate((c(10, 0, 12) * sh[..., None], sh[..., None]), -1)
        paste_c(img, sh_rgba, cx, cy)
        paste_c(img, p.face, fx(x), fy(CY))
        # lamp fades in and the flame takes
        q = ease_out(prog(k, 0.55, 1.0), 2.6)
        if q > 0:
            sc = 0.86 + 0.14 * q
            spr = xform(p.diya * q, sc, 0)
            sx = fx(x) - spr.shape[1] * 0.50
            sy = fy(CY) - spr.shape[0] * 0.32 + 14
            over(img, spr, sx, sy)
            wick_x = sx + p.wick[0] * spr.shape[1] / p.diya.shape[1]
            wick_y = sy + p.wick[1] * spr.shape[0] / p.diya.shape[0]
            pwr = ease_back(prog(k, 0.8, 1.25), 2.0)
            if pwr > 0:
                fl = p.flame.render(t % 9.0, pwr)
                # keep the halo inside the medallion
                over(img, fl, wick_x - fl.shape[1] * 0.5, wick_y - p.flame.ay)
        paste_c(img, ring, fx(x), fy(CY))
    return img


_P = None
_LAY = {}


def _init():
    global _P
    _P = Parts()


def _render_chunk(args):
    name, i0, i1, fps = args
    lay = _LAY.get(name) or layout_name(name)
    _LAY[name] = lay
    out = []
    for i in range(i0, i1):
        out.append(to_rgba8(draw_frame(_P, lay, i / fps)))
    return i0, np.stack(out)


def safe(name):
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def encode(name, out_dir, fps, frames_iter):
    path = out_dir / f"LT_{safe(name)}.mov"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le", "-alpha_bits", "16", "-vendor", "apl0",
           "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", str(path)]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    full = np.zeros((H, W, 4), np.uint8)
    for fr in frames_iter:
        full[:] = 0
        full[FY0:FY0 + FH, FX0:FX0 + FW] = fr
        pr.stdin.write(full.tobytes())
    pr.stdin.close()
    if pr.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on {name}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("names")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--fps", type=int, default=60)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "lower_thirds").mkdir(parents=True, exist_ok=True)
    names = [ln.strip() for ln in Path(a.names).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if a.test:
        p = Parts()
        tdir = out / "_test"
        tdir.mkdir(exist_ok=True)
        lay = layout_name(names[0])
        print(names[0], "size", lay["size"])
        for s in (0.25, 0.7, 1.1, 1.5, 1.9, 2.3, 3.5):
            fr = draw_frame(p, lay, s)
            bg = np.zeros((FH, FW, 4), np.float32)
            bg[..., :3], bg[..., 3] = c(70, 90, 110), 1
            res = fr + bg * (1 - fr[..., 3:])
            cv2.imwrite(str(tdir / f"lt_{s:.2f}.png"), cv2.cvtColor(np.clip(res[..., :3] * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        for nm in names:
            print(f"{layout_name(nm)['size']:3d}  {nm}")
        return
    total = round((IN_S + HOLD_S + OUT_S) * a.fps)
    chunk = 24
    with ProcessPoolExecutor(a.workers, initializer=_init) as ex:
        for nm in names:
            jobs = [(nm, i, min(i + chunk, total), a.fps) for i in range(0, total, chunk)]

            def frames():
                for _, block in ex.map(_render_chunk, jobs):
                    yield from block

            path = encode(nm, out / "lower_thirds", a.fps, frames())
            print(path.name, f"{path.stat().st_size / 1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
