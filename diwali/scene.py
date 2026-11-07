"""Composed pieces shared by the background, SuperSource, overlay and stinger."""
import math

import cv2
import numpy as np

from art import (BLUSH, CRIMSON, GOLD, MARI_ORANGE, MARI_YELLOW, PLUM, WINE, Flame, cloth, diya, draw_mask, fbm,
                 gold_paint, jaal, leaf, lotus, marigold, over, rangoli_mask, smooth, xform)


def bank(px, seed=0, count=5, kind="lotus"):
    """A few variants of a motif at one pixel size, so a row never repeats exactly."""
    out = []
    big = max(px * 2, 160)        # paint large and reduce, so small blooms stay clean rather than noisy
    for i in range(count):
        if kind == "lotus":
            s = lotus(big, seed + i)
        elif kind == "orange":
            s = marigold(big, seed + i, **MARI_ORANGE)
        else:
            s = marigold(big, seed + i, **MARI_YELLOW)
        out.append(cv2.resize(s, (px, px), interpolation=cv2.INTER_AREA))
    return out


def paste_c(dst, spr, cx, cy):
    return over(dst, spr, cx - spr.shape[1] / 2, cy - spr.shape[0] / 2)


def strand_sprite(height, u, seed=0, flower=None, tassel=True):
    """Ladi: flowers threaded close together on a cord, with a small tassel of petals below.

    Returned hanging from the top-centre of the sprite, width about 60u.
    """
    rng = np.random.default_rng(seed + 40)
    fs = max(10, int(46 * u))
    orange = bank(fs, seed, 4, "orange")
    yellow = bank(fs, seed + 9, 4, "yellow")
    w = int(fs * 2.6)
    h = int(height + fs * 2.6)
    im = np.zeros((h, w, 4), np.float32)
    cx = w / 2
    y = -fs * 0.2
    step = fs * 0.70
    i = 0
    while y < height:
        pal = yellow if (i // 3) % 2 == 0 else orange
        s = pal[rng.integers(0, 4)]
        x = cx + rng.normal(0, fs * 0.035)
        paste_c(im, xform(s, 1.0 + rng.normal(0, 0.05), rng.random() * 360), x, y)
        y += step * (1 + rng.normal(0, 0.05))
        i += 1
    if tassel:
        last = y
        for dx, dy, pal in ((-0.52, 0.30, orange), (0.52, 0.34, orange), (0.0, 0.62, yellow), (0.0, 0.05, yellow)):
            s = pal[rng.integers(0, 4)]
            paste_c(im, xform(s, 0.88, rng.random() * 360), cx + dx * fs, last + dy * fs)
    return im


def garland_row(canvas, y, x0, x1, u, seed=3, size=66, gap=1.34, leaves=True, cord=True):
    """The row of pink blooms from the client's slide, laid by hand: small jitter, leaves between."""
    rng = np.random.default_rng(seed)
    px = max(12, int(size * u))
    flowers = bank(px, seed, 6, "lotus")
    h, w = canvas.shape[:2]
    step = px * gap
    n = int((x1 - x0) / step)
    step = (x1 - x0) / max(n, 1)
    if cord:
        # twisted gold cord the flowers are threaded on, sagging a little between each bloom
        ph = int(px * 2)

        def fn(dr, ss):
            pts = []
            for i in range(n * 12 + 1):
                x = x0 + i * step / 12
                sag = 3.2 * u * abs(math.sin(math.pi * i / 12))
                pts.append((x * ss, (ph / 2 + sag + math.sin(i * 0.31) * 0.8 * u) * ss))
            dr.line(pts, fill=255, width=max(1, round(2.4 * u * ss)), joint="curve")
            for off in (0.5,):                                    # one gold bead between each pair of blooms
                for i in range(n):
                    x = x0 + (i + off) * step
                    r = 3.4 * u
                    cy = ph / 2 + 3.2 * u
                    dr.ellipse(((x - r) * ss, (cy - r) * ss, (x + r) * ss, (cy + r) * ss), fill=255)
        band = draw_mask(w, ph, fn, 2)
        over(canvas, gold_paint(band, seed), 0, int(y - ph / 2))
    lf = [leaf(px * 0.62, px * 0.10, seed + i) for i in range(4)]
    for i in range(n + 1):                                     # leaves first, peeking out between blooms
        x = x0 + i * step
        if leaves and i < n:
            s = lf[rng.integers(0, 4)]
            for sign in (-1, 1):
                ang = 90 + sign * (58 + rng.normal(0, 6))
                ls = xform(s, 1.0, ang if sign > 0 else ang + 180)
                paste_c(canvas, ls, x + step / 2, y + sign * px * 0.20)
    for i in range(n + 1):
        x = x0 + i * step + rng.normal(0, 1.6 * u)
        s = flowers[rng.integers(0, 6)]
        paste_c(canvas, xform(s, 1.0 + rng.normal(0, 0.035), rng.random() * 360), x, y + rng.normal(0, 1.6 * u))
    return canvas


def ground(h, w, u, seed=11, glow=True, rangoli=True):
    """The plum cloth: dyed ground, jaal in gold, a low warm glow as if lit by the lamps."""
    img = cloth(h, w, seed, scale=u)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = (xx - w / 2) / (w / 2), yy / h
    if glow:
        # the crimson-magenta warmth of the reference, pooled low and held back so the slide's plum still reads
        g = np.exp(-((nx / 1.15) ** 2 + ((ny - 1.05) / 0.58) ** 2))[..., None]
        img = img + (WINE - img) * (0.72 * g)
        g2 = np.exp(-((nx / 0.72) ** 2 + ((ny - 1.12) / 0.27) ** 2))[..., None]
        img = img + (CRIMSON - img) * (0.50 * g2)
    v = np.clip(1 - 0.30 * (np.abs(nx) ** 2.6 + np.abs(ny - 0.5) ** 2.6 * 3), 0.55, 1)[..., None]
    img = img * v
    rgba = np.concatenate((img, np.ones((h, w, 1), np.float32)), -1)
    j = jaal(h, w, int(250 * u), 1.5 * u, seed + 1) * 0.15
    # lattice fades towards the centre so copy sits on quiet cloth
    keep = 0.45 + 0.55 * smooth(0.15, 0.95, np.hypot(nx, (ny - 0.5) * 1.6))
    g = gold_paint(j * keep, seed + 2)
    rgba = g + rgba * (1 - g[..., 3:])
    if rangoli:
        size = int(min(h * 0.86, w * 0.6))
        m = rangoli_mask(size, seed + 3, line=1.7 * u) * 0.17
        spr = gold_paint(m, seed + 4)
        paste_c(rgba, spr, w / 2, h * 0.5)
    return rgba.astype(np.float32)


class Lamps:
    """Diyas with live flames placed on a canvas. Draw the clay once, then the flames per frame."""

    def __init__(self, placements, u, loop=200 / 60, seed=0):
        """placements: (centre_x, base_y, width_in_px, flip)."""
        self.items = []
        sizes = {}
        for i, (cx, base, width, flip) in enumerate(placements):
            spr, (wx, wy) = diya(int(width), seed + i % 4, flip)
            h, w = spr.shape[:2]
            x = cx - w * (0.56 if flip else 0.44)
            y = base - h * 0.98
            fl = Flame(width * 0.34, seed + i, loop)
            self.items.append(dict(spr=spr, x=x, y=y, wx=x + wx, wy=y + wy, flame=fl, w=width))

    def clay(self, canvas, dy=0.0):
        for it in self.items:
            over(canvas, it["spr"], it["x"], it["y"] + dy)

    def flames(self, canvas, t, power=None, dy=0.0):
        for i, it in enumerate(self.items):
            p = 1.0 if power is None else power[i]
            if p <= 0:
                continue
            f = it["flame"].render(t, p)
            over(canvas, f, it["wx"] - f.shape[1] * 0.5, it["wy"] + dy - it["flame"].ay)


def scatter_petals(canvas, u, spots, seed=2):
    """Loose marigold and rose petals around the lamps, as they fall on a real shrine floor."""
    rng = np.random.default_rng(seed)
    from art import _petal, ROSE, ROSE_DK, ROSE_LT, MARI, MARI_DK, MARI_LT
    size = int(44 * u) * 2
    tiles = []
    for pal in ((ROSE_DK, ROSE, ROSE_LT, ROSE_DK), (MARI_DK, MARI, MARI_LT, MARI_DK)):
        for _ in range(4):
            t = np.zeros((size, size, 4), np.float32)
            _petal(t, size * 0.15, size * 0.5, 0, size * 0.72, size * 0.18, pal[0], pal[1], pal[2], pal[3], rng, pointed=0.05, veins=False)
            tiles.append(cv2.resize(t, (size // 2, size // 2), interpolation=cv2.INTER_AREA))
    for (x, y, spread, count) in spots:
        for _ in range(count):
            s = tiles[rng.integers(0, len(tiles))]
            px = x + rng.normal(0, spread)
            py = y + rng.normal(0, spread * 0.14)
            paste_c(canvas, xform(s, 0.5 + rng.random() * 0.5, rng.random() * 360), px, py)
    return canvas


def clear_alpha_rgb(img):
    """Opaque cloth ends up with alpha 1; keep it exactly so."""
    img[..., 3] = np.clip(img[..., 3], 0, 1)
    return img
