"""Painted motifs for the Diwali broadcast pack.

Everything is drawn from code rather than stock clip art, in the manner of a
pichwai temple cloth: pink lotus, gold leaf and marigold on a dark ground, with
hand wobble and brush texture so nothing lands on a perfect grid.

All images are float32 PREMULTIPLIED RGBA in 0..1 unless noted.
"""
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
ROOT = HERE.parent
LOGO = ROOT / "Baps_logo.svg.webp"
FONT = HERE / "fonts" / "Eczar[wght].ttf"


def c(r, g, b):
    return np.array([r, g, b], np.float32) / 255.0


PLUM = c(41, 0, 44)            # sampled from the client's slide
PLUM_LIFT = c(74, 8, 72)
WINE = c(120, 6, 72)
CRIMSON = c(176, 16, 62)
BLUSH = c(250, 230, 210)       # slide title colour
GOLD_DK, GOLD, GOLD_LT = c(150, 98, 28), c(226, 172, 74), c(255, 232, 170)
MARI_DK, MARI, MARI_LT = c(176, 66, 0), c(244, 134, 6), c(255, 198, 36)
LEAF_DK, LEAF, LEAF_LT = c(12, 58, 40), c(30, 104, 62), c(106, 160, 70)
ROSE_DK, ROSE, ROSE_LT, ROSE_TIP = c(138, 8, 72), c(226, 52, 124), c(248, 134, 172), c(255, 226, 232)


def smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def ease_out(x, p=3.0):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** p


def ease_io(x, p=3.0):
    x = min(max(x, 0.0), 1.0)
    return 0.5 * (2 * x) ** p if x < 0.5 else 1 - 0.5 * (2 * (1 - x)) ** p


def ease_back(x, s=1.5):
    x = min(max(x, 0.0), 1.0) - 1
    return x * x * ((s + 1) * x + s) + 1


def prog(t, a, b):
    return min(max((t - a) / (b - a), 0.0), 1.0)


def fbm(h, w, cell, seed=0, octaves=4, gain=0.5):
    """Fractal value noise in 0..1. `cell` is the coarsest feature size in px."""
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for _ in range(octaves):
        gh, gw = max(2, int(h / cell) + 2), max(2, int(w / cell) + 2)
        grid = rng.random((gh, gw), np.float32)
        out += amp * cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= gain
        cell = max(cell / 2, 1.5)
    out /= total
    lo, hi = np.percentile(out, (0.5, 99.5))
    return np.clip((out - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)


def wobble(img, amount, cell, seed=0):
    """Displace an image slightly so ruled lines look drawn by hand."""
    h, w = img.shape[:2]
    dx = (fbm(h, w, cell, seed, 3) - 0.5) * 2 * amount
    dy = (fbm(h, w, cell, seed + 1, 3) - 0.5) * 2 * amount
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return cv2.remap(img, xx + dx, yy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def over(dst, spr, x, y):
    """Composite premultiplied sprite onto dst, its top-left at integer x, y."""
    x, y = int(round(x)), int(round(y))
    h, w = spr.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, dst.shape[1]), min(y + h, dst.shape[0])
    if x0 >= x1 or y0 >= y1:
        return dst
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x]
    dst[y0:y1, x0:x1] = s + dst[y0:y1, x0:x1] * (1 - s[..., 3:])
    return dst


def add_light(dst, spr, x, y):
    """Screen-style light: brightens colour, and raises alpha only as much as the light needs."""
    x, y = int(round(x)), int(round(y))
    h, w = spr.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, dst.shape[1]), min(y + h, dst.shape[0])
    if x0 >= x1 or y0 >= y1:
        return dst
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x]
    d = dst[y0:y1, x0:x1]
    d[..., :3] = 1 - (1 - np.clip(d[..., :3], 0, 1)) * (1 - s[..., :3])
    d[..., 3] = np.maximum(d[..., 3], np.clip(d[..., :3].max(-1), 0, 1) * (d[..., 3] > 0.999) + s[..., 3])
    return dst


def xform(spr, scale=1.0, angle=0.0):
    """Scale and rotate a premultiplied sprite about its centre, growing the canvas to fit."""
    h, w = spr.shape[:2]
    a = math.radians(angle)
    nw = int(math.ceil((abs(w * math.cos(a)) + abs(h * math.sin(a))) * scale)) + 2
    nh = int(math.ceil((abs(w * math.sin(a)) + abs(h * math.cos(a))) * scale)) + 2
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    if scale < 0.6:
        # area-average first so tiny petals don't alias, then rotate
        small = cv2.resize(spr, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        return xform(small, 1.0, angle)
    # linear, not cubic: cubic overshoot on premultiplied art leaves white flecks along edges
    out = np.clip(cv2.warpAffine(spr, m, (nw, nh), flags=cv2.INTER_LINEAR), 0, 1)
    out[..., :3] = np.minimum(out[..., :3], out[..., 3:])
    return out


# ---------------------------------------------------------------- petals


def _petal(canvas, cx, cy, angle, length, width, base_d, base_m, tip, line, rng,
           pointed=0.5, veins=True, shade=None, ruffle=0.0):
    """Paint one petal from (cx, cy) outward along `angle` (radians)."""
    ext = int(length * 1.1 + width + 4)
    h, w = canvas.shape[:2]
    x0, x1 = max(int(cx - ext), 0), min(int(cx + ext), w)
    y0, y1 = max(int(cy - ext), 0), min(int(cy + ext), h)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    ca, sa = math.cos(angle), math.sin(angle)
    u = (dx * ca + dy * sa) / length
    v = (-dx * sa + dy * ca)
    s = np.clip(u, 0, 1)
    # widest above the middle; `pointed` runs from a round spoon (0) to a sharp tip (1)
    spoon = s ** 0.55 * np.clip(1 - s ** 4.5, 0, 1) ** 0.5
    sharp = np.clip(np.sin(np.pi * s ** 0.72), 0, 1) ** 0.9
    prof = spoon * (1 - pointed) + sharp * pointed
    if ruffle:
        prof = prof * (1 + ruffle * np.sin(s * 19 + rng.random() * 6))
    half = width * prof + 1e-3
    d = half - np.abs(v)
    cov = np.clip(d + 0.5, 0, 1) * (u > 0) * (u < 1.02)
    if not cov.any():
        return
    t = s
    col = base_d[None, None] * (1 - smooth(0, 0.5, t))[..., None]
    col = col + base_m[None, None] * (smooth(0, 0.5, t) * (1 - smooth(0.5, 1, t)))[..., None]
    col = col + tip[None, None] * smooth(0.5, 1, t)[..., None]
    across = np.clip(np.abs(v) / half, 0, 1)
    col = col * (1 - 0.28 * across[..., None] ** 2.2)                         # roll off to the edges
    if veins:
        k = 6 + rng.random() * 3
        st = 0.5 + 0.5 * np.sin(v / half * k * np.pi + rng.random() * 3)
        col = col * (1 - 0.10 * (st ** 3)[..., None] * (1 - t[..., None] * 0.6))
        mid = np.exp(-(v / (0.9 + width * 0.03)) ** 2) * smooth(0.1, 0.35, t) * (1 - smooth(0.7, 0.95, t))
        col = col + (np.array([1, 0.92, 0.92], np.float32) - col) * (0.16 * mid)[..., None]   # fine pale centre line
    if shade is not None:
        col = col * shade
    edge = np.clip(1 - d / 1.6, 0, 1)
    col = col + (line - col) * (0.55 * edge)[..., None]
    a = cov[..., None]
    # soft contact shadow on what lies beneath
    sh = np.clip(1 - np.maximum(-d, 0) / max(width * 0.35, 2), 0, 1) ** 2 * 0.26
    dst = canvas[y0:y1, x0:x1]
    # warm shadow: red holds up, so yellows go burnt orange rather than olive and pinks stay pink
    s3 = (sh * (1 - cov))[..., None] * np.array([0.35, 1.0, 0.8], np.float32)
    dst[..., :3] *= 1 - s3
    dst[:] = np.concatenate((col * a, a), -1) + dst * (1 - a)


def lotus(size=256, seed=0, rings=((12, 1.00, 0.27), (10, 0.76, 0.25), (8, 0.52, 0.21)),
          base=ROSE_DK, mid=ROSE, tip=c(255, 196, 214), line=c(150, 16, 84)):
    """Top-down pink bloom after the client's flower row, painted with pichwai fine lines."""
    rng = np.random.default_rng(seed)
    ss = 2
    n = size * ss
    im = np.zeros((n, n, 4), np.float32)
    cx = cy = n / 2
    R = n * 0.47
    for ri, (count, rl, rw) in enumerate(rings):
        off = rng.random() * 6.28
        order = rng.permutation(count)
        for k in order:
            ang = off + 2 * math.pi * k / count + rng.normal(0, 0.035)
            L = R * rl * (1 + rng.normal(0, 0.035))
            W = R * rw * (1 + rng.normal(0, 0.06))
            tone = 1 + rng.normal(0, 0.04)
            shade = np.array([tone, tone, tone], np.float32)
            b = base * (0.85 + 0.12 * ri)
            _petal(im, cx, cy, ang, L, W, b, mid, tip, line, rng, pointed=0.16, shade=shade)
    # heart: dark throat, stamens, seed dots
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    r = np.hypot(xx - cx, yy - cy)
    hr = R * 0.17
    throat = np.clip(R * 0.27 - r, 0, 1) * smooth(R * 0.27, R * 0.10, r)
    im[..., :3] = im[..., :3] * (1 - 0.45 * throat[..., None])
    disc = np.clip(hr - r + 0.5, 0, 1)[..., None]
    col = MARI_LT[None, None] * (1 - smooth(0, hr, r))[..., None] + MARI[None, None] * smooth(0, hr, r)[..., None]
    im = np.concatenate((col * disc, disc), -1) + im * (1 - disc)
    for _ in range(26):
        a, rr = rng.random() * 6.28, hr * (0.45 + 0.75 * rng.random())
        x, y = cx + rr * math.cos(a), cy + rr * math.sin(a)
        dot = np.clip(R * 0.018 - np.hypot(xx - x, yy - y) + 0.5, 0, 1)[..., None]
        im = np.concatenate((GOLD_LT[None, None] * dot, dot), -1) + im * (1 - dot)
    return cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)


MARI_YELLOW = dict(light=c(255, 222, 70), mid=c(252, 176, 22), deep=c(196, 92, 0))
MARI_ORANGE = dict(light=c(255, 178, 34), mid=c(246, 122, 6), deep=c(170, 54, 0))


def marigold(size=128, seed=0, light=MARI_ORANGE["light"], mid=MARI_ORANGE["mid"], deep=MARI_ORANGE["deep"]):
    """Genda: a dense ruffled ball, lit from upper left."""
    rng = np.random.default_rng(seed + 100)
    ss = 2
    n = size * ss
    im = np.zeros((n, n, 4), np.float32)
    cx = cy = n / 2
    R = n * 0.47
    line = deep * 1.05
    rings = ((22, 1.0), (22, 0.90), (20, 0.78), (18, 0.66), (16, 0.54), (13, 0.42), (10, 0.30), (6, 0.18))
    for ring, (count, rr) in enumerate(rings):
        off = rng.random() * 6.28
        for k in rng.permutation(count):
            ang = off + 2 * math.pi * k / count + rng.normal(0, 0.07)
            L = R * rr * (1 + rng.normal(0, 0.035))
            W = R * (0.205 - 0.008 * ring) * (1 + rng.normal(0, 0.12))
            tone = 0.94 + rng.random() * 0.16
            _petal(im, cx, cy, ang, L, W, mid * 0.86, mid, light, line, rng, pointed=0.0,
                   veins=False, shade=np.array([tone, tone, tone], np.float32), ruffle=0.10)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    nx, ny = (xx - cx) / R, (yy - cy) / R
    z = np.sqrt(np.clip(1 - nx ** 2 - ny ** 2, 0, 1))
    lit = np.clip(0.55 + 0.62 * (nx * -0.42 + ny * -0.55 + z * 0.72), 0.62, 1.12)
    im[..., :3] = np.clip(im[..., :3] * lit[..., None], 0, im[..., 3:])
    return cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)


def leaf(length=180, width=46, seed=0, dark=LEAF_DK, mid=LEAF, light=LEAF_LT):
    """Mango leaf for torans. Stem at the left, tip at the right."""
    rng = np.random.default_rng(seed + 200)
    ss = 2
    L, W = int(length * ss), int(width * ss)
    h = int(W * 2.5)
    w = L + 8 * ss
    im = np.zeros((h, w, 4), np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    s = np.clip((xx - 4) / L, 0, 1)
    bend = (np.sin(s * math.pi) * 0.22 * W) * (1 if rng.random() > 0.5 else -1)
    v = yy - h / 2 - bend
    half = W * np.clip(np.sin(np.pi * s ** 0.82), 0, 1) ** 0.78 + 1e-3
    d = half - np.abs(v)
    cov = np.clip(d + 0.5, 0, 1) * (xx > 4)
    sidedark = np.where(v > 0, 0.76, 1.0)
    col = mid[None, None] * (1 - s[..., None] * 0.15) + (light - mid)[None, None] * (0.55 * s ** 2 * (np.abs(v) / half) ** 2)[..., None]
    col = col * sidedark[..., None]
    veins = np.abs(np.sin((s * 24 - np.abs(v) / half * 2.2) * math.pi))
    col = col + (light - col) * ((veins > 0.965) * 0.24)[..., None]
    rib = np.exp(-(v / (1.4 * ss)) ** 2)
    col = col + (light * 1.18 - col) * (0.72 * rib)[..., None]
    col = col + (dark - col) * (0.8 * np.clip(1 - d / (1.2 * ss), 0, 1))[..., None]
    a = cov[..., None]
    im = np.concatenate((col * a, a), -1)
    return cv2.resize(im, (w // ss, h // ss), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- gold + strokes


def gold_paint(mask, seed=0):
    """Fill a 0..1 mask with burnished gold leaf: patchy, speckled, never a flat tint."""
    h, w = mask.shape
    n = fbm(h, w, max(h, w) / 8, seed, 4)
    f = fbm(h, w, 5, seed + 7, 2)
    t = np.clip(0.2 + 0.75 * n + 0.28 * (f - 0.5), 0, 1)[..., None]
    col = GOLD_DK * (1 - smooth(0, 0.5, t)) + GOLD * (smooth(0, 0.5, t) * (1 - smooth(0.55, 1, t))) + GOLD_LT * smooth(0.55, 1, t)
    a = mask[..., None]
    return np.concatenate((col * a, a), -1).astype(np.float32)


def draw_mask(w, h, fn, ss=3):
    """Run fn(draw, ss) on a supersampled L image and return a 0..1 array."""
    img = Image.new("L", (w * ss, h * ss), 0)
    fn(ImageDraw.Draw(img), ss)
    return np.asarray(img.resize((w, h), Image.LANCZOS), np.float32) / 255.0


def jaal_tile(size, line=2.0):
    """One tile of ogee lattice with a small lotus buti in each cell. Returns a 0..1 mask."""
    h = size
    w = int(size * 0.66)

    def fn(dr, ss):
        W_, H_ = w * ss, h * ss
        lw = max(1, round(line * ss))
        pts_l, pts_r = [], []
        for i in range(161):
            t = i / 160
            y = t * H_
            x = 0.5 * W_ * (0.5 + 0.5 * math.cos(2 * math.pi * t)) ** 0.8
            pts_l.append((x, y))
            pts_r.append((W_ - x, y))
        for dx in (0, W_, -W_):
            dr.line([(p[0] + dx, p[1]) for p in pts_l], fill=255, width=lw)
            dr.line([(p[0] + dx, p[1]) for p in pts_r], fill=255, width=lw)
        for (cx, cy) in ((W_ / 2, H_ / 2), (0, 0), (W_, 0), (0, H_), (W_, H_)):
            R = 0.075 * H_
            for k in range(8):
                a = k * math.pi / 4
                poly = []
                for j in range(21):
                    s = j / 20
                    off = 0.38 * math.sin(math.pi * s) ** 0.8 * R * 0.62
                    poly.append((cx + s * R * math.cos(a) - off * math.sin(a), cy + s * R * math.sin(a) + off * math.cos(a)))
                for j in range(20, -1, -1):
                    s = j / 20
                    off = -0.38 * math.sin(math.pi * s) ** 0.8 * R * 0.62
                    poly.append((cx + s * R * math.cos(a) - off * math.sin(a), cy + s * R * math.sin(a) + off * math.cos(a)))
                dr.line(poly, fill=255, width=max(1, lw // 2))
            r = 0.012 * H_
            dr.ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
            for k in range(8):                                     # dots between the petals
                a = k * math.pi / 4 + math.pi / 8
                x, y = cx + R * 0.9 * math.cos(a), cy + R * 0.9 * math.sin(a)
                r = 0.007 * H_
                dr.ellipse((x - r, y - r, x + r, y + r), fill=255)

    return draw_mask(w, h, fn)


def jaal(h, w, cell, line=2.0, seed=3):
    tile = jaal_tile(cell, line)
    th, tw = tile.shape
    out = np.tile(tile, (h // th + 2, w // tw + 2))[:h, :w]
    return np.clip(wobble(out, cell * 0.018, cell * 0.8, seed), 0, 1)


def rangoli_mask(size, seed=5, line=2.4):
    """Chowk-style floor rangoli: petals, scallops and dots, all drawn by polar repetition."""
    def fn(dr, ss):
        n = size * ss
        c0 = n / 2
        R = n * 0.49
        lw = max(1, round(line * ss))

        def circle(r, width=lw):
            dr.ellipse((c0 - r, c0 - r, c0 + r, c0 + r), outline=255, width=width)

        def dots(r, count, s, phase=0.0):
            for k in range(count):
                a = 2 * math.pi * (k + phase) / count
                x, y = c0 + r * math.cos(a), c0 + r * math.sin(a)
                dr.ellipse((x - s, y - s, x + s, y + s), fill=255)

        def petals(r0, r1, count, wid, phase=0.0):
            for k in range(count):
                a = 2 * math.pi * (k + phase) / count
                pts = []
                for sign in (1, -1):
                    rng_ = range(0, 25) if sign == 1 else range(24, -1, -1)
                    for j in rng_:
                        s = j / 24
                        off = sign * wid * math.sin(math.pi * s ** 0.8) ** 0.9
                        r = r0 + (r1 - r0) * s
                        pts.append((c0 + r * math.cos(a + off), c0 + r * math.sin(a + off)))
                dr.line(pts + pts[:1], fill=255, width=lw)

        circle(R * 0.20)
        petals(R * 0.05, R * 0.195, 8, 0.32)
        dots(R * 0.235, 24, R * 0.008)
        circle(R * 0.265)
        petals(R * 0.27, R * 0.47, 16, 0.135)
        petals(R * 0.27, R * 0.40, 16, 0.10, 0.5)
        circle(R * 0.49)
        dots(R * 0.515, 48, R * 0.0065)
        circle(R * 0.54, max(1, lw // 2))
        petals(R * 0.545, R * 0.735, 24, 0.085)
        for k in range(24):                                       # small teardrops between big petals
            a = 2 * math.pi * (k + 0.5) / 24
            x, y = c0 + R * 0.65 * math.cos(a), c0 + R * 0.65 * math.sin(a)
            s = R * 0.018
            dr.ellipse((x - s, y - s, x + s, y + s), outline=255, width=lw)
        circle(R * 0.755)
        for k in range(36):                                       # scalloped outer ring
            a0 = 2 * math.pi * k / 36
            pts = []
            for j in range(19):
                s = j / 18
                a = a0 + s * 2 * math.pi / 36
                r = R * (0.77 + 0.075 * math.sin(math.pi * s))
                pts.append((c0 + r * math.cos(a), c0 + r * math.sin(a)))
            dr.line(pts, fill=255, width=lw)
        dots(R * 0.875, 36, R * 0.009)
        petals(R * 0.89, R * 0.995, 36, 0.05, 0.5)

    m = draw_mask(size, size, fn, ss=2)
    return np.clip(wobble(m, size * 0.0016, size * 0.09, seed), 0, 1)


def cloth(h, w, seed=11, base=PLUM, lift=PLUM_LIFT, weave=0.018, scale=1.0):
    """Hand-dyed cloth ground: uneven dye, faint brush drag, fine weave. Opaque RGB."""
    n = fbm(h, w, 460 * scale, seed, 5, 0.55)
    streak = cv2.resize(fbm(max(h // 8, 8), w, 90 * scale, seed + 1, 3), (w, h), interpolation=cv2.INTER_CUBIC)
    t = np.clip(0.55 * n + 0.45 * streak - 0.28, 0, 1)[..., None] * 0.6
    img = base * (1 - t) + lift * t
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    period = max(2.4, 2.4 * scale)
    w1 = np.sin(xx * 2 * math.pi / period + 4 * fbm(h, w, 30 * scale, seed + 2, 2))
    w2 = np.sin(yy * 2 * math.pi / period + 4 * fbm(h, w, 30 * scale, seed + 3, 2))
    grain = fbm(h, w, 3, seed + 4, 1) - 0.5
    img = img * (1 + (weave * w1 * w2 + weave * 1.4 * grain))[..., None]
    return np.clip(img, 0, 1).astype(np.float32)


# ---------------------------------------------------------------- diya + flame


def diya(width=300, seed=0, flip=False):
    """Hand-pinched clay lamp with a painted rim. Returns (sprite, (wick_x, wick_y))."""
    rng = np.random.default_rng(seed + 300)
    ss = 3
    W, H = int(width * 1.42), int(width * 0.74)
    cx, rim_y = W * 0.44, H * 0.34
    a = width * 0.40 * (1 + rng.normal(0, 0.012))
    e = a * 0.30
    depth = width * 0.335
    wob = [(rng.random() * 6.28, 0.008 + rng.random() * 0.012, f) for f in (3, 5, 8)]

    def rim_r(th, scale=1.0):
        spout = 0.50 * math.exp(-((math.atan2(math.sin(th), math.cos(th))) / 0.36) ** 2)
        w = 1 + sum(am * math.sin(f * th + ph) for ph, am, f in wob)
        return a * scale * (w + spout)

    def rim_pt(th, scale=1.0, dy=0.0):
        r = rim_r(th, scale)
        return cx + r * math.cos(th), rim_y + dy + 0.30 * r * math.sin(th)

    ths = [i * 2 * math.pi / 360 for i in range(360)]
    bowl = [rim_pt(t) for t in ths if 0 <= t <= math.pi]
    for i in range(61):                                       # underside, round to a small foot
        s = i / 60
        th = math.pi - s * math.pi
        bowl.append((cx + a * 0.98 * math.cos(th) * (1 - 0.06 * math.sin(s * math.pi)),
                     rim_y + depth * math.sin(th) ** 0.82))

    def poly(pts):
        def fn(dr, ss_):
            dr.polygon([(x * ss_, y * ss_) for x, y in pts], fill=255)
        return draw_mask(W, H, fn, ss)

    body_m = poly(bowl)
    top_m = poly([rim_pt(t) for t in ths])
    inner_m = poly([rim_pt(t, 0.84, 1.5) for t in ths])
    oil_m = poly([rim_pt(t, 0.72, 3.5) for t in ths])
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    nx = (xx - cx) / a
    ny = (yy - rim_y) / depth
    mott = fbm(H, W, width * 0.14, seed + 2, 4)
    fine = fbm(H, W, 4, seed + 3, 2)
    clay_dk, clay, clay_lt = c(92, 34, 16), c(178, 84, 40), c(232, 142, 84)
    lit = np.clip(0.60 - 0.42 * nx - 0.45 * ny + 0.22 * (mott - 0.5), 0, 1)[..., None]
    col = clay_dk * (1 - smooth(0, 0.55, lit)) + clay * (smooth(0, 0.55, lit) * (1 - smooth(0.55, 1, lit))) + clay_lt * smooth(0.55, 1, lit)
    col = col * (1 + 0.10 * (fine - 0.5))[..., None]

    # painted bands: kumkum red, then a run of chalk-white dots, as on a home-painted diya
    band_mid = 0.30 * np.sqrt(np.clip(1 - nx ** 2, 0, 1))
    red = smooth(0.06, 0.04, np.abs(ny - band_mid - 0.03)) * (np.abs(nx) < 1.03)
    wob_paint = 0.75 + 0.25 * fbm(H, W, 16, seed + 4, 2)
    col = col + (c(174, 18, 40) - col) * (0.92 * red * wob_paint)[..., None]

    def dotfn(dr, ss_):
        for k in range(17):
            th = math.pi * (k + 0.5) / 17
            x = cx + a * 0.96 * math.cos(th)
            y = rim_y + depth * (0.33 + 0.30) * math.sin(th) ** 0.8 + rng.normal(0, 0.8)
            r = width * 0.0105 * (1 + rng.normal(0, 0.12)) * (0.65 + 0.35 * math.sin(th))
            dr.ellipse(((x - r) * ss_, (y - r) * ss_, (x + r) * ss_, (y + r) * ss_), fill=255)

    dots = draw_mask(W, H, dotfn, ss)
    col = col + (c(252, 240, 214) - col) * (0.94 * dots)[..., None]

    def zig(dr, ss_):
        pts = []
        for k in range(49):
            th = math.pi * k / 48
            amp = 0.05 * (1 if k % 2 else -1)
            pts.append(((cx + a * 0.93 * math.cos(th)) * ss_, (rim_y + depth * (0.47 + amp) * math.sin(th) ** 0.8 + 3) * ss_))
        dr.line(pts, fill=255, width=max(1, round(width * 0.0065 * ss_)))
    zigm = draw_mask(W, H, zig, ss)
    col = col + (GOLD_LT - col) * (0.85 * zigm)[..., None]

    im = np.concatenate((col * body_m[..., None], body_m[..., None]), -1)
    # top face: thick lip, inside shadow, then oil
    lip = c(214, 128, 72) * (0.86 + 0.26 * mott[..., None])
    lip = lip * (1 - 0.26 * smooth(-0.4, 1.0, nx))[..., None]
    t_a = top_m[..., None]
    im = np.concatenate((lip * t_a, t_a), -1) + im * (1 - t_a)
    inside = c(70, 22, 14) * (0.8 + 0.5 * smooth(0.5, -1.0, nx))[..., None]
    i_a = inner_m[..., None]
    im = np.concatenate((inside * i_a, i_a), -1) + im * (1 - i_a)
    oil = c(120, 62, 8) + (c(226, 150, 30) - c(120, 62, 8)) * smooth(-0.5, 1.0, nx)[..., None]
    o_a = oil_m[..., None]
    im = np.concatenate((oil * o_a, o_a), -1) + im * (1 - o_a)
    spec = np.exp(-(((xx - cx - a * 0.1) / (a * 0.26)) ** 2 + ((yy - rim_y - 2) / (e * 0.14 + 1)) ** 2)) * oil_m
    im[..., :3] = im[..., :3] + (1 - im[..., :3]) * (0.55 * spec)[..., None] * im[..., 3:]

    # cotton wick, from the oil out to the spout
    tipx, tipy = rim_pt(0)
    wx0, wy0 = tipx - a * 0.46, rim_y + 3
    wx1, wy1 = tipx - a * 0.045, tipy - width * 0.040

    def wick(dr, ss_):
        pts = []
        for k in range(21):
            s = k / 20
            pts.append(((wx0 + (wx1 - wx0) * s) * ss_, (wy0 + (wy1 - wy0) * s - math.sin(s * math.pi) * width * 0.012) * ss_))
        dr.line(pts, fill=255, width=round(width * 0.020 * ss_), joint="curve")

    wm = draw_mask(W, H, wick, ss)
    w_col = c(240, 226, 196) * (1 - 0.3 * smooth(0.0, 1.0, (xx - wx0) / (wx1 - wx0 + 1)))[..., None]
    char = np.exp(-(((xx - wx1) ** 2 + (yy - wy1) ** 2) / (width * 0.022) ** 2))
    w_col = w_col + (c(30, 16, 12) - w_col) * char[..., None]
    w_a = wm[..., None]
    im = np.concatenate((w_col * w_a, w_a), -1) + im * (1 - w_a)
    wick_at = (wx1, wy1)
    if flip:
        im = im[:, ::-1].copy()
        wick_at = (W - wx1, wy1)
    return im.astype(np.float32), wick_at


class Flame:
    """A flame that repeats exactly every `loop` seconds, so overlays loop seamlessly."""

    def __init__(self, height, seed=0, loop=200 / 60):
        self.h = float(height)
        self.loop = loop
        rng = np.random.default_rng(seed + 700)
        self.ph = rng.random(9) * 6.283
        self.f = rng.choice([1, 2, 3], 2), rng.choice([3, 4, 5], 2), rng.choice([6, 7, 8, 9], 2)
        H = int(self.h * 3.0)
        W = int(self.h * 3.0)
        self.W, self.H = W, H
        self.ax, self.ay = W / 2, H * 0.74            # where the flame stands (the wick tip)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.x, self.y = xx + 0.5 - self.ax, self.ay - (yy + 0.5)
        r2 = (self.x / (self.h * 0.95)) ** 2 + ((self.y - self.h * 0.42) / (self.h * 1.05)) ** 2
        self.halo = np.exp(-r2 * 1.7)
        self.halo_core = np.exp(-r2 * 9.0)

    def render(self, t, power=1.0):
        """Premultiplied RGBA of the flame plus its halo. power scales the flame on ignition."""
        if power <= 0:
            return np.zeros((self.H, self.W, 4), np.float32)
        w0 = 2 * math.pi * t / self.loop
        p, (f1, f2, f3) = self.ph, self.f
        hgt = self.h * (0.93 + 0.055 * math.sin(f1[0] * w0 + p[0]) + 0.035 * math.sin(f2[0] * w0 + p[1])
                        + 0.018 * math.sin(f3[0] * w0 + p[2])) * power
        wid = 0.235 * self.h * (1 + 0.05 * math.sin(f2[1] * w0 + p[3])) * (0.55 + 0.45 * power)
        s = np.clip(self.y / max(hgt, 1), 0, 1.2)
        lean = (0.11 * math.sin(f1[1] * w0 + p[4]) + 0.05 * math.sin(f2[0] * w0 + p[5])) * self.h
        rip = 0.040 * self.h * np.sin(s * 6.0 - f3[0] * w0 + p[6]) + 0.015 * self.h * np.sin(s * 13 - f3[1] * w0 + p[7])
        dx = self.x - (lean * s ** 1.65 + rip * s ** 1.1)
        prof = np.clip(s, 1e-3, 1) ** 0.44 * np.clip(1 - s, 0, 1) ** 0.82 * 1.72
        half = wid * prof + 0.2
        d = np.abs(dx) / half
        body = smooth(1.04, 0.72, d) * (self.y > -self.h * 0.03) * (s < 1)
        outer = c(255, 118, 8)
        midc = c(255, 186, 30)
        core = c(255, 247, 214)
        k1 = smooth(0.95, 0.42, d)[..., None]
        k2 = (smooth(0.62, 0.12, d) * smooth(0.96, 0.25, s))[..., None]
        col = outer * (1 - k1) + midc * k1
        col = col * (1 - k2) + core * k2
        # the cool dark root of a real flame, around the wick
        root = (smooth(0.20, 0.0, s) * smooth(1.0, 0.1, d))[..., None]
        col = col * (1 - 0.82 * root) + c(66, 54, 176) * (0.82 * root)
        al = body * (1 - 0.48 * smooth(0.2, 0.0, s))
        a = al[..., None]
        fl = np.concatenate((col * a, a), -1)
        flick = 0.92 + 0.08 * math.sin(f2[0] * w0 + p[8])
        halo = (0.30 * self.halo ** 1.6 + 0.34 * self.halo_core) * flick * power
        hc = c(255, 142, 40) * halo[..., None]
        h_rgba = np.concatenate((hc, halo[..., None] * 0.95), -1).astype(np.float32)
        return fl + h_rgba * (1 - fl[..., 3:])


def font(size, weight=620):
    f = ImageFont.truetype(str(FONT), size)
    try:
        f.set_variation_by_axes([weight])
    except OSError:
        pass
    return f


def load_logo(height):
    im = Image.open(LOGO).convert("RGBA")
    im = im.resize((round(im.width * height / im.height), height), Image.LANCZOS)
    a = np.asarray(im, np.float32) / 255.0
    a[..., :3] *= a[..., 3:]
    return a


def to_png(img, path, level=6):
    """Save premultiplied float RGBA as a straight-alpha 8-bit PNG."""
    a = np.clip(img[..., 3:], 0, 1)
    rgb = np.divide(img[..., :3], a, out=np.zeros_like(img[..., :3]), where=a > 1e-4)
    u8 = np.clip(np.concatenate((rgb, a), -1) * 255 + 0.5, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(u8, cv2.COLOR_RGBA2BGRA), [cv2.IMWRITE_PNG_COMPRESSION, level])


def to_rgba8(img):
    a = np.clip(img[..., 3:], 0, 1)
    rgb = np.divide(img[..., :3], a, out=np.zeros_like(img[..., :3]), where=a > 1e-4)
    return np.clip(np.concatenate((rgb, a), -1) * 255 + 0.5, 0, 255).astype(np.uint8)
