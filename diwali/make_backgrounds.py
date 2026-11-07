"""Themed backgrounds (1080p and 4608x2304 LED) and the ATEM SuperSource art.

    python make_backgrounds.py <output_dir> [--previews-only]
"""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from art import GOLD_LT, gold_paint, draw_mask, smooth, to_png, xform
from scene import Lamps, garland_row, ground, paste_c, scatter_petals, strand_sprite, bank
from art import over, c


def lamp_placements(W, H, u):
    """Row of lights: clusters of three in the footer corners, inner lamps smaller, facing the centre."""
    base = H - 20 * u
    big, mid, small = 150 * u, 118 * u, 90 * u
    edge = 44 * u
    a = edge + big * 0.5
    b = a + big * 0.72 + mid * 0.44
    d = b + mid * 0.72 + small * 0.46
    left = [(d, base - 4 * u, small, False), (b, base - 2 * u, mid, False), (a, base, big, False)]
    right = [(W - x, y, s, True) for x, y, s, _ in left]
    return left + right


def compose_background(W, H, seed=11, loop=200 / 60):
    """Returns the static plate (everything except flames) and its Lamps."""
    u = H / 1080
    img = ground(H, W, u, seed)
    # top strands hang behind the garland row, at the outer edges so copy stays clear
    pos = [(52 * u, 470 * u, 1), (112 * u, 330 * u, 2), (W - 52 * u, 470 * u, 3), (W - 112 * u, 330 * u, 4)]
    for x, length, s in pos:
        st = strand_sprite(length, u, s)
        over(img, st, x - st.shape[1] / 2, 0)
    # same layout as the client's slide: title above the top row, names between, footer below the lower
    garland_row(img, 150 * u, -30 * u, W + 30 * u, u, seed=3)
    garland_row(img, 884 * u, -30 * u, W + 30 * u, u, seed=8)
    lamps = Lamps(lamp_placements(W, H, u), u, loop)
    spots = [(190 * u, H - 40 * u, 120 * u, 18), (W - 190 * u, H - 40 * u, 120 * u, 18)]
    scatter_petals(img, u, spots, seed)
    lamps.clay(img)
    return img, lamps


def supersource_art():
    """Two picture windows with a gold jharokha frame. Fill plus key, with the box settings to match."""
    W, H = 1920, 1080
    u = 1.0
    img = ground(H, W, u, seed=21, rangoli=False)
    garland_row(img, 92, -30, W + 30, u, seed=5, size=58)
    garland_row(img, 1012, -30, W + 30, u, seed=6, size=58)
    win_w, win_h = 832, 468
    gap = 84
    cy = 500
    cxs = (W / 2 - gap / 2 - win_w / 2, W / 2 + gap / 2 + win_w / 2)
    windows = []
    for cx in cxs:
        windows.append((cx - win_w / 2, cy - win_h / 2, cx + win_w / 2, cy + win_h / 2))

    def cusped(x0, y0, x1, y1, grow=0.0):
        """Rectangle with lobed upper corners and small chamfers below: a jharokha opening."""
        x0, y0, x1, y1 = x0 - grow, y0 - grow, x1 + grow, y1 + grow
        r = 62 + 2 * grow      # grows faster than the outline, so the gold line stays unbroken at the shoulder
        pts = []
        import math
        # top-left lobes, going clockwise around the shape
        def lobes(cx_, cy_, a0, a1, count=3):
            out = []
            for k in range(count):
                b0 = a0 + (a1 - a0) * k / count
                b1 = a0 + (a1 - a0) * (k + 1) / count
                for j in range(17):
                    s = j / 16
                    a = b0 + (b1 - b0) * s
                    rr = r * (0.80 + 0.20 * math.sin(math.pi * s))     # lobes meeting in small cusps
                    out.append((cx_ + rr * math.cos(a), cy_ + rr * math.sin(a)))
            return out
        # sides and top stay truly square to the picture; the cusps sit only in the upper corners
        pts += [(x0, y0 + r)]
        pts += lobes(x0 + r, y0 + r, math.pi, 1.5 * math.pi)
        pts += [(x0 + r, y0), (x1 - r, y0)]
        pts += lobes(x1 - r, y0 + r, 1.5 * math.pi, 2 * math.pi)
        pts += [(x1, y0 + r)]
        ch = 16 + grow * 0.3
        pts += [(x1, y1 - ch), (x1 - ch, y1), (x0 + ch, y1), (x0, y1 - ch)]
        return pts

    hole = draw_mask(W, H, lambda dr, ss: [dr.polygon([(x * ss, y * ss) for x, y in cusped(*win)], fill=255) for win in windows], 3)
    frame_o = draw_mask(W, H, lambda dr, ss: [dr.polygon([(x * ss, y * ss) for x, y in cusped(*win, grow=16)], fill=255) for win in windows], 3)
    frame_i = draw_mask(W, H, lambda dr, ss: [dr.polygon([(x * ss, y * ss) for x, y in cusped(*win, grow=7)], fill=255) for win in windows], 3)
    frame_e = draw_mask(W, H, lambda dr, ss: [dr.polygon([(x * ss, y * ss) for x, y in cusped(*win, grow=23)], fill=255) for win in windows], 3)
    frame_f = draw_mask(W, H, lambda dr, ss: [dr.polygon([(x * ss, y * ss) for x, y in cusped(*win, grow=21)], fill=255) for win in windows], 3)
    # soft shadow the frame throws on the cloth
    sh = cv2.GaussianBlur(frame_e, (0, 0), 12)
    sh = np.roll(sh, 7, axis=0)
    img[..., :3] *= 1 - 0.55 * sh[..., None]
    fine = np.clip(frame_e - frame_f, 0, 1)
    g = gold_paint(fine * 0.9, 31)
    img = g + img * (1 - g[..., 3:])
    band = np.clip(frame_o - frame_i, 0, 1)
    g = gold_paint(band, 33)
    img = g + img * (1 - g[..., 3:])
    # bead run between the two gold lines
    inner_band = np.clip(frame_i - hole, 0, 1)
    img[..., :3] = img[..., :3] * (1 - inner_band[..., None]) + c(30, 0, 30) * inner_band[..., None]
    img[..., :3] = np.clip(img[..., :3], 0, 1)
    # cut the windows first, so the blooms and lamps painted afterwards sit over the picture's edge
    img *= (1 - hole)[..., None]
    bloom = bank(64, 40, 4, "lotus")
    rng = np.random.default_rng(4)
    for x0, y0, x1, y1 in windows:
        for (x, y) in ((x0 - 10, y1 + 10), (x1 + 10, y1 + 10), (x0 + 6, y0 + 6), (x1 - 6, y0 + 6)):
            paste_c(img, xform(bloom[rng.integers(0, 4)], 1.0, rng.random() * 360), x, y)
        apex = (x0 + x1) / 2
        paste_c(img, bloom[1], apex, y0 - 16)
    # a low row of lights under the two windows
    step = 126
    places = []
    for i in range(5):                              # five each side, flames facing the centre, a gap between
        x = (i + 0.75) * step
        size = 104 - i * 3
        places.append((W / 2 - x, 924, size, False))
        places.append((W / 2 + x, 924, size, True))
    lamps = Lamps(places, 1.0)
    lamps.clay(img)
    lamps.flames(img, 0.4)
    alpha = img[..., 3].copy()
    art = img
    boxes = []
    for x0, y0, x1, y1 in windows:
        boxes.append(((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))
    return art, alpha, boxes


def encode_loop(plate, lamps, W, H, path, frames=200, fps=60):
    """Seamless loop of the plate with live flames. Only the flame regions are repainted each frame."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", "15", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    base8 = np.clip(plate[..., :3] * 255 + 0.5, 0, 255).astype(np.uint8)
    for i in range(frames):
        fr = base8.copy()
        for it in lamps.items:
            f = it["flame"].render(i / fps)
            x = int(round(it["wx"] - f.shape[1] * 0.5))
            y = int(round(it["wy"] - it["flame"].ay))
            x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + f.shape[1], W), min(y + f.shape[0], H)
            s = f[y0 - y:y1 - y, x0 - x:x1 - x]
            d = plate[y0:y1, x0:x1, :3]
            fr[y0:y1, x0:x1] = np.clip((s[..., :3] + d * (1 - s[..., 3:])) * 255 + 0.5, 0, 255).astype(np.uint8)
        pr.stdin.write(fr.tobytes())
    pr.stdin.close()
    if pr.wait() != 0:
        raise RuntimeError("ffmpeg failed")


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "backgrounds").mkdir(exist_ok=True)
    (out / "supersource").mkdir(exist_ok=True)
    for name, W, H in (("Diwali_Background_1920x1080", 1920, 1080), ("Diwali_Background_LED_4608x2304", 4608, 2304)):
        img, lamps = compose_background(W, H)
        encode_loop(img, lamps, W, H, out / "backgrounds" / f"{name}_LOOP_60fps.mp4")
        lamps.flames(img, 0.7)
        rgb = np.clip(img[..., :3] * 255 + 0.5, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out / "backgrounds" / f"{name}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 6])
        print(name, flush=True)
    art, alpha, boxes = supersource_art()
    to_png(art, out / "supersource" / "Diwali_SuperSource_Art_RGBA.png")
    # fill on cloth colour (not premultiplied black) plus separate key, for the Fill+Key workflow
    fill = np.clip(np.divide(art[..., :3], np.maximum(alpha[..., None], 1e-3)), 0, 1)
    fill = fill * alpha[..., None]        # premultiplied, what the ATEM expects with Pre Multiplied Key on
    cv2.imwrite(str(out / "supersource" / "Diwali_SuperSource_Art_Fill.png"),
                cv2.cvtColor(np.clip(fill * 255 + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "supersource" / "Diwali_SuperSource_Art_Key.png"), np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8))
    print("boxes", boxes, flush=True)


if __name__ == "__main__":
    main()
