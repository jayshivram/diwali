"""Bundle the Diwali lower third into one self-contained HTML file: artwork and font embedded, works offline.

    python make_lowerthird_html.py <output.html>
"""
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from art import FONT, Flame, to_rgba8
from make_lowerthirds import Parts

HERE = Path(__file__).parent
SIZE, COLS, FRAMES, LOOP = 150, 15, 90, 3.0


def webp(arr):
    buf = io.BytesIO()
    Image.fromarray(to_rgba8(arr), "RGBA").save(buf, "WEBP", quality=94, method=6, alpha_quality=100, exact=True)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


def flame_sheet():
    f = Flame(50, 3, loop=LOOP)                      # sprite is exactly 150 px square
    rows = FRAMES // COLS
    sheet = np.zeros((rows * SIZE, COLS * SIZE, 4), np.float32)
    for n in range(FRAMES):
        y, x = n // COLS * SIZE, n % COLS * SIZE
        sheet[y:y + SIZE, x:x + SIZE] = f.render(n / FRAMES * LOOP)[:SIZE, :SIZE]
    return sheet, f.ay


def main():
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    p = Parts()
    sheet, ay = flame_sheet()
    assets = dict(ring=webp(p.ring), face=webp(p.face), diya=webp(p.diya), bar=webp(p.bar), bloom=webp(p.end_bloom), flame=webp(sheet))
    names = [n.strip() for n in (HERE / "names.txt").read_text(encoding="utf-8").splitlines() if n.strip()]
    html = (HERE / "lowerthird_template.html").read_text(encoding="utf-8")
    for key, val in (("__ASSETS__", json.dumps(assets)), ("__NAMES__", json.dumps(names)),
                     ("__WICK__", json.dumps([round(float(v), 2) for v in p.wick])),
                     ("__FLAME__", json.dumps(dict(size=SIZE, cols=COLS, frames=FRAMES, loop=LOOP, ay=ay))),
                     ("__FONT__", base64.b64encode(FONT.read_bytes()).decode())):
        html = html.replace(key, val)
    out.write_text(html, encoding="utf-8")
    print(out, f"{out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
