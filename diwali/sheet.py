"""Contact sheet of the painted motifs, for checking the art before rendering."""
import sys

import cv2
import numpy as np

from art import PLUM, Flame, diya, jaal, leaf, lotus, marigold, over, rangoli_mask, to_png, cloth, gold_paint, GOLD

W, H = 1800, 1000
bg = np.concatenate((cloth(H, W), np.ones((H, W, 1), np.float32)), -1)
m = jaal(H, W, 220, 1.6) * 0.20
g = gold_paint(m)
bg = g + bg * (1 - g[..., 3:])
over(bg, lotus(240, 1), 20, 20)
over(bg, lotus(120, 2), 280, 20)
over(bg, lotus(64, 3), 410, 20)
over(bg, marigold(200, 1), 500, 20)
over(bg, marigold(80, 2), 710, 20)
over(bg, marigold(40, 3), 800, 20)
over(bg, leaf(280, 34, 1), 860, 20)
over(bg, leaf(140, 18, 2), 860, 170)
for i, (w, x) in enumerate(((560, 20), (260, 840), (140, 1220))):
    s, (wx, wy) = diya(w, i)
    y = 420 if i == 0 else 480
    over(bg, s, x, y)
    f = Flame(w * 0.34, i).render(0.6 + i * 0.3)
    over(bg, f, x + wx - f.shape[1] * 0.5, y + wy - f.shape[0] * 0.74)
r = rangoli_mask(460, line=2.0)
g = gold_paint(r * 0.9)
over(bg, g, 1330, 20)
to_png(bg, sys.argv[1])
