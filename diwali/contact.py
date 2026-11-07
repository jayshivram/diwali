"""Contact sheet: python contact.py out.png cols width file1 file2 ... (alpha shown over steel blue)."""
import sys

import cv2
import numpy as np

out, cols, width = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
tiles = []
for p in sys.argv[4:]:
    im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if im.shape[2] == 4:
        a = im[..., 3:].astype(np.float32) / 255
        im = (im[..., :3] * a + np.array([143, 111, 63], np.float32) * (1 - a)).astype(np.uint8)
    s = width / im.shape[1]
    tiles.append(cv2.resize(im, (width, round(im.shape[0] * s)), interpolation=cv2.INTER_AREA))
blank = np.zeros_like(tiles[0])
rows = [np.hstack((tiles[i:i + cols] + [blank] * cols)[:cols]) for i in range(0, len(tiles), cols)]
cv2.imwrite(out, np.vstack(rows))
