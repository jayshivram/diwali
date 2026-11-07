# Diwali broadcast pack

Graphics for BAPS Shri Swaminarayan Mandir, Dar es Salaam. Operator settings are in
[output_diwali/README.txt](output_diwali/README.txt).

| Folder in `output_diwali/` | Contents |
|---|---|
| `lower_thirds_html/` | Lower third as one HTML file, for OBS, vMix or a browser |
| `stinger/` | 180-frame 60 fps PNG sequence with alpha, and a ProRes copy |
| `diya_overlay/` | IN, LOOP and OUT clips, as ProRes and PNG sequences |
| `backgrounds/` | 1920x1080 and 4608x2304, as still PNG and looping MP4 |
| `supersource/` | ATEM SuperSource art, as RGBA or fill and key |
| `previews/` | Stinger review video |

## Lower thirds

The ProRes lower third videos aren't in this repo, so ignore the `lower_thirds/` section of
the operator notes unless you render them. To use the HTML version, open
`output_diwali/lower_thirds_html/Diwali_Lower_Third.html`, or add `?name=Your%20Name` to the
address for a fixed name. Details are in the operator notes.

## Downloading

The `.mov` files are stored with Git LFS. GitHub's "Download ZIP" may give small pointer
files instead of the videos, so clone with Git LFS installed:

```bash
git lfs install
```

```bash
git clone https://github.com/jayshivram/diwali.git
```

## Rendering again

The code is in `diwali/` and needs Python with numpy, pillow and opencv-python-headless, plus
ffmpeg on the path. Names are in `diwali/names.txt`. Run the scripts from inside `diwali/`:

```bash
python make_lowerthird_html.py ../output_diwali/lower_thirds_html/Diwali_Lower_Third.html
```

```bash
python make_lowerthirds.py ../output_diwali names.txt
```

`make_stinger_diwali.py`, `make_overlay.py` and `make_backgrounds.py` each take the output
folder as their argument.

Eczar is under the SIL Open Font License, included in `diwali/fonts/`.
