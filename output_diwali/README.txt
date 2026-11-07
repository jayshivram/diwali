DIWALI BROADCAST PACK - BAPS Shri Swaminarayan Mandir, Dar es Salaam

All video is 60 fps. All alpha is straight (unpremultiplied).


STINGER  (stinger/)
  png_60fps/   180 PNG frames at 1920x1080 with alpha, for the ATEM Media Pool
  BAPS_Diwali_Stinger_ProRes4444.mov   same animation as one file

  ATEM Stinger settings
    Clip Duration    180
    Trigger Point    85
    Mix Rate         1
    Pre Multiplied Key   ON  (same as your earlier stingers; if edges look wrong, try it OFF)
  Frames 51 to 122 fully cover the picture, so the cut at 85 is hidden.


LOWER THIRDS  (lower_thirds/)
  One ProRes 4444 .mov with alpha per name, for Mitti.
  9 seconds: 3 in, 3 on screen, 3 out. The exit plays the entry backwards.
  The bar ends at 45% of the frame width, as on v2. Long names shrink to fit.

  To add or change names: edit diwali/names.txt, then run make_lowerthirds.py.


HTML LOWER THIRD  (lower_thirds_html/)
  Diwali_Lower_Third.html is the same lower third as one self-contained file, for apps
  with a browser source such as OBS or vMix. Mitti plays video files, so use the .mov
  files there. The background is transparent in OBS.

  Choose the name in the address:
    Diwali_Lower_Third.html?name=Pujya%20Swami      %20 is a space, %26 is &
    &hold=5     seconds on screen (default 3)
    &hold=inf   stays on until taken out
    &loop=1     keeps replaying, for checking

  Opened with no name in an ordinary browser, it shows a panel with the names list and
  a box for typing a new name. Space takes it in or out, R replays, H hides the panel.
  In OBS, hiding and showing the source plays it again.

  It draws at 1920x1080. A larger source scales it up and it will look slightly soft.


DIYA OVERLAY  (diya_overlay/)
  Row of lit lamps along the foot of the frame, marigolds in the top corners.
  IN    90 frames   lamps rise and light from the centre outward
  LOOP  200 frames  seamless, set this one to loop
  OUT   90 frames   flames go out, lamps sink away
  Play IN, then LOOP for as long as you need, then OUT. The three join without a jump.
  Each comes as ProRes 4444 for Mitti, and as a PNG sequence for the ATEM.


BACKGROUNDS  (backgrounds/)
  1920x1080 and 4608x2304 (LED), each as a still PNG and a looping MP4 with live flames.
  Layout follows your Thaal Seva slide: title above the top flower row, names between
  the rows, footer below.


SUPERSOURCE  (supersource/)
  Art for two picture windows. Load as Art with the frame in front of the boxes.
    _RGBA.png   one file with alpha, for the Media Pool
    _Fill.png + _Key.png   same art as a separate fill and key

  Box settings to start from. Check them on the multiview, as these have not been
  tried on a switcher:
    Box 1   Size 0.45   Position X -7.63   Position Y 0.67
    Box 2   Size 0.45   Position X  7.63   Position Y 0.67


NOT TESTED ON YOUR EQUIPMENT
  Files were checked by decoding them on this computer. Nothing has been played on
  the ATEM, in Mitti or on the LED wall yet, so run each one through before the event.
