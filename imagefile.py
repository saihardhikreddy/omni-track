"""Reading and saving images for the static scan in app.py.

cv2.imread and cv2.imwrite go through the C runtime's narrow-character file API on Windows,
so a path with characters outside the system code page (a folder named "Stadyum_İzmir" or
"స్టేడియం", or most OneDrive folders under a non-English user name) reads as "could not read
the image" even though the file is there. Reading the bytes with numpy and decoding them in
memory works for any path.
"""

import os

import cv2
import numpy as np


def clean_dropped_path(text):
    """The path as typed or dragged into the terminal: surrounding spaces and quotes removed,
    and the "& " that PowerShell puts in front of a dragged-in quoted path."""
    text = text.strip()
    if text.startswith("& "):
        text = text[2:].lstrip()
    return text.strip(" \"'")


def read_image(path):
    """The image at path as a BGR array, or None when the file is missing or not an image."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path, image):
    """Save image to path in the format its extension names. Returns False if that failed."""
    ext = os.path.splitext(str(path))[1] or ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False
    try:
        encoded.tofile(path)
    except OSError:
        return False
    return True
