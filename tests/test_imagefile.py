"""Image reading and saving for the static scan (imagefile.py)."""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from imagefile import clean_dropped_path, read_image, write_image  # noqa: E402


def test_dropped_paths_lose_quotes_and_the_powershell_ampersand():
    assert clean_dropped_path('  "C:\\My Photos\\crowd.jpg"  ') == "C:\\My Photos\\crowd.jpg"
    assert clean_dropped_path("& 'C:\\My Photos\\crowd.jpg'") == "C:\\My Photos\\crowd.jpg"
    assert clean_dropped_path("/home/me/crowd.jpg\n") == "/home/me/crowd.jpg"
    assert clean_dropped_path("   ") == ""


def test_round_trip_through_a_non_ascii_folder(tmp_path):
    folder = tmp_path / "స్టేడియం İzmir"
    folder.mkdir()
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[:, :30] = (0, 0, 255)
    path = str(folder / "crowd.png")
    assert write_image(path, image)
    loaded = read_image(path)
    assert loaded is not None and loaded.shape == (40, 60, 3)
    assert np.array_equal(loaded, image)  # PNG is lossless


def test_missing_empty_and_non_image_files_read_as_none(tmp_path):
    assert read_image(str(tmp_path / "nope.jpg")) is None
    assert read_image(str(tmp_path)) is None  # a folder
    empty = tmp_path / "empty.jpg"
    empty.write_bytes(b"")
    assert read_image(str(empty)) is None
    text = tmp_path / "notes.jpg"
    text.write_text("not an image")
    assert read_image(str(text)) is None


def test_write_reports_failure(tmp_path):
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    assert not write_image(str(tmp_path / "missing-folder" / "out.jpg"), image)
    assert write_image(str(tmp_path / "out"), image)  # no extension: saved as JPEG
    assert read_image(str(tmp_path / "out")) is not None
