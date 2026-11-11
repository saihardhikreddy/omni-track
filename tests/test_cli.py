"""
Unit tests for Omni-Track CLI parsers and diagnostic tools.
"""

import sys
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import test_cam


def test_parse_args_defaults():
    with patch.object(sys, "argv", ["test_cam.py"]):
        args = test_cam.parse_args()
        assert args.camera is None
        assert args.width == 1920
        assert args.height == 1080


def test_parse_args_custom_camera():
    with patch.object(sys, "argv", ["test_cam.py", "--camera", "2", "--width", "1280", "--height", "720"]):
        args = test_cam.parse_args()
        assert args.camera == 2
        assert args.width == 1280
        assert args.height == 720


class _DeadCamera:
    released = False

    def isOpened(self):
        return False

    def release(self):
        self.released = True


class _LiveCamera:
    def __init__(self):
        self.frames = 0
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        import numpy as np
        self.frames += 1
        return True, np.zeros((8, 8, 3), np.uint8)

    def set(self, prop, value):
        return True

    def get(self, prop):
        return 8

    def release(self):
        self.released = True


def test_main_fails_when_no_camera_works():
    # A diagnostic must report failure through its exit code so scripts can react.
    with patch.object(sys, "argv", ["test_cam.py", "--camera", "7"]), \
            patch.object(test_cam.cv2, "VideoCapture", return_value=_DeadCamera()):
        assert test_cam.main() == 1


def test_stadium_args_are_validated():
    import pytest
    import stadium_elite

    args = stadium_elite.parse_args(["--conf", "0.3", "--slice-size", "512"])
    assert (args.conf, args.slice_size) == (0.3, 512)
    for bad in (["--conf", "0"], ["--conf", "1.5"], ["--slice-size", "0"]):
        with pytest.raises(SystemExit):
            stadium_elite.parse_args(bad)


def test_camera_backend_per_platform():
    assert test_cam.camera_backend("win32") == test_cam.cv2.CAP_DSHOW
    assert test_cam.camera_backend("linux") == test_cam.cv2.CAP_ANY
    assert test_cam.camera_backend("darwin") == test_cam.cv2.CAP_ANY


def test_app_does_not_hard_code_directshow():
    # app.py loads GPU models on import, so check its source instead.
    source = (REPO_ROOT / "app.py").read_text(encoding="utf-8")
    assert "cv2.CAP_DSHOW" not in source


def test_unopened_cameras_are_released():
    cams = []

    def make(*_):
        cams.append(_DeadCamera())
        return cams[-1]

    with patch.object(sys, "argv", ["test_cam.py"]), patch.object(test_cam.cv2, "VideoCapture", make):
        assert test_cam.main() == 1
    assert len(cams) == 4 and all(c.released for c in cams)


def test_feed_stops_when_the_window_is_closed():
    # Closing the window with its X button used to leave the loop running, reopening it.
    cam = _LiveCamera()
    with patch.object(sys, "argv", ["test_cam.py", "--camera", "0"]), \
            patch.object(test_cam.cv2, "VideoCapture", return_value=cam), \
            patch.object(test_cam.cv2, "imshow"), \
            patch.object(test_cam.cv2, "waitKey", return_value=-1), \
            patch.object(test_cam.cv2, "destroyAllWindows"), \
            patch.object(test_cam.cv2, "getWindowProperty", side_effect=[1.0, 1.0, 0.0]):
        assert test_cam.main() == 0
    assert cam.frames == 4  # the probe read, then three shown frames
    assert cam.released


def test_feed_stops_on_escape():
    cam = _LiveCamera()
    with patch.object(sys, "argv", ["test_cam.py", "--camera", "0"]), \
            patch.object(test_cam.cv2, "VideoCapture", return_value=cam), \
            patch.object(test_cam.cv2, "imshow"), \
            patch.object(test_cam.cv2, "waitKey", return_value=27), \
            patch.object(test_cam.cv2, "destroyAllWindows"), \
            patch.object(test_cam.cv2, "getWindowProperty", return_value=1.0):
        assert test_cam.main() == 0
    assert cam.frames == 2
