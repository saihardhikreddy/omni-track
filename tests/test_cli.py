"""
Unit tests for Omni-Track CLI parsers and diagnostic tools.
"""

import sys
from pathlib import Path
from unittest.mock import patch
import pytest

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
