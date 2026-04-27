import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from text_segmentation import _RawBox, _best_vertical_split


def test_best_vertical_split_detects_valley_between_joined_letters():
    binary = np.zeros((40, 80), dtype=np.uint8)
    binary[5:35, 8:28] = 255
    binary[5:35, 40:60] = 255

    box = _RawBox((0, 0, 80, 40), area=1200)
    split_column = _best_vertical_split(box, binary, median_width=28)

    assert split_column is not None
    assert 28 <= split_column <= 50
