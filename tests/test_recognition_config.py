import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recognition_config import LETTER_LABELS, labels_for


def test_letter_labels_are_uppercase_a_to_z():
    assert labels_for("letters") == LETTER_LABELS
    assert LETTER_LABELS[0] == "A"
    assert LETTER_LABELS[-1] == "Z"
    assert len(LETTER_LABELS) == 26
