"""ShapeMatch geometry validation report."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from validate_shape_match_reference import Detection, _match  # noqa: E402


def _detection(x, y, angle=0.0, scale=1.0):
    return Detection(x=x, y=y, angle=angle, scale=scale, score=0.5)


def test_match_pairs_detections_by_position():
    reference = [_detection(100.0, 100.0), _detection(300.0, 200.0)]
    candidate = [_detection(300.4, 200.2), _detection(100.2, 99.8)]
    pairs, lost, extra = _match(reference, candidate, max_px=10.0)
    assert not lost
    assert not extra
    assert len(pairs) == 2
    mapping = {reference_index: candidate_index for candidate_index, reference_index, _ in pairs}
    assert mapping == {0: 1, 1: 0}


def test_match_reports_lost_and_extra_detections():
    reference = [_detection(100.0, 100.0)]
    candidate = [_detection(400.0, 400.0)]
    pairs, lost, extra = _match(reference, candidate, max_px=10.0)
    assert pairs == []
    assert lost == [0]
    assert extra == [0]


def test_match_keeps_multiple_instances_apart():
    reference = [_detection(100.0, 100.0), _detection(160.0, 100.0)]
    candidate = [_detection(160.0, 100.0), _detection(100.0, 100.0)]
    pairs, lost, extra = _match(reference, candidate, max_px=10.0)
    assert not lost
    assert not extra
    assert len(pairs) == 2
    assert len({reference_index for _, reference_index, _ in pairs}) == 2
    assert len({candidate_index for candidate_index, _, _ in pairs}) == 2
