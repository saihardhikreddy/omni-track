"""Tests for the live mode's crowd maths (density grid, rank-1 SVD core, mesh)."""
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import topography as tp  # noqa: E402


def test_density_grid_counts_and_clamps():
    M = tp.density_grid([(0, 0), (639, 359), (640, 360), (-5, 10)], 640, 360, cols=4, rows=2)
    assert M.sum() == 4
    assert M[0, 0] == 2           # (0, 0) and the box hanging off the left edge
    assert M[1, 3] == 2           # the bottom-right corner and a point exactly on the edge
    # A centre a little off-frame (box half outside) used to index from the far end of the grid.
    assert M[0, 3] == 0


def test_centroids():
    assert tp.centroids([(0, 0, 10, 20), (5, 5, 6, 6)]) == [(5, 10), (5, 5)]


def test_isolated_group_is_an_anomaly():
    # A uniform crowd (3 people in each of a 4 x 3 block of cells) and one person far away.
    cell_w, cell_h = 1920 / tp.GRID_COLS, 1080 / tp.GRID_ROWS
    crowd = [(int((c + 0.5) * cell_w), int((r + 0.5) * cell_h)) for c in range(2, 6) for r in range(2, 5)] * 3
    M = tp.density_grid(crowd + [(1800, 1000)], 1920, 1080)
    core, anomalies = tp.classify_cells(M)
    assert anomalies == [(int(1000 / cell_h), int(1800 / cell_w))]
    assert sorted(core) == [(r, c) for r in range(2, 5) for c in range(2, 6)]


def test_two_equal_groups_are_both_core():
    # Two separate groups of the same size: neither is "the" crowd, so neither is an anomaly.
    # The SVD used to pick one at random and flag the other.
    M = np.zeros((tp.GRID_ROWS, tp.GRID_COLS))
    M[2, 3] = M[2, 4] = 3
    M[10, 20] = M[10, 21] = 3
    core, anomalies = tp.classify_cells(M)
    assert anomalies == []
    assert sorted(core) == [(2, 3), (2, 4), (10, 20), (10, 21)]
    # A smaller group is still an anomaly.
    M[10, 21] = 0
    M[10, 20] = 1
    assert tp.classify_cells(M)[1] == [(10, 20)]


def test_one_person_or_nobody_has_no_topography():
    assert tp.classify_cells(tp.density_grid([], 100, 100)) == ([], [])
    assert tp.classify_cells(tp.density_grid([(50, 50)], 100, 100)) == ([], [])


def test_mesh_is_fully_connected():
    pts = [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert len(tp.mesh_edges(pts)) == 6
    assert tp.mesh_edges(pts[:1]) == []


def test_cell_rect_tiles_the_frame():
    assert tp.cell_rect(0, 0, 1280, 720) == (0, 0, 40, 40)
    assert tp.cell_rect(17, 31, 1280, 720) == (1240, 680, 1280, 720)
    assert tp.cell_rect(1, 2, 1280, 720)[:2] == tp.cell_rect(0, 1, 1280, 720)[2:]


def test_label_stays_on_screen_for_the_top_row():
    # Cells in the top row used to get their label at y = -5, off the frame.
    x, y = tp.label_origin(tp.cell_rect(0, 5, 1280, 720))
    assert y > 0 and x >= 200
    # Lower cells keep the label just above the cell.
    assert tp.label_origin(tp.cell_rect(4, 5, 1280, 720)) == (200, 155)


def test_label_stays_on_screen_for_the_last_columns():
    # "ANOMALY" is 74 px wide; on a 640 px frame the last columns are 20 px wide, so a label
    # starting at the cell ran off the right edge.
    x, y = tp.label_origin(tp.cell_rect(4, 31, 640, 360), text_w=74, frame_w=640)
    assert x + 74 <= 640 and y == tp.label_origin(tp.cell_rect(4, 31, 640, 360))[1]
    x, _ = tp.label_origin(tp.cell_rect(0, 31, 640, 360), text_w=74, frame_w=640)  # top-right corner
    assert x + 74 <= 640
    # Labels that fit keep their place, and a frame narrower than the label starts it at 0.
    assert tp.label_origin(tp.cell_rect(4, 5, 1280, 720), text_w=74, frame_w=1280) == (200, 155)
    assert tp.label_origin((10, 100, 20, 110), text_w=74, frame_w=50)[0] == 0
