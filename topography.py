"""Crowd topography maths used by the live mode in app.py, kept free of camera and model code
so it can be tested on its own.

Each tracked person's box centre is counted into a coarse grid (the density matrix M). The
rank-1 SVD approximation of M is the crowd's dominant spatial pattern; an occupied cell that
the pattern barely explains (under 20 % of its peak) is flagged as an anomaly, e.g. a small
isolated group away from the main crowd.
"""

import itertools

import numpy as np

GRID_COLS, GRID_ROWS = 32, 18
CORE_FRACTION = 0.2


def centroids(boxes):
    """Integer centre (cx, cy) of each (x1, y1, x2, y2) box."""
    return [(int((x1 + x2) / 2), int((y1 + y2) / 2)) for (x1, y1, x2, y2) in boxes]


def density_grid(points, frame_w, frame_h, cols=GRID_COLS, rows=GRID_ROWS):
    """Count points into a rows x cols grid over the frame. Points on or past the edge go in the last cell."""
    M = np.zeros((rows, cols), dtype=np.float64)
    for cx, cy in points:
        gx = min(max(int((cx / frame_w) * cols), 0), cols - 1)
        gy = min(max(int((cy / frame_h) * rows), 0), rows - 1)
        M[gy, gx] += 1
    return M


def rank1_core(M):
    """|rank-1 SVD approximation| of M (all zeros when M has fewer than two people)."""
    if np.sum(M) <= 1:
        return np.zeros_like(M)
    U, S, Vh = np.linalg.svd(M, full_matrices=False)
    return np.abs(S[0] * np.outer(U[:, 0], Vh[0, :]))


def classify_cells(M, core_fraction=CORE_FRACTION):
    """-> (core cells, anomaly cells) as lists of (row, col), for occupied cells only."""
    core_m = rank1_core(M)
    if not core_m.any():
        return [], []
    limit = core_fraction * float(core_m.max())
    core, anomalies = [], []
    for row, col in zip(*np.nonzero(M)):
        (core if core_m[row, col] > limit else anomalies).append((int(row), int(col)))
    return core, anomalies


def cell_rect(row, col, frame_w, frame_h, cols=GRID_COLS, rows=GRID_ROWS):
    """Pixel corners (x1, y1, x2, y2) of grid cell (row, col) on a frame_w x frame_h frame."""
    return (int(col / cols * frame_w), int(row / rows * frame_h),
            int((col + 1) / cols * frame_w), int((row + 1) / rows * frame_h))


def label_origin(rect, text_h=12, gap=5):
    """Where to write a cell's label: just above the cell, or just inside its top edge when
    the cell touches the top of the frame and there is no room above (text drawn at a
    negative y is simply lost)."""
    x1, y1 = rect[0], rect[1]
    return (x1, y1 - gap) if y1 - gap - text_h >= 0 else (x1 + 3, y1 + text_h + gap)


def mesh_edges(points):
    """Every pair of points: the fully connected K_n graph drawn between people."""
    return list(itertools.combinations(points, 2))
