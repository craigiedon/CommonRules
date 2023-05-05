import numpy as np


def rot_mat(r: float) -> np.ndarray:
    return np.array([
        [np.cos(r), -np.sin(r)],
        [np.sin(r), np.cos(r)]]
    )


def angle_diff(a1: float, a2: float) -> float:
    return np.arctan2(np.sin(a1 - a2), np.cos(a1 - a2))