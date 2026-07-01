from dataclasses import dataclass

import numpy as np

from mesh_to_inp.config import MacroStressConfig


@dataclass(frozen=True)
class FaceResultants:
    xmin: np.ndarray
    xmax: np.ndarray
    ymin: np.ndarray
    ymax: np.ndarray
    zmin: np.ndarray
    zmax: np.ndarray


def macro_stress_tensor(stress: MacroStressConfig) -> np.ndarray:
    return np.array(
        [
            [stress.sxx, stress.sxy, stress.sxz],
            [stress.sxy, stress.syy, stress.syz],
            [stress.sxz, stress.syz, stress.szz],
        ],
        dtype=float,
    )


def compute_rve_lengths(points: np.ndarray) -> tuple[float, float, float]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)

    lengths = maxs - mins

    if np.any(lengths <= 0.0):
        raise ValueError("RVE bounding box has zero or negative length.")

    return float(lengths[0]), float(lengths[1]), float(lengths[2])


def compute_face_resultants(
    points: np.ndarray,
    stress: MacroStressConfig,
) -> FaceResultants:
    sigma = macro_stress_tensor(stress)
    lx, ly, lz = compute_rve_lengths(points)

    ax = ly * lz
    ay = lx * lz
    az = lx * ly

    nx_pos = np.array([1.0, 0.0, 0.0])
    ny_pos = np.array([0.0, 1.0, 0.0])
    nz_pos = np.array([0.0, 0.0, 1.0])

    fx_pos = ax * sigma @ nx_pos
    fy_pos = ay * sigma @ ny_pos
    fz_pos = az * sigma @ nz_pos

    return FaceResultants(
        xmin=-fx_pos,
        xmax=fx_pos,
        ymin=-fy_pos,
        ymax=fy_pos,
        zmin=-fz_pos,
        zmax=fz_pos,
    )