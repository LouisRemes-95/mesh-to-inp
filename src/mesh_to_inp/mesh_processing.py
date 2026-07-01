from pathlib import Path

import meshio
import numpy as np

from mesh_to_inp.errors import UserError



def read_mesh_safe(path: Path):
    try:
        return meshio.read(path)

    except FileNotFoundError:
        raise UserError(f"Input file does not exist: {path}")

    except PermissionError:
        raise UserError(f"Cannot read file (permission denied): {path}")

    except meshio._exceptions.ReadError:
        raise UserError(f"File is not a valid mesh or unsupported format: {path}")


def build_region_separated_mesh(mesh: meshio.Mesh, key: str):
    tetra_cells = mesh.cells_dict["tetra"]
    tetra_regions = mesh.cell_data_dict[key]["tetra"]

    region_lut: dict[int, np.ndarray] = {}
    points_chunks = []
    tetras_chunks = []
    offset = 0

    for region_id in np.unique(tetra_regions):
        region_mask = tetra_regions == region_id
        region_tetras = tetra_cells[region_mask, :]
        region_points = np.unique(region_tetras.ravel())

        lut = np.full(mesh.points.shape[0], 0, dtype=smallest_uint_dtype(region_points.size + offset - 1))
        lut[region_points] = np.arange(region_points.size) + offset
        region_lut[int(region_id)] = lut

        points_chunks.append(mesh.points[region_points, :])
        tetras_chunks.append(lut[region_tetras].astype(np.int64))

        offset += region_points.size

    out_points = np.vstack(points_chunks)
    out_tetras = np.vstack(tetras_chunks)
    return out_points, out_tetras, region_lut


def smallest_uint_dtype(max_value: int):
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if max_value <= np.iinfo(dtype).max:
            return dtype
    raise ValueError("Value too large")