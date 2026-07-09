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


def build_region_separated_mesh(mesh, key: str):
    points = mesh.points
    tetra_cells = mesh.cells_dict["tetra"]
    tetra_regions = mesh.cell_data_dict[key]["tetra"]

    unique_regions = sorted(set(int(region) for region in tetra_regions))

    region_lut: dict[int, np.ndarray] = {}
    out_points: list[np.ndarray] = []

    for region in unique_regions:
        used_nodes = np.unique(tetra_cells[tetra_regions == region].ravel())

        lut = np.full(len(points), -1, dtype=int)

        for original_node_id in used_nodes:
            lut[original_node_id] = len(out_points)
            out_points.append(points[original_node_id])

        region_lut[region] = lut

    out_tetras: list[np.ndarray] = []
    original_to_output_element_id = np.full(len(tetra_cells), -1, dtype=int)

    for original_tet_id, tet in enumerate(tetra_cells):
        region = int(tetra_regions[original_tet_id])
        mapped_tet = region_lut[region][tet]

        if np.any(mapped_tet < 0):
            raise ValueError(
                f"Failed to map tetra {original_tet_id} for region {region}."
            )

        out_tetras.append(mapped_tet)
        original_to_output_element_id[original_tet_id] = len(out_tetras)

    return (
        np.asarray(out_points),
        np.asarray(out_tetras, dtype=tetra_cells.dtype),
        region_lut,
        original_to_output_element_id,
    )


def smallest_uint_dtype(max_value: int):
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if max_value <= np.iinfo(dtype).max:
            return dtype
    raise ValueError("Value too large")