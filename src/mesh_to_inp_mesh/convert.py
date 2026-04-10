from pathlib import Path

import meshio
import numpy as np

from mesh_to_inp_mesh.errors import UserError


def convert(input_path, output_path: Path) -> None:
    """
    Convert a meshio mesh file to Abaqus part and insert cohesive elements
    between tetrahedral regions.

    Inputs 
    ------ 
    input_path: 
        Input mesh path 
    output_path: 
        Output .inp path
    """

    mesh = _read_mesh_safe(input_path)

    key = next(iter(mesh.cell_data), None)
    if key is None:
        meshio.write(output_path, mesh, file_format="abaqus")
        return

    out_points, out_tetras, region_lut = _build_region_separated_mesh(mesh, key)
    tris_regions = _extract_interface_triangles(mesh, key)

    cohesive_mesh = meshio.Mesh(
        points=out_points,
        cells=[("tetra", out_tetras)],
    )

    meshio.write(output_path, cohesive_mesh, file_format="abaqus")

    lines = _read_lines(output_path)
    lines = _rewrite_abaqus_lines(lines)

    start_elem_id = _find_next_element_id(lines)
    lines.extend(_make_cohesive_element_lines(tris_regions, region_lut, start_elem_id))
    lines.extend(["*END PART"])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _read_mesh_safe(path: Path):
    try:
        return meshio.read(path)

    except FileNotFoundError:
        raise UserError(f"Input file does not exist: {path}")

    except PermissionError:
        raise UserError(f"Cannot read file (permission denied): {path}")

    except meshio._exceptions.ReadError:
        raise UserError(f"File is not a valid mesh or unsupported format: {path}")


def _build_region_separated_mesh(mesh: meshio.Mesh, key: str):
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

        lut = np.full(mesh.points.shape[0], 0, dtype=_smallest_uint_dtype(region_points.size + offset - 1))
        lut[region_points] = np.arange(region_points.size) + offset
        region_lut[int(region_id)] = lut

        points_chunks.append(mesh.points[region_points, :])
        tetras_chunks.append(lut[region_tetras].astype(np.int64))

        offset += region_points.size

    out_points = np.vstack(points_chunks)
    out_tetras = np.vstack(tetras_chunks)
    return out_points, out_tetras, region_lut


def _extract_interface_triangles(mesh: meshio.Mesh, key: str) -> np.ndarray:
    tris = mesh.cells_dict["tetra"][:, [[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]]].reshape(-1, 3)
    regions = np.repeat(mesh.cell_data_dict[key]["tetra"], 4)[:, None]
    sorted_tris_region = np.hstack([np.sort(tris, axis=1), regions])

    order_by_region = np.argsort(sorted_tris_region[:, -1])
    tris = tris[order_by_region, :]
    sorted_tris_region = sorted_tris_region[order_by_region, :]

    _, inverse, counts = np.unique(sorted_tris_region, axis=0, return_inverse=True, return_counts=True)
    is_boundary = counts[inverse] == 1
    tris = tris[is_boundary, :]
    sorted_tris_region = sorted_tris_region[is_boundary, :]

    _, index, inverse = np.unique(sorted_tris_region[:, :3], axis=0, return_index=True, return_inverse=True)
    tris_regions = np.hstack([tris, sorted_tris_region[:, 3:], sorted_tris_region[index[inverse], 3:]])
    keep = np.ones(len(tris_regions), dtype=bool)
    keep[index] = False
    return tris_regions[keep]


def _read_lines(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]
    

def _rewrite_abaqus_lines(lines: list[str]) -> list[str]:
    header = []
    body = []
    in_header = False

    for stripped in lines:
        if stripped.startswith("*HEADING"):
            in_header = True
            header.append(stripped)
            continue

        if stripped.startswith("*") and in_header:
            in_header = False
            body.append("*PART, NAME=PART")

        if in_header:
            header.append(stripped)
            continue

        if stripped == "*ELEMENT, TYPE=C3D4":
            stripped = "*ELEMENT, TYPE=C3D4, ELSET=TETRA"

        body.append(stripped)

    if not header:
        return body

    return [header[0], " ".join(header[1:]), "Automatic python generated cohesive elements", *body]


def _find_next_element_id(lines: list[str]) -> int:
    for line in reversed(lines):
        if line and not line.startswith("*"):
            return int(line.split(",")[0]) + 1
    raise ValueError("Could not find any element definition line.")


def _make_cohesive_element_lines( tris_regions: np.ndarray, region_lut: dict[int, np.ndarray], start_elem_id: int) -> list[str]:
    lines = ["*ELEMENT, TYPE=COH3D6, ELSET=COHESIVE"]

    for i, cohe_elem in enumerate(tris_regions):
        lines.append(",".join(map(str, np.concatenate(([start_elem_id + i], region_lut[(cohe_elem[3])][cohe_elem[:3]].astype(np.int64)+1, region_lut[(cohe_elem[4])][cohe_elem[:3]].astype(np.int64)+1)))))

    return lines


def _smallest_uint_dtype(max_value: int):
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if max_value <= np.iinfo(dtype).max:
            return dtype
    raise ValueError("Value too large")