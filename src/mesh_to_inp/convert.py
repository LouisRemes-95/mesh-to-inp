from pathlib import Path

import meshio
import numpy as np

from mesh_to_inp.mesh_processing import read_mesh_safe, build_region_separated_mesh
from mesh_to_inp.cohesive import extract_interface_triangles, make_cohesive_element_lines

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

    mesh = read_mesh_safe(input_path)

    key = next(iter(mesh.cell_data), None)
    if key is None:
        meshio.write(output_path, mesh, file_format="abaqus")
        return

    out_points, out_tetras, region_lut = build_region_separated_mesh(mesh, key)
    tris_regions = extract_interface_triangles(mesh, key)

    cohesive_mesh = meshio.Mesh(
        points=out_points,
        cells=[("tetra", out_tetras)],
    )

    meshio.write(output_path, cohesive_mesh, file_format="abaqus")

    lines = _read_lines(output_path)
    lines = _rewrite_abaqus_lines(lines)

    start_elem_id = _find_next_element_id(lines)
    lines.extend(make_cohesive_element_lines(tris_regions, region_lut, start_elem_id))
    lines.extend(["*END PART"])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


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