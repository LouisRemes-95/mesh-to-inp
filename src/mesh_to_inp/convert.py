from pathlib import Path

import meshio

from mesh_to_inp.config import CaseConfig
from mesh_to_inp.mesh_processing import read_mesh_safe, build_region_separated_mesh
from mesh_to_inp.cohesive import extract_interface_triangles, make_cohesive_element_lines
from mesh_to_inp.abaqus_writer import (
    read_lines,
    rewrite_abaqus_lines,
    find_next_element_id,
    make_material_lines,
    make_solid_section_lines,
    make_cohesive_section_lines,
    make_assembly_lines,
    make_step_with_cloads_lines,
)
from mesh_to_inp.loading import (
    compute_face_resultants,
    compute_boundary_tributary_areas,
    compute_nodal_forces_from_face_resultants,
)

from mesh_to_inp.constraints import make_default_rigid_body_constraints


def convert(case: CaseConfig) -> None:
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

    input_path = case.mesh.input
    output_path = case.job.output

    mesh = read_mesh_safe(input_path)

    key = next(iter(mesh.cell_data), None)
    if key is None:
        meshio.write(output_path, mesh, file_format="abaqus")
        return

    out_points, out_tetras, region_lut = build_region_separated_mesh(mesh, key)
    tris_regions = extract_interface_triangles(mesh, key)
    rigid_body_constraints = make_default_rigid_body_constraints(out_points)

    face_resultants = compute_face_resultants(out_points, case.macro_stress)

    original_tetras = mesh.cells_dict["tetra"]
    original_regions = mesh.cell_data_dict[key]["tetra"]

    tributary_areas = compute_boundary_tributary_areas(
        points=mesh.points,
        tetras=original_tetras,
        tetra_regions=original_regions,
        region_lut=region_lut,
    )

    nodal_forces = compute_nodal_forces_from_face_resultants(
        face_resultants=face_resultants,
        tributary_areas=tributary_areas,
    )

    cohesive_mesh = meshio.Mesh(
        points=out_points,
        cells=[("tetra", out_tetras)],
    )

    meshio.write(output_path, cohesive_mesh, file_format="abaqus")

    lines = read_lines(output_path)
    lines = rewrite_abaqus_lines(lines)

    start_elem_id = find_next_element_id(lines)
    lines.extend(make_cohesive_element_lines(tris_regions, region_lut, start_elem_id))

    if case.solid_section:
        lines.extend(make_solid_section_lines(case.solid_section))

    if case.cohesive_section:
        lines.extend(make_cohesive_section_lines(case.cohesive_section))

    lines.extend(["*END PART"])

    if case.materials:
        lines.extend(make_material_lines(case.materials))

    lines.extend(make_assembly_lines())
    lines.extend(
        make_step_with_cloads_lines(
            nodal_forces=nodal_forces,
            step=case.step,
            rigid_body_constraints=rigid_body_constraints,
        )
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))