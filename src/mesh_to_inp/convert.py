import meshio

from mesh_to_inp.config import CaseConfig
from mesh_to_inp.mesh_processing import read_mesh_safe, build_region_separated_mesh
from mesh_to_inp.interface_contact import build_interface_surface_pairs
from mesh_to_inp.abaqus_writer import (
    read_lines,
    rewrite_abaqus_lines,
    make_material_lines,
    make_solid_section_lines,
    make_assembly_lines,
    make_step_with_cloads_lines,
    make_interface_surface_lines,
    make_cohesive_contact_interaction_lines,
)
from mesh_to_inp.loading import (
    compute_face_resultants,
    compute_boundary_tributary_areas,
    compute_nodal_forces_from_face_resultants,
)
from mesh_to_inp.constraints import make_default_rigid_body_constraints


def convert(case: CaseConfig) -> None:
    """
    Convert a meshio mesh file to an Abaqus .inp file using cohesive contact.

    Interface mode:
        - duplicate nodes by material/region index
        - keep only C3D4 bulk tetrahedra
        - create element-based master/slave surfaces at region interfaces
        - apply cohesive/contact interaction between those surfaces
    """

    input_path = case.mesh.input
    output_path = case.job.output

    mesh = read_mesh_safe(input_path)

    key = next(iter(mesh.cell_data), None)
    if key is None:
        meshio.write(output_path, mesh, file_format="abaqus")
        return

    (
        out_points,
        out_tetras,
        region_lut,
        original_to_output_element_id,
    ) = build_region_separated_mesh(mesh, key)

    original_tetras = mesh.cells_dict["tetra"]
    original_regions = mesh.cell_data_dict[key]["tetra"]

    surface_pairs = build_interface_surface_pairs(
        tetras=original_tetras,
        tetra_regions=original_regions,
        original_to_output_element_id=original_to_output_element_id,
    )

    rigid_body_constraints = make_default_rigid_body_constraints(out_points)

    face_resultants = compute_face_resultants(
        out_points,
        case.macro_stress,
    )

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

    abaqus_mesh = meshio.Mesh(
        points=out_points,
        cells=[("tetra", out_tetras)],
    )

    meshio.write(output_path, abaqus_mesh, file_format="abaqus")

    lines = read_lines(output_path)
    lines = rewrite_abaqus_lines(lines)

    if case.solid_section:
        lines.extend(make_solid_section_lines(case.solid_section))

    lines.extend(["*END PART"])

    # In contact-interaction mode, only real bulk materials are written as *MATERIAL.
    # The cohesive/interface material data is used inside *SURFACE INTERACTION.
    bulk_materials = [
        material
        for material in case.materials
        if material.cohesive is None
    ]

    if bulk_materials:
        lines.extend(make_material_lines(bulk_materials))

    # Assembly must contain the surfaces, because *SURFACE is only allowed
    # inside PART, INSTANCE, or ASSEMBLY levels.
    assembly_lines = make_assembly_lines()

    end_assembly_index = assembly_lines.index("*END ASSEMBLY")

    assembly_lines = (
        assembly_lines[:end_assembly_index]
        + make_interface_surface_lines(surface_pairs=surface_pairs)
        + assembly_lines[end_assembly_index:]
    )

    lines.extend(assembly_lines)

    cohesive_contact_material = _find_cohesive_contact_material(case)

    lines.extend(
        make_cohesive_contact_interaction_lines(
            surface_pairs=surface_pairs,
            cohesive_material=cohesive_contact_material,
        )
    )

    lines.extend(
        make_step_with_cloads_lines(
            nodal_forces=nodal_forces,
            step=case.step,
            rigid_body_constraints=rigid_body_constraints,
        )
    )

    clean_lines = [
        line if line.strip() else "**"
        for line in lines
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(clean_lines))


def _find_cohesive_contact_material(case: CaseConfig):
    cohesive_materials = [
        material
        for material in case.materials
        if material.cohesive is not None
    ]

    if len(cohesive_materials) != 1:
        raise ValueError(
            "Expected exactly one cohesive/contact material in the case file."
        )

    return cohesive_materials[0]