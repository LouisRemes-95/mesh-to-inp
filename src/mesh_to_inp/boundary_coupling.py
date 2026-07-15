from dataclasses import dataclass

import numpy as np


C3D4_FACE_NODES = {
    "S1": (0, 1, 2),
    "S2": (0, 3, 1),
    "S3": (1, 3, 2),
    "S4": (2, 3, 0),
}


@dataclass(frozen=True)
class BoundaryFaceRef:
    element_id: int
    face_label: str


@dataclass(frozen=True)
class BoundaryPatch:
    side: str
    region: int
    surface_name: str
    rp_name: str
    coupling_name: str
    rp_node_id: int
    rp_position: np.ndarray
    imposed_displacement: np.ndarray
    faces: list[BoundaryFaceRef]


def build_boundary_patches_for_macro_strain(
    points: np.ndarray,
    out_tetras: np.ndarray,
    original_tetras: np.ndarray,
    original_regions: np.ndarray,
    original_to_output_element_id: np.ndarray,
    region_lut: dict[int, np.ndarray],
    macro_strain,
    rp_start_node_id: int,
    boundary_band_fraction: float = 1.0e-6,
) -> list[BoundaryPatch]:
    """
    Build one external boundary patch per RVE side and region.

    Assumption:
        RVE boundaries are flat/aligned with x/y/z min/max.

    One RP is created for each (side, region) patch.
    """

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    lengths = maxs - mins
    center = 0.5 * (mins + maxs)

    band = boundary_band_fraction * max(float(lengths.max()), 1.0)

    strain = _macro_strain_tensor(macro_strain)

    external_faces = _extract_external_faces(
        original_tetras=original_tetras,
        original_regions=original_regions,
        original_to_output_element_id=original_to_output_element_id,
    )

    faces_by_patch: dict[tuple[str, int], list[BoundaryFaceRef]] = {}
    patch_points: dict[tuple[str, int], list[np.ndarray]] = {}

    for face_nodes, region, output_elem_id, face_label in external_faces:
        coords = points[list(face_nodes)]
        centroid = coords.mean(axis=0)

        side = _classify_boundary_side(
            centroid=centroid,
            mins=mins,
            maxs=maxs,
            band=band,
        )

        if side is None:
            continue

        key = (side, region)

        faces_by_patch.setdefault(key, []).append(
            BoundaryFaceRef(
                element_id=output_elem_id,
                face_label=face_label,
            )
        )

        # Use duplicated output node positions for RP position estimate.
        lut = region_lut[region]
        for original_node in face_nodes:
            output_node = int(lut[int(original_node)])
            if output_node >= 0:
                patch_points.setdefault(key, []).append(points[int(original_node)])

    patches: list[BoundaryPatch] = []
    next_rp_node_id = rp_start_node_id

    for (side, region), faces in sorted(faces_by_patch.items()):
        pts = patch_points[(side, region)]
        rp_position = np.mean(np.asarray(pts), axis=0)
        imposed_displacement = strain @ (rp_position - center)

        suffix = f"{side}_R{region}"

        patches.append(
            BoundaryPatch(
                side=side,
                region=region,
                surface_name=f"S_{suffix}",
                rp_name=f"RP_{suffix}",
                coupling_name=f"COUP_{suffix}",
                rp_node_id=next_rp_node_id,
                rp_position=rp_position,
                imposed_displacement=imposed_displacement,
                faces=faces,
            )
        )

        next_rp_node_id += 1

    return patches


def _extract_external_faces(
    original_tetras: np.ndarray,
    original_regions: np.ndarray,
    original_to_output_element_id: np.ndarray,
) -> list[tuple[tuple[int, int, int], int, int, str]]:
    face_map: dict[tuple[int, int, int], list[tuple[tuple[int, int, int], int, int, str]]] = {}

    for tet_id, tet in enumerate(original_tetras):
        region = int(original_regions[tet_id])
        output_elem_id = int(original_to_output_element_id[tet_id])

        if output_elem_id <= 0:
            raise ValueError(f"Invalid output element id for tetra {tet_id}.")

        for face_label, local_nodes in C3D4_FACE_NODES.items():
            face_nodes = tuple(int(tet[i]) for i in local_nodes)
            face_key = tuple(sorted(face_nodes))

            face_map.setdefault(face_key, []).append(
                (face_nodes, region, output_elem_id, face_label)
            )

    external_faces = []

    for entries in face_map.values():
        if len(entries) == 1:
            external_faces.append(entries[0])

    return external_faces


def _classify_boundary_side(
    centroid: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    band: float,
) -> str | None:
    distances = {
        "XMIN": abs(float(centroid[0] - mins[0])),
        "XMAX": abs(float(centroid[0] - maxs[0])),
        "YMIN": abs(float(centroid[1] - mins[1])),
        "YMAX": abs(float(centroid[1] - maxs[1])),
        "ZMIN": abs(float(centroid[2] - mins[2])),
        "ZMAX": abs(float(centroid[2] - maxs[2])),
    }

    side, distance = min(distances.items(), key=lambda item: item[1])

    if distance <= band:
        return side

    return None


def _macro_strain_tensor(macro_strain) -> np.ndarray:
    return np.asarray(
        [
            [macro_strain.exx, macro_strain.exy, macro_strain.exz],
            [macro_strain.exy, macro_strain.eyy, macro_strain.eyz],
            [macro_strain.exz, macro_strain.eyz, macro_strain.ezz],
        ],
        dtype=float,
    )