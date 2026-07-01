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


@dataclass(frozen=True)
class BoundaryTributaryAreas:
    xmin: dict[int, float]
    xmax: dict[int, float]
    ymin: dict[int, float]
    ymax: dict[int, float]
    zmin: dict[int, float]
    zmax: dict[int, float]

@dataclass(frozen=True)
class NodalForces:
    forces: dict[int, np.ndarray]


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

    fx_pos = ax * sigma @ np.array([1.0, 0.0, 0.0])
    fy_pos = ay * sigma @ np.array([0.0, 1.0, 0.0])
    fz_pos = az * sigma @ np.array([0.0, 0.0, 1.0])

    return FaceResultants(
        xmin=-fx_pos,
        xmax=fx_pos,
        ymin=-fy_pos,
        ymax=fy_pos,
        zmin=-fz_pos,
        zmax=fz_pos,
    )


def compute_boundary_tributary_areas(
    points: np.ndarray,
    tetras: np.ndarray,
    tetra_regions: np.ndarray,
    region_lut: dict[int, np.ndarray],
    boundary_band_fraction: float = 0.01,
    normal_tolerance: float = 0.9,
) -> BoundaryTributaryAreas:
    """
    Compute tributary areas on the six outer RVE boundary faces.

    Strategy:
    - Use original mesh topology, before duplicated interface nodes.
    - Keep only triangular faces that appear once in the original tetra mesh.
    - Use bounding-box slabs to reject internal void/hole surfaces.
    - Use normal alignment to keep only faces compatible with each box side.
    - Use projected area, not raw triangle area.
    - Return final duplicated Abaqus node IDs.
    """

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    lengths = maxs - mins

    if np.any(lengths <= 0.0):
        raise ValueError("RVE bounding box has zero or negative length.")

    if boundary_band_fraction <= 0.0:
        raise ValueError("boundary_band_fraction must be > 0.")

    if not (0.0 <= normal_tolerance <= 1.0):
        raise ValueError("normal_tolerance must be between 0 and 1.")

    bands = boundary_band_fraction * lengths

    external_triangles = extract_external_triangles_with_region(
        tetras=tetras,
        tetra_regions=tetra_regions,
    )

    return BoundaryTributaryAreas(
        xmin=_tributary_areas_on_box_side(
            points=points,
            triangles_with_region=external_triangles,
            axis=0,
            side="min",
            side_value=mins[0],
            band=bands[0],
            face_normal=np.array([-1.0, 0.0, 0.0]),
            region_lut=region_lut,
            normal_tolerance=normal_tolerance,
        ),
        xmax=_tributary_areas_on_box_side(
            points=points,
            triangles_with_region=external_triangles,
            axis=0,
            side="max",
            side_value=maxs[0],
            band=bands[0],
            face_normal=np.array([1.0, 0.0, 0.0]),
            region_lut=region_lut,
            normal_tolerance=normal_tolerance,
        ),
        ymin=_tributary_areas_on_box_side(
            points=points,
            triangles_with_region=external_triangles,
            axis=1,
            side="min",
            side_value=mins[1],
            band=bands[1],
            face_normal=np.array([0.0, -1.0, 0.0]),
            region_lut=region_lut,
            normal_tolerance=normal_tolerance,
        ),
        ymax=_tributary_areas_on_box_side(
            points=points,
            triangles_with_region=external_triangles,
            axis=1,
            side="max",
            side_value=maxs[1],
            band=bands[1],
            face_normal=np.array([0.0, 1.0, 0.0]),
            region_lut=region_lut,
            normal_tolerance=normal_tolerance,
        ),
        zmin=_tributary_areas_on_box_side(
            points=points,
            triangles_with_region=external_triangles,
            axis=2,
            side="min",
            side_value=mins[2],
            band=bands[2],
            face_normal=np.array([0.0, 0.0, -1.0]),
            region_lut=region_lut,
            normal_tolerance=normal_tolerance,
        ),
        zmax=_tributary_areas_on_box_side(
            points=points,
            triangles_with_region=external_triangles,
            axis=2,
            side="max",
            side_value=maxs[2],
            band=bands[2],
            face_normal=np.array([0.0, 0.0, 1.0]),
            region_lut=region_lut,
            normal_tolerance=normal_tolerance,
        ),
    )


def extract_external_triangles_with_region(
    tetras: np.ndarray,
    tetra_regions: np.ndarray,
) -> np.ndarray:
    """
    Extract triangular faces appearing only once in the original tetra mesh.

    Returns rows:
        node_1, node_2, node_3, region_id

    This excludes tetra-tetra internal faces and material interfaces.
    It does not by itself exclude internal void/hole surfaces.
    Those are filtered later by the bounding-box slab criterion.
    """

    local_faces = np.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [1, 2, 3],
            [0, 3, 2],
        ],
        dtype=int,
    )

    tris = tetras[:, local_faces].reshape(-1, 3)
    regions = np.repeat(tetra_regions, 4)

    sorted_tris = np.sort(tris, axis=1)

    _, inverse, counts = np.unique(
        sorted_tris,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )

    is_external = counts[inverse] == 1

    return np.column_stack([tris[is_external], regions[is_external]])


def _tributary_areas_on_box_side(
    points: np.ndarray,
    triangles_with_region: np.ndarray,
    axis: int,
    side: str,
    side_value: float,
    band: float,
    face_normal: np.ndarray,
    region_lut: dict[int, np.ndarray],
    normal_tolerance: float,
) -> dict[int, float]:
    triangles = triangles_with_region[:, :3].astype(int)
    regions = triangles_with_region[:, 3].astype(int)

    tri_points = points[triangles]
    centroids = tri_points.mean(axis=1)
    normals = _triangle_normals(tri_points)

    # The normal orientation from tetra ordering can be inward or outward.
    # Use absolute alignment only to check parallelism to the target box side.
    alignment = np.abs(normals @ face_normal)

    if side == "min":
        in_slab = centroids[:, axis] <= side_value + band
    elif side == "max":
        in_slab = centroids[:, axis] >= side_value - band
    else:
        raise ValueError("side must be either 'min' or 'max'.")

    accepted = in_slab & (alignment >= normal_tolerance)

    accepted_triangles = triangles[accepted]
    accepted_regions = regions[accepted]
    accepted_normals = normals[accepted]

    tributary: dict[int, float] = {}

    for tri, region_id, normal in zip(
        accepted_triangles,
        accepted_regions,
        accepted_normals,
    ):
        raw_area = _triangle_area(
            points[tri[0]],
            points[tri[1]],
            points[tri[2]],
        )

        # Projected area onto the nominal RVE face.
        projected_area = raw_area * abs(float(normal @ face_normal))

        contribution = projected_area / 3.0

        for original_node_id in tri:
            # region_lut maps original node id -> zero-based duplicated node id.
            abaqus_node_id = int(region_lut[int(region_id)][int(original_node_id)]) + 1
            tributary[abaqus_node_id] = tributary.get(abaqus_node_id, 0.0) + contribution

    return tributary


def _triangle_normals(tri_points: np.ndarray) -> np.ndarray:
    raw_normals = np.cross(
        tri_points[:, 1, :] - tri_points[:, 0, :],
        tri_points[:, 2, :] - tri_points[:, 0, :],
    )

    lengths = np.linalg.norm(raw_normals, axis=1)

    normals = np.zeros_like(raw_normals)
    nonzero = lengths > 0.0
    normals[nonzero] = raw_normals[nonzero] / lengths[nonzero, None]

    return normals


def _triangle_area(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(p1 - p0, p2 - p0)))


def debug_boundary_detection(
    points: np.ndarray,
    tetras: np.ndarray,
    tetra_regions: np.ndarray,
    boundary_band_fraction: float = 0.01,
    normal_tolerance: float = 0.5,
) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    lengths = maxs - mins
    bands = boundary_band_fraction * lengths

    external_triangles = extract_external_triangles_with_region(
        tetras=tetras,
        tetra_regions=tetra_regions,
    )

    triangles = external_triangles[:, :3].astype(int)
    tri_points = points[triangles]
    centroids = tri_points.mean(axis=1)
    normals = _triangle_normals(tri_points)

    print("Boundary detection debug:")
    print(f"  mins: {mins}")
    print(f"  maxs: {maxs}")
    print(f"  lengths: {lengths}")
    print(f"  boundary_band_fraction: {boundary_band_fraction}")
    print(f"  bands: {bands}")
    print(f"  normal_tolerance: {normal_tolerance}")
    print(f"  external triangles: {len(external_triangles)}")

    checks = [
        ("XMIN", 0, "min", mins[0], bands[0], np.array([-1.0, 0.0, 0.0])),
        ("XMAX", 0, "max", maxs[0], bands[0], np.array([1.0, 0.0, 0.0])),
        ("YMIN", 1, "min", mins[1], bands[1], np.array([0.0, -1.0, 0.0])),
        ("YMAX", 1, "max", maxs[1], bands[1], np.array([0.0, 1.0, 0.0])),
        ("ZMIN", 2, "min", mins[2], bands[2], np.array([0.0, 0.0, -1.0])),
        ("ZMAX", 2, "max", maxs[2], bands[2], np.array([0.0, 0.0, 1.0])),
    ]

    for name, axis, side, side_value, band, face_normal in checks:
        if side == "min":
            in_slab = centroids[:, axis] <= side_value + band
        else:
            in_slab = centroids[:, axis] >= side_value - band

        alignment = np.abs(normals @ face_normal)
        aligned = alignment >= normal_tolerance
        accepted = in_slab & aligned

        projected_area = 0.0

        for tri, normal in zip(triangles[accepted], normals[accepted]):
            raw_area = _triangle_area(
                points[tri[0]],
                points[tri[1]],
                points[tri[2]],
            )
            projected_area += raw_area * abs(float(normal @ face_normal))

        print(
            f"  {name}: "
            f"slab={int(np.sum(in_slab))}, "
            f"aligned={int(np.sum(aligned))}, "
            f"accepted={int(np.sum(accepted))}, "
            f"projected_area={projected_area}"
        )

def compute_nodal_forces_from_face_resultants(
    face_resultants: FaceResultants,
    tributary_areas: BoundaryTributaryAreas,
) -> NodalForces:
    nodal_forces: dict[int, np.ndarray] = {}

    _add_face_nodal_forces(
        nodal_forces,
        tributary_areas.xmin,
        face_resultants.xmin,
        face_name="XMIN",
    )
    _add_face_nodal_forces(
        nodal_forces,
        tributary_areas.xmax,
        face_resultants.xmax,
        face_name="XMAX",
    )
    _add_face_nodal_forces(
        nodal_forces,
        tributary_areas.ymin,
        face_resultants.ymin,
        face_name="YMIN",
    )
    _add_face_nodal_forces(
        nodal_forces,
        tributary_areas.ymax,
        face_resultants.ymax,
        face_name="YMAX",
    )
    _add_face_nodal_forces(
        nodal_forces,
        tributary_areas.zmin,
        face_resultants.zmin,
        face_name="ZMIN",
    )
    _add_face_nodal_forces(
        nodal_forces,
        tributary_areas.zmax,
        face_resultants.zmax,
        face_name="ZMAX",
    )

    return NodalForces(forces=nodal_forces)


def _add_face_nodal_forces(
    nodal_forces: dict[int, np.ndarray],
    face_areas: dict[int, float],
    face_resultant: np.ndarray,
    face_name: str,
) -> None:
    total_area = sum(face_areas.values())

    if total_area <= 0.0:
        if np.linalg.norm(face_resultant) > 0.0:
            raise ValueError(
                f"Cannot distribute non-zero resultant on {face_name}: "
                "tributary area is zero."
            )
        return

    for node_id, area in face_areas.items():
        weight = area / total_area
        force = weight * face_resultant

        if node_id not in nodal_forces:
            nodal_forces[node_id] = np.zeros(3, dtype=float)

        nodal_forces[node_id] += force


def print_nodal_force_summary(nodal_forces: NodalForces) -> None:
    total = np.zeros(3, dtype=float)

    for force in nodal_forces.forces.values():
        total += force

    print("Nodal force summary:")
    print(f"  number of loaded nodes: {len(nodal_forces.forces)}")
    print(f"  total force: {total}")