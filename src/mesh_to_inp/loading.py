from dataclasses import dataclass

import numpy as np


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


@dataclass(frozen=True)
class FaceForceContributions:
    forces_by_face: dict[str, dict[int, np.ndarray]]


@dataclass(frozen=True)
class ForceMomentSummary:
    total_force: np.ndarray
    total_moment: np.ndarray


def macro_stress_tensor(stress) -> np.ndarray:
    return np.asarray(
        [
            [stress.sxx, stress.sxy, stress.sxz],
            [stress.sxy, stress.syy, stress.syz],
            [stress.sxz, stress.syz, stress.szz],
        ],
        dtype=float,
    )


def compute_rve_lengths(points: np.ndarray) -> np.ndarray:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)

    return maxs - mins


def compute_face_resultants(points: np.ndarray, stress) -> FaceResultants:
    """
    Compute nominal macro-stress resultants on the six bounding-box faces.

    Stress convention:
        [sxx sxy sxz]
        [sxy syy syz]
        [sxz syz szz]

    Face resultants:
        F = sigma @ n * A

    where n is the outward face normal and A is the nominal projected area.
    """

    sigma = macro_stress_tensor(stress)

    lx, ly, lz = compute_rve_lengths(points)

    ax = ly * lz
    ay = lx * lz
    az = lx * ly

    nxmin = np.asarray([-1.0, 0.0, 0.0])
    nxmax = np.asarray([1.0, 0.0, 0.0])
    nymin = np.asarray([0.0, -1.0, 0.0])
    nymax = np.asarray([0.0, 1.0, 0.0])
    nzmin = np.asarray([0.0, 0.0, -1.0])
    nzmax = np.asarray([0.0, 0.0, 1.0])

    return FaceResultants(
        xmin=ax * (sigma @ nxmin),
        xmax=ax * (sigma @ nxmax),
        ymin=ay * (sigma @ nymin),
        ymax=ay * (sigma @ nymax),
        zmin=az * (sigma @ nzmin),
        zmax=az * (sigma @ nzmax),
    )


def compute_boundary_tributary_areas(
    points: np.ndarray,
    tetras: np.ndarray,
    tetra_regions: np.ndarray,
    region_lut: dict[int, np.ndarray],
    boundary_band_fraction: float = 0.01,
    normal_tolerance: float = 0.5,
) -> BoundaryTributaryAreas:
    """
    Compute nodal tributary projected areas on the six exterior RVE faces.

    Important:
        The input tetras and points are the ORIGINAL mesh topology.
        The returned node ids are the DUPLICATED Abaqus node ids, 1-based.

    Why:
        The original topology is needed to detect true external faces.
        region_lut maps original node ids to duplicated output node ids.
    """

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    lengths = maxs - mins

    max_length = float(lengths.max())
    band = boundary_band_fraction * max_length

    external_triangles = extract_external_triangles_with_region(
        tetras=tetras,
        tetra_regions=tetra_regions,
    )

    xmin = _tributary_areas_on_box_side(
        points=points,
        triangles_with_region=external_triangles,
        axis=0,
        side="min",
        side_value=mins[0],
        band=band,
        face_normal=np.asarray([-1.0, 0.0, 0.0]),
        region_lut=region_lut,
        normal_tolerance=normal_tolerance,
    )

    xmax = _tributary_areas_on_box_side(
        points=points,
        triangles_with_region=external_triangles,
        axis=0,
        side="max",
        side_value=maxs[0],
        band=band,
        face_normal=np.asarray([1.0, 0.0, 0.0]),
        region_lut=region_lut,
        normal_tolerance=normal_tolerance,
    )

    ymin = _tributary_areas_on_box_side(
        points=points,
        triangles_with_region=external_triangles,
        axis=1,
        side="min",
        side_value=mins[1],
        band=band,
        face_normal=np.asarray([0.0, -1.0, 0.0]),
        region_lut=region_lut,
        normal_tolerance=normal_tolerance,
    )

    ymax = _tributary_areas_on_box_side(
        points=points,
        triangles_with_region=external_triangles,
        axis=1,
        side="max",
        side_value=maxs[1],
        band=band,
        face_normal=np.asarray([0.0, 1.0, 0.0]),
        region_lut=region_lut,
        normal_tolerance=normal_tolerance,
    )

    zmin = _tributary_areas_on_box_side(
        points=points,
        triangles_with_region=external_triangles,
        axis=2,
        side="min",
        side_value=mins[2],
        band=band,
        face_normal=np.asarray([0.0, 0.0, -1.0]),
        region_lut=region_lut,
        normal_tolerance=normal_tolerance,
    )

    zmax = _tributary_areas_on_box_side(
        points=points,
        triangles_with_region=external_triangles,
        axis=2,
        side="max",
        side_value=maxs[2],
        band=band,
        face_normal=np.asarray([0.0, 0.0, 1.0]),
        region_lut=region_lut,
        normal_tolerance=normal_tolerance,
    )

    return BoundaryTributaryAreas(
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        zmin=zmin,
        zmax=zmax,
    )


def extract_external_triangles_with_region(
    tetras: np.ndarray,
    tetra_regions: np.ndarray,
) -> np.ndarray:
    """
    Extract true external tetra faces from the original tetra connectivity.

    Returns rows:
        n1, n2, n3, region_id

    A face is external if it appears only once in the tetra face hash.
    """

    face_pattern = np.asarray(
        [
            [0, 1, 2],
            [0, 3, 1],
            [1, 3, 2],
            [2, 3, 0],
        ],
        dtype=int,
    )

    face_records = []

    for tet_id, tet in enumerate(tetras):
        region = int(tetra_regions[tet_id])

        for local_face in face_pattern:
            face_nodes = tuple(int(tet[i]) for i in local_face)
            key = tuple(sorted(face_nodes))

            face_records.append(
                (
                    key,
                    face_nodes,
                    region,
                )
            )

    face_records.sort(key=lambda item: item[0])

    external_faces = []
    i = 0

    while i < len(face_records):
        key = face_records[i][0]
        j = i + 1

        while j < len(face_records) and face_records[j][0] == key:
            j += 1

        count = j - i

        if count == 1:
            _, face_nodes, region = face_records[i]
            external_faces.append((*face_nodes, region))

        i = j

    if not external_faces:
        return np.empty((0, 4), dtype=int)

    return np.asarray(external_faces, dtype=int)


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
    """
    Compute nodal projected tributary area for one bounding-box side.

    Selection:
        1. triangle centroid must lie within a coordinate slab near the side
        2. triangle normal must be sufficiently aligned with the side normal

    The projected triangle area is used:
        A_projected = A_triangle * abs(n_triangle dot n_face)
    """

    if triangles_with_region.size == 0:
        return {}

    tri_nodes = triangles_with_region[:, :3]
    tri_regions = triangles_with_region[:, 3]

    coords = points[tri_nodes]
    centroids = coords.mean(axis=1)

    if side == "min":
        slab_mask = centroids[:, axis] <= side_value + band
    elif side == "max":
        slab_mask = centroids[:, axis] >= side_value - band
    else:
        raise ValueError(f"Unknown side: {side}")

    normals = _triangle_normals(coords)
    alignment = np.abs(normals @ face_normal)

    aligned_mask = alignment >= normal_tolerance
    selected_mask = slab_mask & aligned_mask

    selected_tri_nodes = tri_nodes[selected_mask]
    selected_regions = tri_regions[selected_mask]
    selected_coords = coords[selected_mask]
    selected_alignment = alignment[selected_mask]

    tributary_areas: dict[int, float] = {}

    for nodes, region_id, triangle_coords, normal_alignment in zip(
        selected_tri_nodes,
        selected_regions,
        selected_coords,
        selected_alignment,
    ):
        raw_area = _triangle_area(triangle_coords)
        projected_area = raw_area * float(normal_alignment)
        nodal_share = projected_area / 3.0

        lut = region_lut[int(region_id)]

        for original_node_id in nodes:
            output_node_id = int(lut[int(original_node_id)])

            if output_node_id < 0:
                raise ValueError(
                    f"Original node {original_node_id} is not mapped for "
                    f"region {region_id}."
                )

            abaqus_node_id = output_node_id + 1
            tributary_areas[abaqus_node_id] = (
                tributary_areas.get(abaqus_node_id, 0.0) + nodal_share
            )

    return tributary_areas


def _triangle_normals(coords: np.ndarray) -> np.ndarray:
    """
    Compute unit normals for triangle coordinates with shape (n, 3, 3).
    """

    v1 = coords[:, 1, :] - coords[:, 0, :]
    v2 = coords[:, 2, :] - coords[:, 0, :]

    normals = np.cross(v1, v2)
    norms = np.linalg.norm(normals, axis=1)

    valid = norms > 0.0

    unit_normals = np.zeros_like(normals)
    unit_normals[valid] = normals[valid] / norms[valid, None]

    return unit_normals


def _triangle_area(coords: np.ndarray) -> float:
    v1 = coords[1] - coords[0]
    v2 = coords[2] - coords[0]

    return 0.5 * float(np.linalg.norm(np.cross(v1, v2)))


def compute_face_force_contributions(
    face_resultants: FaceResultants,
    tributary_areas: BoundaryTributaryAreas,
) -> FaceForceContributions:
    """
    Compute force contributions face by face.

    The resultant of each face is exactly preserved by construction.
    """

    forces_by_face: dict[str, dict[int, np.ndarray]] = {}

    _add_face_force_contributions(
        forces_by_face=forces_by_face,
        face_name="xmin",
        face_areas=tributary_areas.xmin,
        face_resultant=face_resultants.xmin,
    )
    _add_face_force_contributions(
        forces_by_face=forces_by_face,
        face_name="xmax",
        face_areas=tributary_areas.xmax,
        face_resultant=face_resultants.xmax,
    )
    _add_face_force_contributions(
        forces_by_face=forces_by_face,
        face_name="ymin",
        face_areas=tributary_areas.ymin,
        face_resultant=face_resultants.ymin,
    )
    _add_face_force_contributions(
        forces_by_face=forces_by_face,
        face_name="ymax",
        face_areas=tributary_areas.ymax,
        face_resultant=face_resultants.ymax,
    )
    _add_face_force_contributions(
        forces_by_face=forces_by_face,
        face_name="zmin",
        face_areas=tributary_areas.zmin,
        face_resultant=face_resultants.zmin,
    )
    _add_face_force_contributions(
        forces_by_face=forces_by_face,
        face_name="zmax",
        face_areas=tributary_areas.zmax,
        face_resultant=face_resultants.zmax,
    )

    return FaceForceContributions(forces_by_face=forces_by_face)


def _add_face_force_contributions(
    forces_by_face: dict[str, dict[int, np.ndarray]],
    face_name: str,
    face_areas: dict[int, float],
    face_resultant: np.ndarray,
) -> None:
    total_area = float(sum(face_areas.values()))

    if total_area <= 0.0:
        if np.linalg.norm(face_resultant) > 0.0:
            raise ValueError(
                f"Cannot distribute non-zero resultant on empty face {face_name}: "
                f"{face_resultant}"
            )

        forces_by_face[face_name] = {}
        return

    face_forces: dict[int, np.ndarray] = {}

    for node_id, area in face_areas.items():
        weight = float(area) / total_area
        face_forces[int(node_id)] = weight * face_resultant

    forces_by_face[face_name] = face_forces


def correct_moment_preserving_face_resultants(
    face_force_contributions: FaceForceContributions,
    points: np.ndarray,
    center: np.ndarray | None = None,
    correction_tolerance: float = 1.0e-10,
) -> FaceForceContributions:
    """
    Correct the force distribution to remove global residual moment while
    preserving the resultant force of each individual face.

    Constraints imposed on the correction:
        for each face:
            sum(delta_f_i) = 0

        globally:
            sum(r_i cross delta_f_i) = - residual_moment

    Here:
        r_i = x_i - center

    Node ids are Abaqus 1-based.
    """

    if center is None:
        center = 0.5 * (points.min(axis=0) + points.max(axis=0))

    summary_before = summarize_face_force_contributions(
        face_force_contributions=face_force_contributions,
        points=points,
        center=center,
    )

    residual_moment = summary_before.total_moment

    if np.linalg.norm(residual_moment) <= correction_tolerance:
        return face_force_contributions

    variables: list[tuple[str, int, int]] = []
    face_names = list(face_force_contributions.forces_by_face)

    for face_name in face_names:
        face_forces = face_force_contributions.forces_by_face[face_name]

        for node_id in sorted(face_forces):
            for component in range(3):
                variables.append((face_name, node_id, component))

    n_constraints = 3 * len(face_names) + 3
    n_variables = len(variables)

    if n_variables == 0:
        raise ValueError("Cannot correct moment: no force variables available.")

    A = np.zeros((n_constraints, n_variables), dtype=float)
    b = np.zeros(n_constraints, dtype=float)

    # Face resultant preservation constraints.
    face_row_offset: dict[str, int] = {}

    row = 0
    for face_name in face_names:
        face_row_offset[face_name] = row
        row += 3

    moment_row = row

    for col, (face_name, node_id, component) in enumerate(variables):
        # Preserve face force resultant:
        # sum(delta_f_component on this face) = 0
        A[face_row_offset[face_name] + component, col] = 1.0

        # Correct moment:
        # M = r x f
        position = points[node_id - 1]
        r = position - center
        rx, ry, rz = r

        if component == 0:
            # fx contributes:
            # Mx += 0
            # My += rz * fx
            # Mz += -ry * fx
            A[moment_row + 1, col] = rz
            A[moment_row + 2, col] = -ry

        elif component == 1:
            # fy contributes:
            # Mx += -rz * fy
            # My += 0
            # Mz += rx * fy
            A[moment_row + 0, col] = -rz
            A[moment_row + 2, col] = rx

        elif component == 2:
            # fz contributes:
            # Mx += ry * fz
            # My += -rx * fz
            # Mz += 0
            A[moment_row + 0, col] = ry
            A[moment_row + 1, col] = -rx

        else:
            raise ValueError(f"Invalid force component: {component}")

    b[moment_row : moment_row + 3] = -residual_moment

    correction, _residuals, rank, _singular_values = np.linalg.lstsq(
        A,
        b,
        rcond=None,
    )

    corrected: dict[str, dict[int, np.ndarray]] = {}

    for face_name, face_forces in face_force_contributions.forces_by_face.items():
        corrected[face_name] = {
            node_id: force.copy()
            for node_id, force in face_forces.items()
        }

    for value, (face_name, node_id, component) in zip(correction, variables):
        corrected[face_name][node_id][component] += value

    corrected_contributions = FaceForceContributions(forces_by_face=corrected)

    summary_after = summarize_face_force_contributions(
        face_force_contributions=corrected_contributions,
        points=points,
        center=center,
    )

    if np.linalg.norm(summary_after.total_moment) > correction_tolerance:
        print(
            "Warning: residual moment correction did not fully converge. "
            f"Before={summary_before.total_moment}, "
            f"After={summary_after.total_moment}, "
            f"rank={rank}"
        )

    _check_face_resultants_preserved(
        before=face_force_contributions,
        after=corrected_contributions,
        tolerance=correction_tolerance,
    )

    return corrected_contributions


def sum_face_force_contributions(
    face_force_contributions: FaceForceContributions,
) -> NodalForces:
    nodal_forces: dict[int, np.ndarray] = {}

    for face_forces in face_force_contributions.forces_by_face.values():
        for node_id, force in face_forces.items():
            if node_id not in nodal_forces:
                nodal_forces[node_id] = np.zeros(3, dtype=float)

            nodal_forces[node_id] += force

    return NodalForces(forces=nodal_forces)


def compute_nodal_forces_from_face_resultants(
    face_resultants: FaceResultants,
    tributary_areas: BoundaryTributaryAreas,
) -> NodalForces:
    """
    Backward-compatible old helper.

    Prefer the newer pipeline:
        compute_face_force_contributions
        correct_moment_preserving_face_resultants
        sum_face_force_contributions
    """

    face_force_contributions = compute_face_force_contributions(
        face_resultants=face_resultants,
        tributary_areas=tributary_areas,
    )

    return sum_face_force_contributions(face_force_contributions)


def summarize_face_force_contributions(
    face_force_contributions: FaceForceContributions,
    points: np.ndarray,
    center: np.ndarray | None = None,
) -> ForceMomentSummary:
    if center is None:
        center = 0.5 * (points.min(axis=0) + points.max(axis=0))

    total_force = np.zeros(3, dtype=float)
    total_moment = np.zeros(3, dtype=float)

    for face_forces in face_force_contributions.forces_by_face.values():
        for node_id, force in face_forces.items():
            position = points[node_id - 1]
            r = position - center

            total_force += force
            total_moment += np.cross(r, force)

    return ForceMomentSummary(
        total_force=total_force,
        total_moment=total_moment,
    )


def summarize_face_resultants(
    face_force_contributions: FaceForceContributions,
) -> dict[str, np.ndarray]:
    resultants: dict[str, np.ndarray] = {}

    for face_name, face_forces in face_force_contributions.forces_by_face.items():
        resultant = np.zeros(3, dtype=float)

        for force in face_forces.values():
            resultant += force

        resultants[face_name] = resultant

    return resultants


def _check_face_resultants_preserved(
    before: FaceForceContributions,
    after: FaceForceContributions,
    tolerance: float,
) -> None:
    before_resultants = summarize_face_resultants(before)
    after_resultants = summarize_face_resultants(after)

    for face_name in before_resultants:
        difference = after_resultants[face_name] - before_resultants[face_name]

        if np.linalg.norm(difference) > tolerance:
            raise ValueError(
                f"Moment correction changed face resultant for {face_name}. "
                f"Difference: {difference}"
            )


def print_force_moment_summary(
    label: str,
    face_force_contributions: FaceForceContributions,
    points: np.ndarray,
) -> None:
    summary = summarize_face_force_contributions(
        face_force_contributions=face_force_contributions,
        points=points,
    )

    print(label)
    print(f"  total force : {summary.total_force}")
    print(f"  total moment: {summary.total_moment}")

    face_resultants = summarize_face_resultants(face_force_contributions)

    for face_name, resultant in face_resultants.items():
        print(f"  {face_name:>4} resultant: {resultant}")


def print_nodal_force_summary(nodal_forces: NodalForces) -> None:
    total = np.zeros(3)

    for force in nodal_forces.forces.values():
        total += force

    print("Nodal force summary:")
    print(f"  number of loaded nodes: {len(nodal_forces.forces)}")
    print(f"  total force: {total}")


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
    band = boundary_band_fraction * float(lengths.max())

    external_triangles = extract_external_triangles_with_region(
        tetras=tetras,
        tetra_regions=tetra_regions,
    )

    print("Boundary detection debug:")
    print(f"  mins: {mins}")
    print(f"  maxs: {maxs}")
    print(f"  lengths: {lengths}")
    print(f"  band: {band}")
    print(f"  external triangles: {len(external_triangles)}")
    print(f"  normal tolerance: {normal_tolerance}")