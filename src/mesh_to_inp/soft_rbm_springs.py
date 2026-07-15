from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SoftRigidBodySpring:
    name: str
    selected_node_id: int
    ground_node_id: int
    element_id: int
    dof: int
    selected_position: np.ndarray
    ground_position: np.ndarray
    target_position: np.ndarray
    distance_to_target: float


@dataclass(frozen=True)
class SoftRigidBodySpringSet:
    stiffness: float
    springs: list[SoftRigidBodySpring]


def build_soft_rbm_springs(
    points: np.ndarray,
    stiffness: float,
    ground_node_start_id: int = 90_000_001,
    spring_element_start_id: int = 91_000_001,
) -> SoftRigidBodySpringSet:
    """
    Build six weak scalar SPRING2 constraints to regularize rigid-body motion.

    Node ids are Abaqus 1-based.

    Target corners and constrained directions:

        A: xmax ymax zmin, U1
        B: xmax ymin zmax, U1
        C: xmin ymax zmax, U1
        D: xmin ymin zmax, U2
        E: xmin ymax zmin, U2
        F: xmax ymin zmin, U3

    These six scalar springs control the six rigid-body modes through weak
    penalties instead of hard displacement constraints.
    """

    if stiffness <= 0.0:
        raise ValueError(
            f"Soft rigid-body spring stiffness must be positive. Got {stiffness}."
        )

    mins = points.min(axis=0)
    maxs = points.max(axis=0)

    xmin, ymin, zmin = mins
    xmax, ymax, zmax = maxs

    targets = [
        ("A_XMAX_YMAX_ZMIN_U1", np.asarray([xmax, ymax, zmin], dtype=float), 1),
        ("B_XMAX_YMIN_ZMAX_U1", np.asarray([xmax, ymin, zmax], dtype=float), 1),
        ("C_XMIN_YMAX_ZMAX_U1", np.asarray([xmin, ymax, zmax], dtype=float), 1),
        ("D_XMIN_YMIN_ZMAX_U2", np.asarray([xmin, ymin, zmax], dtype=float), 2),
        ("E_XMIN_YMAX_ZMIN_U2", np.asarray([xmin, ymax, zmin], dtype=float), 2),
        ("F_XMAX_YMIN_ZMIN_U3", np.asarray([xmax, ymin, zmin], dtype=float), 3),
    ]

    selected_nodes: set[int] = set()
    springs: list[SoftRigidBodySpring] = []

    for i, (name, target_position, dof) in enumerate(targets):
        selected_node_id, selected_position, distance = _nearest_unused_node(
            points=points,
            target_position=target_position,
            used_node_ids=selected_nodes,
        )

        selected_nodes.add(selected_node_id)

        ground_node_id = ground_node_start_id + i
        element_id = spring_element_start_id + i

        springs.append(
            SoftRigidBodySpring(
                name=name,
                selected_node_id=selected_node_id,
                ground_node_id=ground_node_id,
                element_id=element_id,
                dof=dof,
                selected_position=selected_position,
                ground_position=selected_position.copy(),
                target_position=target_position,
                distance_to_target=distance,
            )
        )

    _check_rbm_constraint_rank(
        springs=springs,
        center=0.5 * (mins + maxs),
    )

    return SoftRigidBodySpringSet(
        stiffness=float(stiffness),
        springs=springs,
    )


def _nearest_unused_node(
    points: np.ndarray,
    target_position: np.ndarray,
    used_node_ids: set[int],
) -> tuple[int, np.ndarray, float]:
    distances = np.linalg.norm(points - target_position[None, :], axis=1)
    order = np.argsort(distances)

    for point_index in order:
        node_id = int(point_index) + 1

        if node_id not in used_node_ids:
            position = points[point_index].copy()
            distance = float(distances[point_index])

            return node_id, position, distance

    raise ValueError("Could not find an unused node for soft RBM spring.")


def _check_rbm_constraint_rank(
    springs: list[SoftRigidBodySpring],
    center: np.ndarray,
) -> None:
    """
    Check that the six scalar springs can control the six rigid-body modes.

    Rigid displacement field:

        u = T + omega x r

    Each scalar spring constrains one component of u weakly.
    """

    rows = []

    for spring in springs:
        r = spring.selected_position - center
        rx, ry, rz = r

        row = np.zeros(6, dtype=float)

        # Translation part: Tx, Ty, Tz.
        row[spring.dof - 1] = 1.0

        # Rotation part: omega = [wx, wy, wz].
        if spring.dof == 1:
            # ux = Tx + wy * rz - wz * ry
            row[4] = rz
            row[5] = -ry

        elif spring.dof == 2:
            # uy = Ty + wz * rx - wx * rz
            row[3] = -rz
            row[5] = rx

        elif spring.dof == 3:
            # uz = Tz + wx * ry - wy * rx
            row[3] = ry
            row[4] = -rx

        else:
            raise ValueError(f"Invalid DOF: {spring.dof}")

        rows.append(row)

    matrix = np.vstack(rows)
    rank = np.linalg.matrix_rank(matrix)

    if rank < 6:
        raise ValueError(
            "Selected soft spring directions do not control all six rigid-body "
            f"modes. Constraint rank is {rank}, expected 6."
        )


def print_soft_rbm_spring_summary(
    spring_set: SoftRigidBodySpringSet,
) -> None:
    print("Soft rigid-body spring stabilization:")
    print(f"  stiffness: {spring_set.stiffness}")

    for spring in spring_set.springs:
        print(
            f"  {spring.name}: "
            f"node={spring.selected_node_id}, "
            f"ground={spring.ground_node_id}, "
            f"element={spring.element_id}, "
            f"dof=U{spring.dof}, "
            f"distance_to_target={spring.distance_to_target}"
        )