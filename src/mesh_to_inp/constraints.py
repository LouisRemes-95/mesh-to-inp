from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RigidBodyConstraint:
    node_id: int
    first_dof: int
    last_dof: int
    value: float = 0.0


def make_default_rigid_body_constraints(points: np.ndarray) -> list[RigidBodyConstraint]:
    """
    Select three robust corner nodes from the final Abaqus mesh.

    A: closest to (xmin, ymin, zmin), fix U1,U2,U3
    B: closest to (xmax, ymin, zmin), fix U2,U3
    C: closest to (xmin, ymax, zmin), fix U3

    Returned node ids are Abaqus 1-based node labels.
    """

    mins = points.min(axis=0)
    maxs = points.max(axis=0)

    corner_a = np.array([mins[0], mins[1], mins[2]])
    corner_b = np.array([maxs[0], mins[1], mins[2]])
    corner_c = np.array([mins[0], maxs[1], mins[2]])

    node_a = _closest_node_id(points, corner_a)
    node_b = _closest_node_id_excluding(points, corner_b, excluded={node_a})
    node_c = _closest_node_id_excluding(points, corner_c, excluded={node_a, node_b})

    return [
        RigidBodyConstraint(node_id=node_a, first_dof=1, last_dof=3),
        RigidBodyConstraint(node_id=node_b, first_dof=2, last_dof=3),
        RigidBodyConstraint(node_id=node_c, first_dof=3, last_dof=3),
    ]


def _closest_node_id(points: np.ndarray, target: np.ndarray) -> int:
    distances_squared = np.sum((points - target) ** 2, axis=1)
    return int(np.argmin(distances_squared)) + 1


def _closest_node_id_excluding(
    points: np.ndarray,
    target: np.ndarray,
    excluded: set[int],
) -> int:
    distances_squared = np.sum((points - target) ** 2, axis=1)

    for node_id in excluded:
        distances_squared[node_id - 1] = np.inf

    return int(np.argmin(distances_squared)) + 1