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
    Select a local three-node rigid-body constraint triad.

    A:
        node closest to the minimum-coordinate corner.
        Fix U1, U2, U3.

    B:
        nearby node from A, preferably in the +X direction.
        Fix U2, U3.

    C:
        nearby node from A, preferably in the +Y direction.
        Fix U3.

    Returned node ids are Abaqus 1-based node labels.
    """

    node_a = _minimum_coordinate_node_id(points)
    point_a = points[node_a - 1]

    node_b = _nearest_directional_neighbor(
        points=points,
        origin=point_a,
        direction=np.array([1.0, 0.0, 0.0]),
        excluded={node_a},
    )

    node_c = _nearest_directional_neighbor(
        points=points,
        origin=point_a,
        direction=np.array([0.0, 1.0, 0.0]),
        excluded={node_a, node_b},
    )

    return [
        RigidBodyConstraint(node_id=node_a, first_dof=1, last_dof=3),
        RigidBodyConstraint(node_id=node_b, first_dof=2, last_dof=3),
        RigidBodyConstraint(node_id=node_c, first_dof=3, last_dof=3),
    ]


def _minimum_coordinate_node_id(points: np.ndarray) -> int:
    """
    Choose the node nearest the global minimum corner.

    This is robust for slightly non-planar boundaries because it does not
    require exact xmin/ymin/zmin equality.
    """

    mins = points.min(axis=0)
    distances_squared = np.sum((points - mins) ** 2, axis=1)

    return int(np.argmin(distances_squared)) + 1


def _nearest_directional_neighbor(
    points: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
    excluded: set[int],
    minimum_directional_fraction: float = 0.5,
) -> int:
    """
    Select a nearby node from origin, preferentially in a given direction.

    The score favours:
      - short distance from origin
      - positive projection along the requested direction
      - limited sideways drift

    This is more robust than asking for exact same y/z or x/z coordinates.
    """

    direction = direction / np.linalg.norm(direction)

    vectors = points - origin
    distances = np.linalg.norm(vectors, axis=1)
    projections = vectors @ direction

    valid = projections > 0.0

    for node_id in excluded:
        valid[node_id - 1] = False

    if not np.any(valid):
        return _closest_node_id_excluding(points, origin, excluded)

    positive_distances = distances[valid]
    nearest_distance = float(np.min(positive_distances))

    # First try nodes in the requested direction among the nearest local shell.
    # The factor gives tolerance for non-structured/non-straight boundaries.
    local_radius = 1.5 * nearest_distance

    local = valid & (distances <= local_radius)

    if np.any(local):
        # Prefer nodes with a strong directional component.
        directional_fraction = np.zeros_like(distances)
        nonzero = distances > 0.0
        directional_fraction[nonzero] = projections[nonzero] / distances[nonzero]

        directional = local & (directional_fraction >= minimum_directional_fraction)

        if np.any(directional):
            candidate_indices = np.where(directional)[0]
            best = candidate_indices[np.argmin(distances[candidate_indices])]
            return int(best) + 1

        candidate_indices = np.where(local)[0]
        # If no candidate is strongly aligned, maximize projection/distance.
        best = candidate_indices[
            np.argmax(projections[candidate_indices] / distances[candidate_indices])
        ]
        return int(best) + 1

    # Fallback: among all positive-projection nodes, choose the nearest.
    candidate_indices = np.where(valid)[0]
    best = candidate_indices[np.argmin(distances[candidate_indices])]

    return int(best) + 1


def _closest_node_id_excluding(
    points: np.ndarray,
    target: np.ndarray,
    excluded: set[int],
) -> int:
    distances_squared = np.sum((points - target) ** 2, axis=1)

    for node_id in excluded:
        distances_squared[node_id - 1] = np.inf

    return int(np.argmin(distances_squared)) + 1