from dataclasses import dataclass

import numpy as np


# C3D4 local face definitions.
# Local node ids are Python 0-based positions inside one tetra connectivity row.
C3D4_FACE_NODES = {
    "S1": (0, 1, 2),
    "S2": (0, 3, 1),
    "S3": (1, 3, 2),
    "S4": (2, 3, 0),
}


@dataclass(frozen=True)
class ElementFaceRef:
    element_id: int
    face_label: str


@dataclass(frozen=True)
class InterfaceSurfacePair:
    region_master: int
    region_slave: int
    component_id: int
    master_name: str
    slave_name: str
    interaction_name: str
    master_faces: list[ElementFaceRef]
    slave_faces: list[ElementFaceRef]


@dataclass(frozen=True)
class _InterfaceFace:
    face_nodes: tuple[int, int, int]
    master_ref: ElementFaceRef
    slave_ref: ElementFaceRef


def build_interface_surface_pairs(
    tetras: np.ndarray,
    tetra_regions: np.ndarray,
    original_to_output_element_id: np.ndarray,
) -> list[InterfaceSurfacePair]:
    """
    Build element-face master/slave surfaces for every connected region-region
    interface patch.

    Master/slave rule:
        lower region id = master
        higher region id = slave

    Important:
        Abaqus contact pair surfaces should be continuous. Therefore, one
        region-region interface is split into edge-connected components.
    """

    face_map: dict[tuple[int, int, int], list[tuple[int, int, str]]] = {}

    for tet_id, tet in enumerate(tetras):
        region = int(tetra_regions[tet_id])

        for face_label, local_face_nodes in C3D4_FACE_NODES.items():
            face_nodes = tuple(int(tet[i]) for i in local_face_nodes)
            face_key = tuple(sorted(face_nodes))

            face_map.setdefault(face_key, []).append(
                (tet_id, region, face_label)
            )

    faces_by_pair: dict[tuple[int, int], list[_InterfaceFace]] = {}

    for face_key, adjacent_faces in face_map.items():
        if len(adjacent_faces) != 2:
            continue

        tet_a, region_a, face_label_a = adjacent_faces[0]
        tet_b, region_b, face_label_b = adjacent_faces[1]

        if region_a == region_b:
            continue

        region_master = min(region_a, region_b)
        region_slave = max(region_a, region_b)
        pair_key = (region_master, region_slave)

        elem_a = int(original_to_output_element_id[tet_a])
        elem_b = int(original_to_output_element_id[tet_b])

        if elem_a <= 0 or elem_b <= 0:
            raise ValueError(
                "Invalid output element id while building interface surfaces."
            )

        ref_a = ElementFaceRef(
            element_id=elem_a,
            face_label=face_label_a,
        )
        ref_b = ElementFaceRef(
            element_id=elem_b,
            face_label=face_label_b,
        )

        if region_a == region_master:
            master_ref = ref_a
            slave_ref = ref_b
        else:
            master_ref = ref_b
            slave_ref = ref_a

        faces_by_pair.setdefault(pair_key, []).append(
            _InterfaceFace(
                face_nodes=tuple(int(n) for n in face_key),
                master_ref=master_ref,
                slave_ref=slave_ref,
            )
        )

    surface_pairs: list[InterfaceSurfacePair] = []

    for (region_master, region_slave), interface_faces in sorted(faces_by_pair.items()):
        components = _split_into_edge_connected_components(interface_faces)

        for component_id, component_faces in enumerate(components, start=1):
            suffix = f"R{region_master}_R{region_slave}_C{component_id}"

            master_name = f"IFACE_{suffix}_MASTER"
            slave_name = f"IFACE_{suffix}_SLAVE"
            interaction_name = f"INT_{suffix}"

            surface_pairs.append(
                InterfaceSurfacePair(
                    region_master=region_master,
                    region_slave=region_slave,
                    component_id=component_id,
                    master_name=master_name,
                    slave_name=slave_name,
                    interaction_name=interaction_name,
                    master_faces=[face.master_ref for face in component_faces],
                    slave_faces=[face.slave_ref for face in component_faces],
                )
            )

    return surface_pairs


def _split_into_edge_connected_components(
    faces: list[_InterfaceFace],
) -> list[list[_InterfaceFace]]:
    """
    Split triangular faces into components connected through full edges.

    We intentionally use shared edges, not shared nodes, because Abaqus complains
    about surfaces joined only at discrete nodes.
    """

    if not faces:
        return []

    edge_to_face_ids: dict[tuple[int, int], list[int]] = {}

    for face_id, face in enumerate(faces):
        for edge in _triangle_edges(face.face_nodes):
            edge_to_face_ids.setdefault(edge, []).append(face_id)

    neighbors: list[set[int]] = [set() for _ in faces]

    for face_ids in edge_to_face_ids.values():
        if len(face_ids) < 2:
            continue

        for i in face_ids:
            for j in face_ids:
                if i != j:
                    neighbors[i].add(j)

    visited: set[int] = set()
    components: list[list[_InterfaceFace]] = []

    for start in range(len(faces)):
        if start in visited:
            continue

        stack = [start]
        visited.add(start)
        component_ids: list[int] = []

        while stack:
            current = stack.pop()
            component_ids.append(current)

            for neighbor in neighbors[current]:
                if neighbor in visited:
                    continue

                visited.add(neighbor)
                stack.append(neighbor)

        components.append([faces[i] for i in component_ids])

    return components


def _triangle_edges(nodes: tuple[int, int, int]) -> list[tuple[int, int]]:
    n0, n1, n2 = nodes

    return [
        tuple(sorted((n0, n1))),
        tuple(sorted((n1, n2))),
        tuple(sorted((n2, n0))),
    ]