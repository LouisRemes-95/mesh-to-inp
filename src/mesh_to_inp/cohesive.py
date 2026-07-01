import numpy as np
import meshio


def extract_interface_triangles(mesh: meshio.Mesh, key: str) -> np.ndarray:
    tris = mesh.cells_dict["tetra"][:, [[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]]].reshape(-1, 3)
    regions = np.repeat(mesh.cell_data_dict[key]["tetra"], 4)[:, None]
    sorted_tris_region = np.hstack([np.sort(tris, axis=1), regions])

    order_by_region = np.argsort(sorted_tris_region[:, -1])
    tris = tris[order_by_region, :]
    sorted_tris_region = sorted_tris_region[order_by_region, :]

    _, inverse, counts = np.unique(
        sorted_tris_region,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )

    is_boundary = counts[inverse] == 1
    tris = tris[is_boundary, :]
    sorted_tris_region = sorted_tris_region[is_boundary, :]

    _, index, inverse = np.unique(
        sorted_tris_region[:, :3],
        axis=0,
        return_index=True,
        return_inverse=True,
    )

    tris_regions = np.hstack(
        [
            tris,
            sorted_tris_region[:, 3:],
            sorted_tris_region[index[inverse], 3:],
        ]
    )

    keep = np.ones(len(tris_regions), dtype=bool)
    keep[index] = False

    return tris_regions[keep]


def make_cohesive_element_lines(
    tris_regions: np.ndarray,
    region_lut: dict[int, np.ndarray],
    start_elem_id: int,
) -> list[str]:
    lines = [
        "",
        "** =============================================================================",
        "** COHESIVE ELEMENTS",
        "** =============================================================================",
        "*ELEMENT, TYPE=COH3D6, ELSET=COHESIVE",
    ]

    for i, cohe_elem in enumerate(tris_regions):
        elem_id = start_elem_id + i

        side_1_nodes = region_lut[int(cohe_elem[3])][cohe_elem[:3]].astype(np.int64) + 1
        side_2_nodes = region_lut[int(cohe_elem[4])][cohe_elem[:3]].astype(np.int64) + 1

        line_data = np.concatenate(([elem_id], side_1_nodes, side_2_nodes))
        lines.append(",".join(map(str, line_data)))

    return lines