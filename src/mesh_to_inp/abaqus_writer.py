from pathlib import Path


def read_lines(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def rewrite_abaqus_lines(lines: list[str]) -> list[str]:
    header = []
    body = []
    in_header = False

    for stripped in lines:
        if stripped.startswith("*HEADING"):
            in_header = True
            header.append(stripped)
            continue

        if stripped.startswith("*") and in_header:
            in_header = False
            body.append("*PART, NAME=PART")

        if in_header:
            header.append(stripped)
            continue

        if stripped == "*ELEMENT, TYPE=C3D4":
            stripped = "*ELEMENT, TYPE=C3D4, ELSET=TETRA"

        body.append(stripped)

    if not header:
        return body

    return [
        header[0],
        " ".join(header[1:]),
        "Automatic python generated cohesive elements",
        *body,
    ]


def find_next_element_id(lines: list[str]) -> int:
    for line in reversed(lines):
        if line and not line.startswith("*"):
            return int(line.split(",")[0]) + 1

    raise ValueError("Could not find any element definition line.")