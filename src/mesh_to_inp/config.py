from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mesh_to_inp.errors import UserError


@dataclass(frozen=True)
class MeshConfig:
    input: Path
    output: Path


@dataclass(frozen=True)
class CaseConfig:
    mesh: MeshConfig


def load_case(path: Path) -> CaseConfig:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

    except FileNotFoundError:
        raise UserError(f"Case file does not exist: {path}")

    except PermissionError:
        raise UserError(f"Cannot read case file: {path}")

    except yaml.YAMLError as e:
        raise UserError(f"Invalid YAML file: {e}")

    if not isinstance(raw, dict):
        raise UserError("Case file must contain a YAML dictionary.")

    return _parse_case(raw, base_dir=path.parent)


def _parse_case(raw: dict[str, Any], base_dir: Path) -> CaseConfig:
    mesh_raw = raw.get("mesh")

    if not isinstance(mesh_raw, dict):
        raise UserError("Missing or invalid 'mesh' section in case file.")

    input_path = mesh_raw.get("input")
    output_path = mesh_raw.get("output")

    if input_path is None:
        raise UserError("Missing required field: mesh.input")

    if output_path is None:
        raise UserError("Missing required field: mesh.output")

    return CaseConfig(
        mesh=MeshConfig(
            input=(base_dir / input_path).resolve(),
            output=(base_dir / output_path).resolve(),
        )
    )