from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mesh_to_inp.errors import UserError


@dataclass(frozen=True)
class JobConfig:
    name: str
    output: Path


@dataclass(frozen=True)
class MeshConfig:
    input: Path


@dataclass(frozen=True)
class ElasticConfig:
    E: float
    nu: float


@dataclass(frozen=True)
class PerfectPlasticConfig:
    yield_stress: float


@dataclass(frozen=True)
class MaterialConfig:
    name: str
    density: float
    elastic: ElasticConfig
    plastic: PerfectPlasticConfig


@dataclass(frozen=True)
class SolidSectionConfig:
    elset: str
    material: str


@dataclass(frozen=True)
class CaseConfig:
    job: JobConfig
    mesh: MeshConfig
    materials: list[MaterialConfig]
    solid_section: SolidSectionConfig


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
    job_raw = raw.get("job")
    mesh_raw = raw.get("mesh")

    if not isinstance(job_raw, dict):
        raise UserError("Missing or invalid 'job' section in case file.")

    if not isinstance(mesh_raw, dict):
        raise UserError("Missing or invalid 'mesh' section in case file.")

    job_name = job_raw.get("name")
    output_path = job_raw.get("output")
    input_path = mesh_raw.get("input")

    if not isinstance(job_name, str) or not job_name.strip():
        raise UserError("Missing or invalid required field: job.name")

    if output_path is None:
        raise UserError("Missing required field: job.output")

    if input_path is None:
        raise UserError("Missing required field: mesh.input")
    
    materials = _parse_materials(raw.get("materials"))
    solid_section = _parse_solid_section(raw.get("sections"), materials)

    return CaseConfig(
        job=JobConfig(
            name=job_name,
            output=(base_dir / output_path).resolve(),
        ),
        mesh=MeshConfig(
            input=(base_dir / input_path).resolve(),
        ),
        materials=materials,
        solid_section=solid_section,
    )


def _parse_materials(raw: Any) -> list[MaterialConfig]:
    if raw is None:
        return []

    if not isinstance(raw, list):
        raise UserError("'materials' must be a list.")

    materials: list[MaterialConfig] = []
    used_names: set[str] = set()

    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise UserError(f"Material #{i + 1} must be a dictionary.")

        name = item.get("name")
        density = item.get("density")
        elastic_raw = item.get("elastic")
        plastic_raw = item.get("plastic")

        if not isinstance(name, str) or not name.strip():
            raise UserError(f"Material #{i + 1} is missing a valid 'name'.")

        name = name.strip()

        if name in used_names:
            raise UserError(f"Duplicate material name: {name}")

        used_names.add(name)

        if not isinstance(density, int | float):
            raise UserError(f"Material '{name}' has invalid 'density'.")

        if density <= 0:
            raise UserError(f"Material '{name}' must have density > 0.")

        if not isinstance(elastic_raw, dict):
            raise UserError(f"Material '{name}' is missing an 'elastic' section.")

        E = elastic_raw.get("E")
        nu = elastic_raw.get("nu")

        if not isinstance(E, int | float):
            raise UserError(f"Material '{name}' has invalid elastic.E.")

        if not isinstance(nu, int | float):
            raise UserError(f"Material '{name}' has invalid elastic.nu.")

        if E <= 0:
            raise UserError(f"Material '{name}' must have elastic.E > 0.")

        if not (-1.0 < nu < 0.5):
            raise UserError(f"Material '{name}' must have -1 < elastic.nu < 0.5.")

        if not isinstance(plastic_raw, dict):
            raise UserError(f"Material '{name}' is missing a 'plastic' section.")

        yield_stress = plastic_raw.get("yield_stress")

        if not isinstance(yield_stress, int | float):
            raise UserError(f"Material '{name}' has invalid plastic.yield_stress.")

        if yield_stress <= 0:
            raise UserError(f"Material '{name}' must have plastic.yield_stress > 0.")

        materials.append(
            MaterialConfig(
                name=name,
                density=float(density),
                elastic=ElasticConfig(
                    E=float(E),
                    nu=float(nu),
                ),
                plastic=PerfectPlasticConfig(
                    yield_stress=float(yield_stress),
                ),
            )
        )

    return materials


def _parse_solid_section(
    raw: Any,
    materials: list[MaterialConfig],
) -> SolidSectionConfig:
    if not isinstance(raw, dict):
        raise UserError("Missing or invalid 'sections' section.")

    solid_raw = raw.get("solid")

    if not isinstance(solid_raw, dict):
        raise UserError("Missing or invalid 'sections.solid' section.")

    elset = solid_raw.get("elset")
    material = solid_raw.get("material")

    if not isinstance(elset, str) or not elset.strip():
        raise UserError("Missing or invalid field: sections.solid.elset")

    if not isinstance(material, str) or not material.strip():
        raise UserError("Missing or invalid field: sections.solid.material")

    material_names = {m.name for m in materials}

    if material not in material_names:
        raise UserError(
            f"Section references unknown material '{material}'. "
            f"Known materials: {sorted(material_names)}"
        )

    return SolidSectionConfig(
        elset=elset.strip(),
        material=material.strip(),
    )