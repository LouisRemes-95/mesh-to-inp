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
class MacroStressConfig:
    sxx: float
    syy: float
    szz: float
    sxy: float
    sxz: float
    syz: float


@dataclass(frozen=True)
class StepIncrementConfig:
    initial: float
    total: float
    minimum: float
    maximum: float
    max_number: int


@dataclass(frozen=True)
class TimeIncrementationControlsConfig:
    enabled: bool
    max_equilibrium_iterations: int | None
    cutback_after_equilibrium_iterations: int | None
    max_attempts_per_increment: int | None
    max_severe_discontinuity_iterations: int | None
    severe_discontinuity_iterations_for_increase: int | None
    cutback_factor_after_divergence: float | None
    cutback_factor_slow_convergence: float | None
    cutback_factor_too_many_iterations: float | None
    increase_factor_after_easy_increments: float | None
    max_increment_increase_factor: float | None


@dataclass(frozen=True)
class StepControlsConfig:
    time_incrementation: TimeIncrementationControlsConfig | None


@dataclass(frozen=True)
class StepConfig:
    name: str
    type: str
    nlgeom: bool
    increments: StepIncrementConfig
    controls: StepControlsConfig


@dataclass(frozen=True)
class CaseConfig:
    job: JobConfig
    mesh: MeshConfig
    materials: list[MaterialConfig]
    solid_section: SolidSectionConfig
    macro_stress: MacroStressConfig
    step: StepConfig


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
    macro_stress = _parse_macro_stress(raw.get("loading"))
    step = _parse_step(raw.get("step"))

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
        macro_stress=macro_stress,
        step=step,
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


def _parse_macro_stress(raw: Any) -> MacroStressConfig:
    if not isinstance(raw, dict):
        raise UserError("Missing or invalid 'loading' section.")

    stress_raw = raw.get("macro_stress")

    if not isinstance(stress_raw, dict):
        raise UserError("Missing or invalid 'loading.macro_stress' section.")

    def get_component(name: str) -> float:
        value = stress_raw.get(name, 0.0)

        if not isinstance(value, int | float):
            raise UserError(f"Invalid loading.macro_stress.{name}")

        return float(value)

    return MacroStressConfig(
        sxx=get_component("sxx"),
        syy=get_component("syy"),
        szz=get_component("szz"),
        sxy=get_component("sxy"),
        sxz=get_component("sxz"),
        syz=get_component("syz"),
    )


def _parse_step(raw: Any) -> StepConfig:
    if raw is None:
        raw = {}

    if not isinstance(raw, dict):
        raise UserError("'step' must be a dictionary.")

    step_type = raw.get("type", "quasi_static")

    if step_type != "quasi_static":
        raise UserError(
            f"Unsupported step.type '{step_type}'. "
            "Currently only 'quasi_static' is supported."
        )

    name = raw.get("name", "MacroStressLoading")
    nlgeom = raw.get("nlgeom", True)
    increments_raw = raw.get("increments", {})

    if not isinstance(name, str) or not name.strip():
        raise UserError("step.name must be a non-empty string.")

    if not isinstance(nlgeom, bool):
        raise UserError("step.nlgeom must be true or false.")

    if not isinstance(increments_raw, dict):
        raise UserError("step.increments must be a dictionary.")

    def get_float(name: str, default: float) -> float:
        value = increments_raw.get(name, default)

        if not isinstance(value, int | float):
            raise UserError(f"step.increments.{name} must be a number.")

        value = float(value)

        if value <= 0.0:
            raise UserError(f"step.increments.{name} must be > 0.")

        return value

    def get_int(name: str, default: int) -> int:
        value = increments_raw.get(name, default)

        if not isinstance(value, int):
            raise UserError(f"step.increments.{name} must be an integer.")

        if value <= 0:
            raise UserError(f"step.increments.{name} must be > 0.")

        return value

    increments = StepIncrementConfig(
        initial=get_float("initial", 0.01),
        total=get_float("total", 1.0),
        minimum=get_float("minimum", 1.0e-8),
        maximum=get_float("maximum", 0.1),
        max_number=get_int("max_number", 100000),
    )

    if increments.minimum > increments.initial:
        raise UserError("step.increments.minimum cannot be larger than initial.")

    if increments.initial > increments.maximum:
        raise UserError("step.increments.initial cannot be larger than maximum.")

    if increments.maximum > increments.total:
        raise UserError("step.increments.maximum cannot be larger than total.")

    controls = _parse_step_controls(raw.get("controls"))

    return StepConfig(
        name=name.strip(),
        type=step_type,
        nlgeom=nlgeom,
        increments=increments,
        controls=controls,
    )


def _parse_step_controls(raw: Any) -> StepControlsConfig:
    if raw is None:
        return StepControlsConfig(time_incrementation=None)

    if not isinstance(raw, dict):
        raise UserError("step.controls must be a dictionary.")

    time_raw = raw.get("time_incrementation")

    if time_raw is None:
        return StepControlsConfig(time_incrementation=None)

    if not isinstance(time_raw, dict):
        raise UserError("step.controls.time_incrementation must be a dictionary.")

    enabled = time_raw.get("enabled", True)

    if not isinstance(enabled, bool):
        raise UserError("step.controls.time_incrementation.enabled must be true or false.")

    def optional_int(field: str) -> int | None:
        value = time_raw.get(field)

        if value is None:
            return None

        if not isinstance(value, int):
            raise UserError(f"step.controls.time_incrementation.{field} must be an integer.")

        if value <= 0:
            raise UserError(f"step.controls.time_incrementation.{field} must be > 0.")

        return value

    def optional_float(field: str) -> float | None:
        value = time_raw.get(field)

        if value is None:
            return None

        if not isinstance(value, int | float):
            raise UserError(f"step.controls.time_incrementation.{field} must be a number.")

        value = float(value)

        if value <= 0.0:
            raise UserError(f"step.controls.time_incrementation.{field} must be > 0.")

        return value

    return StepControlsConfig(
        time_incrementation=TimeIncrementationControlsConfig(
            enabled=enabled,
            max_equilibrium_iterations=optional_int("max_equilibrium_iterations"),
            cutback_after_equilibrium_iterations=optional_int(
                "cutback_after_equilibrium_iterations"
            ),
            max_attempts_per_increment=optional_int("max_attempts_per_increment"),
            max_severe_discontinuity_iterations=optional_int(
                "max_severe_discontinuity_iterations"
            ),
            severe_discontinuity_iterations_for_increase=optional_int(
                "severe_discontinuity_iterations_for_increase"
            ),
            cutback_factor_after_divergence=optional_float(
                "cutback_factor_after_divergence"
            ),
            cutback_factor_slow_convergence=optional_float(
                "cutback_factor_slow_convergence"
            ),
            cutback_factor_too_many_iterations=optional_float(
                "cutback_factor_too_many_iterations"
            ),
            increase_factor_after_easy_increments=optional_float(
                "increase_factor_after_easy_increments"
            ),
            max_increment_increase_factor=optional_float(
                "max_increment_increase_factor"
            ),
        )
    )