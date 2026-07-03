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
class CohesiveStiffnessConfig:
    knn: float
    kss: float
    ktt: float


@dataclass(frozen=True)
class CohesiveDamageConfig:
    normal_strength: float
    shear_strength: float
    fracture_energy: float
    stabilization: float


@dataclass(frozen=True)
class CohesiveConfig:
    stiffness: CohesiveStiffnessConfig
    damage: CohesiveDamageConfig


@dataclass(frozen=True)
class MaterialConfig:
    name: str
    density: float | None = None
    elastic: ElasticConfig | None = None
    plastic: PerfectPlasticConfig | None = None
    cohesive: CohesiveConfig | None = None


@dataclass(frozen=True)
class SolidSectionConfig:
    elset: str
    material: str


@dataclass(frozen=True)
class CohesiveSectionConfig:
    elset: str
    material: str
    response: str


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
    values: list[int | float | None]


@dataclass(frozen=True)
class LineSearchControlsConfig:
    enabled: bool
    values: list[int | float | None]


@dataclass(frozen=True)
class StepControlsConfig:
    analysis: str | None
    time_incrementation: TimeIncrementationControlsConfig | None
    line_search: LineSearchControlsConfig | None


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
    solid_section: SolidSectionConfig | None
    cohesive_section: CohesiveSectionConfig | None
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

    sections_raw = raw.get("sections", {})

    if sections_raw is None:
        sections_raw = {}

    if not isinstance(sections_raw, dict):
        raise UserError("'sections' must be a dictionary.")

    solid_section = _parse_solid_section(
        sections_raw.get("solid"),
        materials,
    )
    cohesive_section = _parse_cohesive_section(
        sections_raw.get("cohesive"),
        materials,
    )

    macro_stress = _parse_macro_stress(raw.get("loading"))
    step = _parse_step(raw.get("step"))

    return CaseConfig(
        job=JobConfig(
            name=job_name.strip(),
            output=(base_dir / output_path).resolve(),
        ),
        mesh=MeshConfig(
            input=(base_dir / input_path).resolve(),
        ),
        materials=materials,
        solid_section=solid_section,
        cohesive_section=cohesive_section,
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

        if not isinstance(name, str) or not name.strip():
            raise UserError(f"Material #{i + 1} is missing a valid 'name'.")

        name = name.strip()

        if name in used_names:
            raise UserError(f"Duplicate material name: {name}")

        used_names.add(name)

        has_bulk_data = any(key in item for key in ("density", "elastic", "plastic"))
        has_cohesive_data = "cohesive" in item

        if has_bulk_data and has_cohesive_data:
            raise UserError(
                f"Material '{name}' mixes bulk and cohesive definitions. "
                "Use separate materials."
            )

        if has_cohesive_data:
            material = _parse_cohesive_material(name, item)
        else:
            material = _parse_bulk_material(name, item)

        materials.append(material)

    return materials


def _parse_bulk_material(name: str, item: dict[str, Any]) -> MaterialConfig:
    density = item.get("density")
    elastic_raw = item.get("elastic")
    plastic_raw = item.get("plastic")

    if not isinstance(density, int | float):
        raise UserError(f"Material '{name}' has invalid 'density'.")

    density = float(density)

    if density <= 0.0:
        raise UserError(f"Material '{name}' must have density > 0.")

    if not isinstance(elastic_raw, dict):
        raise UserError(f"Material '{name}' is missing an 'elastic' section.")

    E = elastic_raw.get("E")
    nu = elastic_raw.get("nu")

    if not isinstance(E, int | float):
        raise UserError(f"Material '{name}' has invalid elastic.E.")

    if not isinstance(nu, int | float):
        raise UserError(f"Material '{name}' has invalid elastic.nu.")

    E = float(E)
    nu = float(nu)

    if E <= 0.0:
        raise UserError(f"Material '{name}' must have elastic.E > 0.")

    if not (-1.0 < nu < 0.5):
        raise UserError(f"Material '{name}' must have -1 < elastic.nu < 0.5.")

    if not isinstance(plastic_raw, dict):
        raise UserError(f"Material '{name}' is missing a 'plastic' section.")

    yield_stress = plastic_raw.get("yield_stress")

    if not isinstance(yield_stress, int | float):
        raise UserError(f"Material '{name}' has invalid plastic.yield_stress.")

    yield_stress = float(yield_stress)

    if yield_stress <= 0.0:
        raise UserError(f"Material '{name}' must have plastic.yield_stress > 0.")

    return MaterialConfig(
        name=name,
        density=density,
        elastic=ElasticConfig(E=E, nu=nu),
        plastic=PerfectPlasticConfig(yield_stress=yield_stress),
        cohesive=None,
    )


def _parse_cohesive_material(name: str, item: dict[str, Any]) -> MaterialConfig:
    cohesive_raw = item.get("cohesive")

    if not isinstance(cohesive_raw, dict):
        raise UserError(f"Material '{name}' has invalid 'cohesive' section.")

    stiffness_raw = cohesive_raw.get("stiffness")
    damage_raw = cohesive_raw.get("damage")

    if not isinstance(stiffness_raw, dict):
        raise UserError(f"Material '{name}' is missing cohesive.stiffness.")

    if not isinstance(damage_raw, dict):
        raise UserError(f"Material '{name}' is missing cohesive.damage.")

    stiffness = CohesiveStiffnessConfig(
        knn=_positive_float(
            stiffness_raw,
            "knn",
            f"Material '{name}' cohesive.stiffness.knn",
        ),
        kss=_positive_float(
            stiffness_raw,
            "kss",
            f"Material '{name}' cohesive.stiffness.kss",
        ),
        ktt=_positive_float(
            stiffness_raw,
            "ktt",
            f"Material '{name}' cohesive.stiffness.ktt",
        ),
    )

    damage = CohesiveDamageConfig(
        normal_strength=_positive_float(
            damage_raw,
            "normal_strength",
            f"Material '{name}' cohesive.damage.normal_strength",
        ),
        shear_strength=_positive_float(
            damage_raw,
            "shear_strength",
            f"Material '{name}' cohesive.damage.shear_strength",
        ),
        fracture_energy=_positive_float(
            damage_raw,
            "fracture_energy",
            f"Material '{name}' cohesive.damage.fracture_energy",
        ),
        stabilization=_positive_float(
            damage_raw,
            "stabilization",
            f"Material '{name}' cohesive.damage.stabilization",
        ),
    )

    return MaterialConfig(
        name=name,
        density=None,
        elastic=None,
        plastic=None,
        cohesive=CohesiveConfig(
            stiffness=stiffness,
            damage=damage,
        ),
    )


def _parse_solid_section(
    raw: Any,
    materials: list[MaterialConfig],
) -> SolidSectionConfig | None:
    if raw is None:
        return None

    if not isinstance(raw, dict):
        raise UserError("'sections.solid' must be a dictionary.")

    elset = raw.get("elset")
    material = raw.get("material")

    if not isinstance(elset, str) or not elset.strip():
        raise UserError("'sections.solid.elset' must be a non-empty string.")

    if not isinstance(material, str) or not material.strip():
        raise UserError("'sections.solid.material' must be a non-empty string.")

    elset = elset.strip()
    material = material.strip()

    known_materials = {m.name: m for m in materials}

    if material not in known_materials:
        raise UserError(
            f"sections.solid.material refers to unknown material '{material}'. "
            f"Known materials: {sorted(known_materials)}"
        )

    if known_materials[material].cohesive is not None:
        raise UserError(
            f"sections.solid.material must refer to a bulk material, not '{material}'."
        )

    return SolidSectionConfig(
        elset=elset,
        material=material,
    )


def _parse_cohesive_section(
    raw: Any,
    materials: list[MaterialConfig],
) -> CohesiveSectionConfig | None:
    if raw is None:
        return None

    if not isinstance(raw, dict):
        raise UserError("'sections.cohesive' must be a dictionary.")

    elset = raw.get("elset")
    material = raw.get("material")
    response = raw.get("response", "TRACTION SEPARATION")

    if not isinstance(elset, str) or not elset.strip():
        raise UserError("'sections.cohesive.elset' must be a non-empty string.")

    if not isinstance(material, str) or not material.strip():
        raise UserError("'sections.cohesive.material' must be a non-empty string.")

    if not isinstance(response, str) or not response.strip():
        raise UserError("'sections.cohesive.response' must be a non-empty string.")

    elset = elset.strip()
    material = material.strip()
    response = response.strip()

    known_materials = {m.name: m for m in materials}

    if material not in known_materials:
        raise UserError(
            f"sections.cohesive.material refers to unknown material '{material}'. "
            f"Known materials: {sorted(known_materials)}"
        )

    if known_materials[material].cohesive is None:
        raise UserError(
            f"sections.cohesive.material must refer to a cohesive material, not '{material}'."
        )

    return CohesiveSectionConfig(
        elset=elset,
        material=material,
        response=response,
    )


def _parse_macro_stress(raw: Any) -> MacroStressConfig:
    if not isinstance(raw, dict):
        raise UserError("Missing or invalid 'loading' section.")

    stress_raw = raw.get("macro_stress")

    if not isinstance(stress_raw, dict):
        raise UserError("Missing or invalid 'loading.macro_stress' section.")

    return MacroStressConfig(
        sxx=_number(stress_raw, "sxx", default=0.0, label="loading.macro_stress.sxx"),
        syy=_number(stress_raw, "syy", default=0.0, label="loading.macro_stress.syy"),
        szz=_number(stress_raw, "szz", default=0.0, label="loading.macro_stress.szz"),
        sxy=_number(stress_raw, "sxy", default=0.0, label="loading.macro_stress.sxy"),
        sxz=_number(stress_raw, "sxz", default=0.0, label="loading.macro_stress.sxz"),
        syz=_number(stress_raw, "syz", default=0.0, label="loading.macro_stress.syz"),
    )


def _parse_step(raw: Any) -> StepConfig:
    if raw is None:
        raw = {}

    if not isinstance(raw, dict):
        raise UserError("'step' must be a dictionary.")

    step_type = raw.get("type", "quasi_static")

    supported_step_types = {"quasi_static", "dynamic_implicit"}

    if step_type not in supported_step_types:
        raise UserError(
            f"Unsupported step.type '{step_type}'. "
            f"Supported values are: {sorted(supported_step_types)}"
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

    increments = StepIncrementConfig(
        initial=_positive_number(
            increments_raw,
            "initial",
            default=0.01,
            label="step.increments.initial",
        ),
        total=_positive_number(
            increments_raw,
            "total",
            default=1.0,
            label="step.increments.total",
        ),
        minimum=_positive_number(
            increments_raw,
            "minimum",
            default=1.0e-8,
            label="step.increments.minimum",
        ),
        maximum=_positive_number(
            increments_raw,
            "maximum",
            default=0.1,
            label="step.increments.maximum",
        ),
        max_number=_positive_int(
            increments_raw,
            "max_number",
            default=100000,
            label="step.increments.max_number",
        ),
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
        return StepControlsConfig(
            analysis=None,
            time_incrementation=None,
            line_search=None,
        )

    if not isinstance(raw, dict):
        raise UserError("step.controls must be a dictionary.")

    analysis = raw.get("analysis")

    if analysis is not None:
        if not isinstance(analysis, str) or not analysis.strip():
            raise UserError("step.controls.analysis must be a non-empty string.")

        analysis = analysis.strip().lower()

        if analysis != "discontinuous":
            raise UserError(
                "step.controls.analysis currently only supports 'discontinuous'."
            )

    time_incrementation = _parse_raw_control_values(
        raw=raw.get("time_incrementation"),
        label="step.controls.time_incrementation",
        expected_length=11,
        config_type=TimeIncrementationControlsConfig,
    )

    line_search = _parse_raw_control_values(
        raw=raw.get("line_search"),
        label="step.controls.line_search",
        expected_length=5,
        config_type=LineSearchControlsConfig,
    )

    return StepControlsConfig(
        analysis=analysis,
        time_incrementation=time_incrementation,
        line_search=line_search,
    )


def _parse_raw_control_values(
    raw: Any,
    label: str,
    expected_length: int,
    config_type,
):
    if raw is None:
        return None

    if not isinstance(raw, dict):
        raise UserError(f"{label} must be a dictionary.")

    enabled = raw.get("enabled", True)

    if not isinstance(enabled, bool):
        raise UserError(f"{label}.enabled must be true or false.")

    values = raw.get("values")

    if values is None:
        return config_type(enabled=enabled, values=[])

    if not isinstance(values, list):
        raise UserError(f"{label}.values must be a list.")

    if len(values) > expected_length:
        raise UserError(
            f"{label}.values must contain at most {expected_length} entries."
        )

    parsed_values: list[int | float | None] = []

    for i, value in enumerate(values):
        if value is None:
            parsed_values.append(None)
            continue

        if not isinstance(value, int | float):
            raise UserError(f"{label}.values[{i}] must be a number or null.")

        parsed_values.append(value)

    while len(parsed_values) < expected_length:
        parsed_values.append(None)

    return config_type(
        enabled=enabled,
        values=parsed_values,
    )


def _number(
    raw: dict[str, Any],
    field: str,
    default: float,
    label: str,
) -> float:
    value = raw.get(field, default)

    if not isinstance(value, int | float):
        raise UserError(f"{label} must be a number.")

    return float(value)


def _positive_number(
    raw: dict[str, Any],
    field: str,
    default: float,
    label: str,
) -> float:
    value = _number(raw, field, default, label)

    if value <= 0.0:
        raise UserError(f"{label} must be > 0.")

    return value


def _positive_float(raw: dict[str, Any], field: str, label: str) -> float:
    value = raw.get(field)

    if not isinstance(value, int | float):
        raise UserError(f"{label} must be a number.")

    value = float(value)

    if value <= 0.0:
        raise UserError(f"{label} must be > 0.")

    return value


def _positive_int(
    raw: dict[str, Any],
    field: str,
    default: int,
    label: str,
) -> int:
    value = raw.get(field, default)

    if not isinstance(value, int):
        raise UserError(f"{label} must be an integer.")

    if value <= 0:
        raise UserError(f"{label} must be > 0.")

    return value