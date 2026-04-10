# mesh_to_inp_mesh

Convert volumetric meshes (e.g. `.mesh` or other `meshio`-supported formats) to Abaqus `.inp` files with automatically generated cohesive interface elements.

This tool reads a volumetric mesh using `meshio`, separates tetrahedral elements by **region IDs**, duplicates nodes per region, and inserts `COH3D6` cohesive elements at interfaces between neighboring regions.

The module is designed as a small reusable step in a larger research workflow for multi-material finite element simulations and interface modeling.

## Installation

This project uses **uv** for environment and dependency management.

### Option 1 — Clone the repository

```bash
git clone https://github.com/LouisRemes-95/mesh_to_inp_mesh.git
cd mesh_to_inp_mesh
uv sync
```

### Option 2 — Install directly from GitHub

You can also install the tool directly into a project environment:

```bash
uv add git+https://github.com/LouisRemes-95/mesh_to_inp_mesh.git
```

This makes the `mesh-to-inp-mesh` command available in the environment.

## Usage

Run the command line interface:

```bash
uv run mesh-to-inp-mesh input.mesh
```

Specify an output file:

```bash
uv run mesh-to-inp-mesh input.mesh -o output.inp
```

If no output path is provided, the `.inp` file is written next to the input mesh using the same name.

## Python Usage

The conversion function can also be used directly in Python:

```python
from pathlib import Path
from mesh_to_inp_mesh.convert import convert

convert(Path("input.mesh"), Path("output.inp"))
```

## Context

This module is intended to be used as one step in a larger processing pipeline:

```
CT scan → cleaning → smoothing → meshing → mesh_to_inp_mesh → Abaqus simulation
```

Each step of the workflow is implemented as a separate module to keep the pipeline modular and reproducible.