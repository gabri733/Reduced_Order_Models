# Data-driven ROM: plate with a hole (stress concentration, plasticity)

Non-intrusive surrogate (PODI) of an APDL finite element model: it maps **design parameters** (geometry and
material) to nodal fields, without re-running the simulation. The load is fixed (150 MPa gross stress, ramped in
10 load steps).

## Setup
Python 3.10 or later (tested with 3.12):
```
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```
- **A licensed ANSYS MAPDL installation** is only needed for the offline phase (`run_doe.py`, `export_mesh.py`).
  Tested with ANSYS 2025 R1. If it is not in the default location (`C:\Program Files\ANSYS Inc\v251\...`), point to the
  executable with the `MAPDL_EXEC` environment variable (e.g. `set MAPDL_EXEC=C:\...\ANSYS251.exe`); if it is not
  set, PyMAPDL looks for the installation on its own.
- **Without ANSYS**, `rom.py` and `plots.py` still work: the snapshots are already in `scripts/data/`.
  `ansys-mapdl-core` must still be installed because `rom.py` imports `fem.py`.
- The `.pkl` files in `scripts/results/` depend on the numpy/scipy versions: if they fail to load, regenerate them
  with `python rom.py`.

## FEM model (`plate_hole.apdl`)
Quarter of a 50 x 75 mm plate with an elliptical hole, PLANE183 in plane stress, bilinear isotropic hardening (BISO),
mapped mesh with fixed topology (1529 nodes, 480 elements). When `a_h` and `b_h` change the mesh is morphed but nodes
and elements stay the same, so snapshots are comparable node by node.

| parameter | meaning | range |
|---|---|---|
| `a_h` | hole semi-axis along X (perpendicular to the load) | 6 - 18 mm |
| `b_h` | hole semi-axis along Y (parallel to the load) | 6 - 18 mm |
| `sy` | yield stress | 250 - 450 MPa |
| `Htan` | hardening tangent modulus | 500 - 5000 MPa |

About 19% of the samples stay elastic (Kt·150 < sy), so the ROM also has to capture the onset of plasticity.

## Pipeline
The scripts are in `scripts/` and must be run from there:
```
cd scripts
python run_doe.py 150 0     # offline: LHS + 150 MAPDL runs (~4.5 s each) -> data/case_*.npz   (resumable)
python export_mesh.py       # connectivity for the plots -> data/mesh.npz
python rom.py               # POD + RBF and GPR, test-set errors, saves results/rom_*.pkl
python plots.py             # figures in results/
```
`sig_app` and `n_ls` are defined both in `plate_hole.apdl` and in `fem.py`: keep them aligned.

## ROM (`rom.py`)
For each field (nodal coordinates, displacements, σyy, von Mises stress, equivalent plastic strain, load curves at the
hole tip): SVD of the snapshots (mean removed), then regression of the modal coefficients with either
**RBF** (kernel and epsilon chosen by Rippa's leave-one-out) or **GPR** (anisotropic SE kernel).
Besides the 4 parameters, the regression input includes the estimated yield ratio `(1+2a/b)·σ_app/sy`, which makes
the elastic-plastic transition less abrupt.

## Results (120 training cases, 30 test cases)
Mean / max L2 error, normalized by the FEM field with the largest norm in the test set.

| field | modes | POD (projection) | RBF | GPR |
|---|---|---|---|---|
| coordinates (morphing) | 6 | 1.6e-6 / 3.6e-6 | 1.5e-5 / 6.9e-5 | 5.7e-6 / 1.6e-5 |
| displacements | 20 | 1.3e-4 / 3.5e-4 | 7.2e-4 / 3.0e-3 | 4.3e-4 / 2.2e-3 |
| σyy | 20 | 2.3e-3 / 4.1e-3 | 9.5e-3 / 2.1e-2 | 5.4e-3 / 1.2e-2 |
| von Mises stress | 20 | 3.6e-3 / 6.8e-3 | 9.9e-3 / 2.0e-2 | 5.9e-3 / 1.3e-2 |
| equivalent plastic strain | 20 | 3.6e-3 / 1.3e-2 | 1.8e-2 / 4.8e-2 | 1.7e-2 / 7.1e-2 |
| σyy curve at hole tip | 8 | 2.6e-9 / 1.2e-8 | 1.2e-2 / 4.1e-2 | 4.4e-3 / 2.3e-2 |
| max plastic strain curve | 7 | 5e-17 / 2e-16 | 1.1e-2 / 3.3e-2 | 3.7e-3 / 1.0e-2 |

GPR inference: ~6 ms per point versus ~4 s for the FEM run. The error is dominated by the regression, not by the POD
truncation (compare with the "POD (projection)" column); equivalent plastic strain is the hardest field because of the
yield threshold. GPR fitting takes ~6 minutes (about 100 independent GPs).

## Notes
- PyMAPDL's `mapdl.mesh.grid.cells_dict` is not reliable with quadratic elements: connectivity is taken from `mapdl.mesh.elem`.
- `tricontourf` hangs on invalid triangulations: the plots use `tripcolor`.
