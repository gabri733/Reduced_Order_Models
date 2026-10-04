"""Wrapper PyMAPDL: lancia plate_hole.apdl per un vettore di parametri mu ed estrae gli snapshot."""
import os
from pathlib import Path

import numpy as np
from ansys.mapdl.core import launch_mapdl

HERE = Path(__file__).parent
APDL_FILE = HERE / "plate_hole.apdl"

# Eseguibile di MAPDL: variabile d'ambiente MAPDL_EXEC, altrimenti il percorso di default sotto se esiste,
# altrimenti None (PyMAPDL cerca da solo l'installazione di ANSYS).
_DEFAULT_EXEC = r"C:\Program Files\ANSYS Inc\v251\ansys\bin\winx64\ANSYS251.exe"
EXEC_FILE = os.environ.get("MAPDL_EXEC") or (_DEFAULT_EXEC if Path(_DEFAULT_EXEC).exists() else None)

# nome, estremo inferiore, estremo superiore (spazio dei parametri di progetto)
PARAMS = [
    ("a_h",  6.0,   18.0),    # semiasse foro in X [mm]
    ("b_h",  6.0,   18.0),    # semiasse foro in Y [mm]
    ("sy",   250.0, 450.0),   # snervamento [MPa]
    ("Htan", 500.0, 5000.0),  # modulo tangente [MPa]
]
SIG_APP = 150.0  # deve coincidere con sig_app nel file APDL
N_LS = 10        # idem per n_ls


def start_mapdl(run_dir, **kwargs):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return launch_mapdl(
        exec_file=EXEC_FILE, run_location=str(run_dir), override=True,
        loglevel="ERROR", additional_switches="-smp", nproc=2, **kwargs,
    )


def run_case(mapdl, mu):
    """Risolve un caso e restituisce un dict con campi nodali finali e curve di carico."""
    a_h, b_h, sy, htan = (float(v) for v in mu)
    mapdl.clear()
    mapdl.parameters["a_h"] = a_h
    mapdl.parameters["b_h"] = b_h
    mapdl.parameters["sy"] = sy
    mapdl.parameters["Htan"] = htan
    mapdl.input(str(APDL_FILE))

    mapdl.post1()
    mapdl.allsel()
    pp = mapdl.post_processing
    n_tip = int(mapdl.queries.node(a_h, 0, 0))  # nodo all'apice del foro (a,0)

    curve_syy = np.zeros(N_LS)  # sigma_yy all'apice del foro vs passo di carico
    curve_epl = np.zeros(N_LS)  # max deformazione plastica equivalente
    for i in range(1, N_LS + 1):
        mapdl.set(i, "LAST")
        syy = np.asarray(pp.nodal_component_stress("Y"))
        nnum = np.asarray(mapdl.mesh.nnum)
        curve_syy[i - 1] = syy[nnum == n_tip][0]
        curve_epl[i - 1] = np.max(pp.nodal_plastic_eqv_strain())

    return dict(
        mu=np.array([a_h, b_h, sy, htan]),
        nnum=np.asarray(mapdl.mesh.nnum),
        xy=np.asarray(mapdl.mesh.nodes)[:, :2],
        ux=np.asarray(pp.nodal_displacement("X")),
        uy=np.asarray(pp.nodal_displacement("Y")),
        syy=syy,
        seqv=np.asarray(pp.nodal_eqv_stress()),
        epl=np.asarray(pp.nodal_plastic_eqv_strain()),
        curve_syy=curve_syy,
        curve_epl=curve_epl,
        n_elem=int(mapdl.mesh.n_elem),
        n_node=int(mapdl.mesh.n_node),
    )
