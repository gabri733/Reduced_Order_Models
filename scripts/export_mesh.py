"""Salva la connettivita' (triangoli sui nodi d'angolo) della mesh di riferimento per i plot.
La topologia e' identica in tutti i casi, quindi basta una sola esecuzione.
"""
from pathlib import Path

import numpy as np

import fem

HERE = Path(__file__).parent


def main():
    mapdl = fem.start_mapdl(HERE / "data" / "_mesh_export")
    try:
        r = fem.run_case(mapdl, [12.0, 12.0, 350.0, 2000.0])
        # mapdl.mesh.grid.cells_dict NON e' affidabile con elementi quadratici (righe corrotte):
        # si usa mesh.elem, dove gli ultimi 8 valori sono i numeri di nodo del PLANE183 (primi 4 = angoli)
        pos = {int(n): i for i, n in enumerate(r["nnum"])}
        quads = np.array([[pos[int(n)] for n in e[-8:-4]] for e in mapdl.mesh.elem])
        tris = np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
        edge = np.linalg.norm(r["xy"][tris] - r["xy"][np.roll(tris, 1, axis=1)], axis=2)
        assert edge.max() < 30, f"connettivita' errata (lato max {edge.max():.1f} mm)"
        np.savez(HERE / "data" / "mesh.npz", tris=tris, nnum=r["nnum"])
        print("triangoli:", tris.shape, "nodi d'angolo usati:", len(np.unique(tris)), "lato max [mm]:", round(edge.max(), 1))
    finally:
        mapdl.exit(force=True)


if __name__ == "__main__":
    main()
