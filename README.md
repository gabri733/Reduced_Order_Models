# ROM data-driven: piastra forata con concentrazione di tensione (plasticità)

Surrogato non intrusivo (PODI) di un modello FEM in APDL: dai **parametri di progetto** (geometria e materiale) ai
campi nodali, senza rifare la simulazione. Il carico è fisso (150 MPa a lordo, rampa in 10 passi).

## Modello FEM (`plate_hole.apdl`)
Quarto di piastra 50 x 75 mm, foro ellittico, PLANE183 in tensione piana, plasticità bilineare (BISO),
mesh mapped a topologia fissa (1529 nodi, 480 elementi): al variare di a, b la mesh si deforma (morphing) ma
nodi ed elementi restano gli stessi, quindi gli snapshot sono confrontabili nodo per nodo.

| parametro | significato | intervallo |
|---|---|---|
| `a_h` | semiasse foro in X (perpendicolare al carico) | 6 - 18 mm |
| `b_h` | semiasse foro in Y (parallelo al carico) | 6 - 18 mm |
| `sy` | tensione di snervamento | 250 - 450 MPa |
| `Htan` | modulo tangente di incrudimento | 500 - 5000 MPa |

Circa il 19% dei campioni resta elastico (Kt·150 < sy): il ROM deve riprodurre anche l'insorgenza della plasticità.

## Pipeline
Gli script sono in `scripts/` e vanno lanciati da lì:
```
cd scripts
python run_doe.py 150 0     # offline: LHS + 150 run MAPDL (~4.5 s l'uno) -> data/case_*.npz   (riprendibile)
python export_mesh.py       # connettivita' per i plot -> data/mesh.npz
python rom.py               # POD + RBF e GPR, errori sul test set, salva results/rom_*.pkl
python plots.py             # grafici in results/
```
Servono `ansys-mapdl-core`, numpy, scipy, matplotlib. Percorso di ANSYS in `fem.py` (`EXEC_FILE`).
`sig_app` e `n_ls` sono duplicati in `plate_hole.apdl` e `fem.py`: vanno tenuti allineati.

## ROM (`rom.py`)
Per ogni campo (coordinate nodali, spostamenti, σyy, σvm, ε plastica eq., curve di carico all'apice del foro):
SVD degli snapshot (media sottratta) → regressione dei coefficienti modali con
**RBF** (kernel ed epsilon scelti con leave-one-out di Rippa) oppure **GPR** (kernel SE anisotropo).
Come ingresso, oltre ai 4 parametri, c'è il rapporto di snervamento stimato `(1+2a/b)·σ_app/sy`:
rende meno brusco il passaggio elastico-plastico.

## Risultati (120 casi di training, 30 di test)
Errore L2 medio / massimo, normalizzato con il campo FEM di norma massima del test set.

| campo | modi | POD (proiezione) | RBF | GPR |
|---|---|---|---|---|
| coordinate (morphing) | 6 | 1.6e-6 / 3.6e-6 | 1.5e-5 / 6.9e-5 | 5.7e-6 / 1.6e-5 |
| spostamenti | 20 | 1.3e-4 / 3.5e-4 | 7.2e-4 / 3.0e-3 | 4.3e-4 / 2.2e-3 |
| σyy | 20 | 2.3e-3 / 4.1e-3 | 9.5e-3 / 2.1e-2 | 5.4e-3 / 1.2e-2 |
| σvm | 20 | 3.6e-3 / 6.8e-3 | 9.9e-3 / 2.0e-2 | 5.9e-3 / 1.3e-2 |
| ε plastica eq. | 20 | 3.6e-3 / 1.3e-2 | 1.8e-2 / 4.8e-2 | 1.7e-2 / 7.1e-2 |
| curva σyy apice | 8 | 2.6e-9 / 1.2e-8 | 1.2e-2 / 4.1e-2 | 4.4e-3 / 2.3e-2 |
| curva ε plastica max | 7 | 5e-17 / 2e-16 | 1.1e-2 / 3.3e-2 | 3.7e-3 / 1.0e-2 |

Inferenza GPR: ~6 ms per punto contro ~4 s del FEM. Il collo di bottiglia è la regressione, non la POD
(si vede dalla colonna "POD (proiezione)"); l'ε plastica è il campo più difficile per via della soglia di snervamento.
Fit GPR: ~6 minuti (circa 140 GP indipendenti).

## Note
- `mapdl.mesh.grid.cells_dict` di PyMAPDL non è affidabile con elementi quadratici: la connettività si prende da `mapdl.mesh.elem`.
- `tricontourf` si blocca con triangolazioni non valide: i plot usano `tripcolor`.
