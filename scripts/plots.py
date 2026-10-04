"""Grafici di validazione della ROM (richiede aver eseguito run_doe.py, export_mesh.py e rom.py)."""
import pickle

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

import fem
import rom

OUT = rom.OUT


def main():
    OUT.mkdir(exist_ok=True)
    mu, snaps = rom.load_cases()
    sp = np.load(OUT / "split.npz")
    tr, te = sp["train"], sp["test"]
    gpr = pickle.load(open(OUT / "rom_gpr.pkl", "rb"))
    rbf = pickle.load(open(OUT / "rom_rbf.pkl", "rb"))
    pg, pr = gpr.predict(mu[te]), rbf.predict(mu[te])

    # 1) decadimento dei valori singolari
    fig, ax = plt.subplots(figsize=(6, 4))
    for f in ("xy", "u", "syy", "seqv", "epl"):
        s = gpr.pod[f].sig
        ax.semilogy(np.arange(1, len(s) + 1), s / s[0], "o-", ms=3, label=f)
    ax.set_xlim(0, 30)
    ax.set(xlabel="modo", ylabel=r"$\sigma_i/\sigma_1$", title="Decadimento dei valori singolari (POD)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "svd_decay.png", dpi=150)

    # 2) curve di carico all'apice del foro: FOM contro ROM (6 casi di test con maggiore plasticita')
    pick = te[np.argsort(-snaps["curve_epl"][-1, te])[:6]]
    local = {g: i for i, g in enumerate(te)}
    load = np.arange(1, fem.N_LS + 1) / fem.N_LS * fem.SIG_APP
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for g in pick:
        l = axs[0].plot(load, snaps["curve_syy"][:, g], "-", lw=1.5)[0]
        axs[0].plot(load, pg["curve_syy"][:, local[g]], "--", color=l.get_color())
        axs[1].plot(load, snaps["curve_epl"][:, g], "-", lw=1.5, color=l.get_color())
        axs[1].plot(load, pg["curve_epl"][:, local[g]], "--", color=l.get_color())
    axs[0].set(xlabel=r"$\sigma_{app}$ [MPa]", ylabel=r"$\sigma_{yy}$ apice foro [MPa]", title="FEM (continuo) / ROM-GPR (tratteggio)")
    axs[1].set(xlabel=r"$\sigma_{app}$ [MPa]", ylabel="max def. plastica eq.", title="Insorgenza della plasticita'")
    fig.tight_layout()
    fig.savefig(OUT / "load_curves.png", dpi=150)

    # 3) parity plot dei valori di picco
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    for ax, key, lab in zip(axs, ("curve_syy", "curve_epl"), (r"$\sigma_{yy}$ apice [MPa]", "max def. plastica eq.")):
        t = snaps[key][-1, te]
        ax.plot(t, pr[key][-1], "s", ms=4, label="RBF", alpha=0.7)
        ax.plot(t, pg[key][-1], "o", ms=4, label="GPR", alpha=0.7)
        lim = [min(t.min(), 0), t.max() * 1.05]
        ax.plot(lim, lim, "k-", lw=0.8)
        ax.set(xlabel="FEM", ylabel="ROM", title=lab)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "parity.png", dpi=150)

    # 4) confronto di campo sul caso di test peggiore (GPR) se la mesh e' disponibile
    mesh_file = rom.DATA / "mesh.npz"
    if mesh_file.exists():
        tris = np.load(mesh_file)["tris"]
        for key, title in (("syy", r"$\sigma_{yy}$ [MPa]"), ("epl", "def. plastica eq.")):
            err = rom.rel_errors(snaps[key][:, te], pg[key])
            k = int(np.argmax(err))
            g = te[k]
            n = snaps["syy"].shape[0]
            xy = snaps["xy"][:, g].reshape(2, n).T
            fig, axs = plt.subplots(1, 3, figsize=(13, 4.5), sharex=True, sharey=True)
            vals = (snaps[key][:, g], pg[key][:, k], pg[key][:, k] - snaps[key][:, g])
            vmax = vals[0].max()
            emax = np.abs(vals[2]).max()
            tri = mtri.Triangulation(xy[:, 0], xy[:, 1], tris)
            # tripcolor invece di tricontourf: quest'ultimo si blocca su questa triangolazione
            for ax, v, t in zip(axs, vals, ("FEM", "ROM-GPR", "errore")):
                if t != "errore":
                    c = ax.tripcolor(tri, v, shading="gouraud", cmap="viridis", vmin=0, vmax=vmax)
                else:
                    c = ax.tripcolor(tri, v, shading="gouraud", cmap="coolwarm", vmin=-emax, vmax=emax)
                fig.colorbar(c, ax=ax)
                ax.set(title=f"{t} - {title}", aspect="equal", xlim=(0, 30), ylim=(0, 30))
            fig.suptitle(f"caso test {k}, mu = {np.round(mu[g], 1)}  (errore relativo {err[k]:.1e})")
            fig.tight_layout()
            fig.savefig(OUT / f"field_{key}.png", dpi=150)
    print("grafici salvati in", OUT)


if __name__ == "__main__":
    main()
