"""ROM data-driven non intrusivo (PODI) per la piastra forata: mu -> campi nodali.

Offline : SVD (POD) degli snapshot di ogni campo + regressione mu -> coefficienti modali
Online  : coefficienti previsti (RBF oppure GPR) -> U_n c(mu) + media

Regressori (lezioni 2 e 3):
  * RBF con scelta di kernel ed epsilon tramite leave-one-out (formula di Rippa)
  * GPR con kernel SE anisotropo e iperparametri da massima marginal likelihood
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

import fem

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "results"
FIELDS = ["xy", "u", "syy", "seqv", "epl", "curve_syy", "curve_epl"]


# ----------------------------------------------------------------------------- dati
def load_cases(data_dir=DATA):
    cases = [np.load(f) for f in sorted(Path(data_dir).glob("case_*.npz"))]
    mu = np.array([c["mu"] for c in cases])
    snaps = {
        "xy": np.array([c["xy"].ravel(order="F") for c in cases]).T,          # (2N, M): [x; y]
        "u": np.array([np.r_[c["ux"], c["uy"]] for c in cases]).T,            # (2N, M)
        "syy": np.array([c["syy"] for c in cases]).T,
        "seqv": np.array([c["seqv"] for c in cases]).T,
        "epl": np.array([c["epl"] for c in cases]).T,
        "curve_syy": np.array([c["curve_syy"] for c in cases]).T,             # (n_ls, M)
        "curve_epl": np.array([c["curve_epl"] for c in cases]).T,
    }
    return mu, snaps


def normalize(mu):
    lo = np.array([p[1] for p in fem.PARAMS])
    hi = np.array([p[2] for p in fem.PARAMS])
    return (np.atleast_2d(mu) - lo) / (hi - lo)


def features(mu, yield_feature=True):
    """Parametri normalizzati + (opzionale) rapporto di snervamento stimato (1+2a/b)*sigma_app/sy.

    La plasticita' si innesca quando il rapporto supera 1: dare alla regressione questa quantita'
    analitica (costo nullo in fase di progetto) rende molto meno brusco il 'ginocchio' elastico-plastico.
    """
    mu = np.atleast_2d(mu)
    X = normalize(mu)
    if yield_feature:
        ratio = (1.0 + 2.0 * mu[:, 0] / mu[:, 1]) * fem.SIG_APP / mu[:, 2]
        X = np.c_[X, (ratio - 0.5) / 2.0]
    return X


# ----------------------------------------------------------------------------- POD
class POD:
    """SVD troncata della matrice degli snapshot (lezione 1), con sottrazione della media."""

    def __init__(self, tol=1e-8, n_max=20):
        self.tol, self.n_max = tol, n_max

    def fit(self, S):
        self.mean = S.mean(axis=1, keepdims=True)
        U, sig, Vt = np.linalg.svd(S - self.mean, full_matrices=False)
        energy = sig**2
        tail = 1.0 - np.cumsum(energy) / max(energy.sum(), 1e-300)
        idx = np.flatnonzero(tail < self.tol)
        self.n = int(min(self.n_max, idx[0] + 1 if idx.size else len(sig)))
        self.U, self.sig = U[:, : self.n], sig
        self.C = (np.diag(sig[: self.n]) @ Vt[: self.n]).T                    # (M, n) coefficienti
        return self

    def project(self, S):
        return (self.U.T @ (S - self.mean)).T

    def reconstruct(self, C):
        return self.mean + self.U @ np.atleast_2d(C).T


# ----------------------------------------------------------------------------- RBF
KERNELS = {
    "gaussian": lambda r, e: np.exp(-((e * r) ** 2)),
    "multiquadric": lambda r, e: np.sqrt(1.0 + (e * r) ** 2),
    "inv_multiquadric": lambda r, e: 1.0 / np.sqrt(1.0 + (e * r) ** 2),
}


def _dist(A, B):
    return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))


class RBF:
    """f(x) = sum_j w_j phi(|x - x_j|). Kernel ed epsilon scelti minimizzando l'errore leave-one-out."""

    def fit(self, X, Y):
        R, ridge, best = _dist(X, X), 1e-10, (np.inf, None, None)
        for name, phi in KERNELS.items():
            for eps in np.logspace(-1, 1.7, 28):
                try:
                    Ainv = np.linalg.inv(phi(R, eps) + ridge * np.eye(len(X)))
                except np.linalg.LinAlgError:
                    continue
                loo = (Ainv @ Y) / np.diag(Ainv)[:, None]                    # formula di Rippa
                err = np.sum(loo**2)
                if err < best[0]:
                    best = (err, name, eps)
        _, self.kernel, self.eps = best
        phi = KERNELS[self.kernel]
        self.X = X
        self.W = np.linalg.solve(phi(R, self.eps) + ridge * np.eye(len(X)), Y)
        self.loo_rms = np.sqrt(best[0] / Y.size)
        return self

    def predict(self, Xq):
        return KERNELS[self.kernel](_dist(Xq, self.X), self.eps) @ self.W


# ----------------------------------------------------------------------------- GPR
class GPR:
    """Un GP per ogni colonna di Y, kernel SE con lunghezze d'onda per parametro (ARD)."""

    @staticmethod
    def _K(A, B, ls, sf2):
        return sf2 * np.exp(-0.5 * (((A[:, None, :] - B[None, :, :]) / ls) ** 2).sum(-1))

    def _nll(self, th, X, y):
        d = X.shape[1]
        ls, sf2, sn2 = np.exp(th[:d]), np.exp(th[d]), np.exp(th[d + 1])
        K = self._K(X, X, ls, sf2) + (sn2 + 1e-9) * np.eye(len(X))
        try:
            L = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10
        alpha = cho_solve(L, y)
        return 0.5 * y @ alpha + np.log(np.diag(L[0])).sum()

    def fit(self, X, Y, restarts=3, seed=0):
        rng = np.random.default_rng(seed)
        d = X.shape[1]
        self.X, self.mu_y, self.sd_y = X, Y.mean(0), Y.std(0) + 1e-12
        Z = (Y - self.mu_y) / self.sd_y
        bounds = [(np.log(0.05), np.log(30))] * d + [(np.log(1e-2), np.log(1e2)), (np.log(1e-9), np.log(1e-1))]
        self.theta, self.alpha, self.chol = [], [], []
        for j in range(Y.shape[1]):
            best = None
            for r in range(restarts):
                th0 = np.r_[np.log(rng.uniform(0.3, 2.0, d)), 0.0, np.log(1e-4)]
                res = minimize(self._nll, th0, args=(X, Z[:, j]), method="L-BFGS-B", bounds=bounds)
                if best is None or res.fun < best.fun:
                    best = res
            th = best.x
            K = self._K(X, X, np.exp(th[:d]), np.exp(th[d])) + (np.exp(th[d + 1]) + 1e-9) * np.eye(len(X))
            L = cho_factor(K, lower=True)
            self.theta.append(th)
            self.chol.append(L)
            self.alpha.append(cho_solve(L, Z[:, j]))
        return self

    def predict(self, Xq, return_std=False):
        d = self.X.shape[1]
        mean, std = np.zeros((len(Xq), len(self.theta))), np.zeros((len(Xq), len(self.theta)))
        for j, th in enumerate(self.theta):
            ls, sf2 = np.exp(th[:d]), np.exp(th[d])
            Ks = self._K(Xq, self.X, ls, sf2)
            mean[:, j] = Ks @ self.alpha[j]
            if return_std:
                v = cho_solve(self.chol[j], Ks.T)
                std[:, j] = np.sqrt(np.maximum(sf2 - (Ks * v.T).sum(1), 0)) * self.sd_y[j]
        mean = mean * self.sd_y + self.mu_y
        return (mean, std) if return_std else mean


# ----------------------------------------------------------------------------- ROM completa
class ROM:
    def __init__(self, regressor="gpr", tol=1e-8, n_max=20, yield_feature=True):
        self.regressor, self.tol, self.n_max, self.yield_feature = regressor, tol, n_max, yield_feature

    def fit(self, mu, snaps):
        X = features(mu, self.yield_feature)
        self.pod, self.reg = {}, {}
        for f in FIELDS:
            self.pod[f] = POD(self.tol, self.n_max).fit(snaps[f])
            self.reg[f] = (GPR() if self.regressor == "gpr" else RBF()).fit(X, self.pod[f].C)
        return self

    def predict(self, mu):
        X = features(mu, self.yield_feature)
        return {f: self.pod[f].reconstruct(self.reg[f].predict(X)) for f in FIELDS}


def rel_errors(true, pred):
    """Errore L2 per campione, normalizzato con la norma del campo FEM piu' grande del set.

    La normalizzazione per campione non e' adatta ai campi che si annullano in parte dei casi
    (deformazione plastica nei casi elastici): il denominatore tenderebbe a zero.
    """
    return np.linalg.norm(true - pred, axis=0) / np.linalg.norm(true, axis=0).max()


# ----------------------------------------------------------------------------- main
def main(n_train=120):
    OUT.mkdir(exist_ok=True)
    mu, snaps = load_cases()
    M = len(mu)
    n_train = min(n_train, M - 1)
    tr, te = np.arange(n_train), np.arange(n_train, M)
    print(f"{M} casi: {len(tr)} train / {len(te)} test")
    S_tr = {f: snaps[f][:, tr] for f in FIELDS}
    S_te = {f: snaps[f][:, te] for f in FIELDS}

    models = {}
    for name in ("rbf", "gpr"):
        t = time.time()
        models[name] = ROM(name).fit(mu[tr], S_tr)
        print(f"fit {name}: {time.time() - t:.1f}s")
    pod_ref = ROM("rbf").fit(mu[tr], S_tr)  # solo per le basi POD

    preds = {name: models[name].predict(mu[te]) for name in models}
    print(f"\nErrore L2 sul test set (media / max), normalizzato con il campo FEM di norma massima")
    print(f"{'campo':10s} {'n':>3s} {'POD-proj':>16s} {'RBF':>16s} {'GPR':>16s}")
    for f in FIELDS:
        proj = pod_ref.pod[f].reconstruct(pod_ref.pod[f].project(S_te[f]))
        row = [rel_errors(S_te[f], proj)]
        for name in ("rbf", "gpr"):
            row.append(rel_errors(S_te[f], preds[name][f]))
        print(f"{f:10s} {pod_ref.pod[f].n:3d} " + " ".join(f"{e.mean():8.2e}/{e.max():7.1e}" for e in row))

    t = time.time()
    for _ in range(100):
        models["gpr"].predict(mu[te][:1])
    print(f"\ntempo di inferenza GPR: {(time.time() - t) * 10:.1f} ms per punto (FEM: ~4000 ms)")

    with open(OUT / "rom_gpr.pkl", "wb") as fh:
        pickle.dump(models["gpr"], fh)
    with open(OUT / "rom_rbf.pkl", "wb") as fh:
        pickle.dump(models["rbf"], fh)
    np.savez(OUT / "split.npz", train=tr, test=te)
    return mu, snaps, tr, te, models, pod_ref


if __name__ == "__main__":
    import rom  # le classi nel pickle devono risultare 'rom.X' e non '__main__.X'

    rom.main(*(int(a) for a in sys.argv[1:2]))
