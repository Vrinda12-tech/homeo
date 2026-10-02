import numpy as np, json
import arm
from koopman_tools import *
np.set_printoptions(precision=4, suppress=True, linewidth=140)
R = {}
Lc, Ln, q, (w1, w2), dt = arm.simulate()
print("true omegas (rad/s):", w1, round(w2, 4), "| per-step angles:", w1*dt, round(w2*dt, 4))

# ---- 1. DynSyn-style snapshot analysis: correlation of length changes
dL = np.diff(Ln, axis=0)
ev = np.sort(np.linalg.eigvalsh(np.corrcoef(dL.T)))[::-1]
print("\n[1] snapshot corr eigenvalues (DynSyn-style):", ev, " cum:", np.cumsum(ev)/ev.sum())
R['snapshot_corr_eigs'] = ev.tolist()

# ---- 2. Hankel / Takens rank
K = 60
H = hankel(Ln - Ln.mean(0), K)
S = np.linalg.svd(H, compute_uv=False)
Sn = S / S[0]
print("\n[2] Hankel size", H.shape, " normalized singular values:\n", Sn[:14])
Hc = hankel(Lc - Lc.mean(0), K)
Sc = np.linalg.svd(Hc, compute_uv=False); Sc /= Sc[0]
print("    clean-signal Hankel sing. values:\n", Sc[:14])
R['hankel_sv_noisy'] = Sn[:14].tolist(); R['hankel_sv_clean'] = Sc[:14].tolist()

# ---- 3. Hankel-DMD eigenvalues -> frequencies
for r in (4, 6, 8):
    Atil, Ur, _, lam, _ = hankel_dmd(Ln, K, r)
    fr = np.angle(lam) / dt
    order = np.argsort(-np.abs(fr))
    print(f"\n[3] r={r}: |lambda| = {np.sort(np.abs(lam))[::-1]}")
    print("    positive angular freqs (rad/s):", np.sort(fr[fr > 1e-6]))
    if r == 4: R['dmd_r4'] = {'abs': np.abs(lam).tolist(), 'omega': fr.tolist()}
print("    truth: w1 =", w1, " w2 =", round(w2, 4))

# ---- 4. Kalman filter on the Koopman state space (train 60% / test 40%)
n_tr = int(0.6 * len(Ln))
mu = Ln[:n_tr].mean(0)
Atil, Ur, Str, lam, _ = hankel_dmd(Ln[:n_tr], K, 4)
Htr = hankel(Ln[:n_tr] - mu, K); Xtr = Ur.T @ Htr
resid = Xtr[:, 1:] - Atil @ Xtr[:, :-1]
Q = np.cov(resid)
C = Ur[:6, :]                              # current-time rows of the POD basis
Rn = (0.15 * Ln[:n_tr].std(0))**2 * np.eye(6)  # known sensor noise (15% of signal sd)
Yte = Ln[n_tr:] - mu; Cte = Lc[n_tr:] - mu
Xh = kalman(Yte, Atil, C, Q, np.diag(np.diag(Rn)), np.zeros(4), np.eye(4) * Xtr.var())
rm = lambda a, b: np.sqrt(np.mean((a - b)**2))
raw, filt = rm(Yte, Cte), rm(Xh @ C.T, Cte)
print(f"\n[4] Kalman-Koopman test RMSE: raw noisy {raw:.5f} -> filtered {filt:.5f}  ({100*(1-filt/raw):.1f}% lower)")
R['kalman'] = {'raw_rmse': raw, 'filtered_rmse': filt}

# ---- 5. Topology of delay-embedded cloud
def cloud(L, r=4):
    Atil, Ur, _, lam, X = hankel_dmd(L, K, r); return X.T
res = {}
for name, (two, noise, shuffle) in {'ring (1 freq)': (False, 0.15, False), 'torus (2 freq)': (True, 0.15, False)}.items():
    _, L_, _, _, _ = arm.simulate(two_freq=two, noise=noise)
    Z = cloud(L_, 2 if not two else 4)
    life = persistence(Z)
    res[name] = life
    print(f"\n[5] {name}: top H1 lifespans {life[1][:4]}  top H2 {life[2][:3]}  Betti~{betti_estimate(life)}")
# noise cloud with SAME covariance as the torus cloud
Zt = cloud(Ln, 4); Cov = np.cov(Zt.T)
Zg = np.random.default_rng(1).multivariate_normal(np.zeros(4), Cov, size=len(Zt))
life = persistence(Zg)
print(f"    Gaussian, same covariance: top H1 {life[1][:4]}  top H2 {life[2][:3]}  Betti~{betti_estimate(life)}")
res['gauss same cov'] = life

# ---- 6. VICReg topology-blindness, exact toy
th = np.random.default_rng(2).uniform(0, 2*np.pi, 3000)
ring = np.c_[np.cos(th), np.sin(th)] * np.sqrt(1.0)          # cov = diag(1/2,1/2)
g = np.random.default_rng(3).standard_normal((3000, 2)) * np.sqrt(0.5)
print("\n[6] covariance of ring:\n", np.cov(ring.T), "\n    covariance of gaussian:\n", np.cov(g.T))
print("    VICReg (var, cov) ring:", vicreg_terms(ring*np.sqrt(2)), " gauss:", vicreg_terms(g*np.sqrt(2)))
lr, lg = persistence(ring, 1), persistence(g, 1)
print("    H1 top lifespan ring:", lr[1][:2], " gauss:", lg[1][:2])
R['vicreg_toy'] = {'vicreg_ring': vicreg_terms(ring*np.sqrt(2)), 'vicreg_gauss': vicreg_terms(g*np.sqrt(2)),
                   'H1_ring': lr[1][:2].tolist(), 'H1_gauss': lg[1][:2].tolist()}
json.dump(R, open('results_analysis.json', 'w'), indent=1, default=float)
np.save('persist_cache.npy', np.array([res], dtype=object), allow_pickle=True)
