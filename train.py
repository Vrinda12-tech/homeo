"""Linear JEPA-style encoder on noisy muscle lengths. Compare regularizers against collapse.
z_t = W x_t ; predictor z_{t+1} ~ P z_t ; pred loss alone has global minimum W=0 (collapse)."""
import numpy as np, jax, jax.numpy as jnp, json, sys
import arm
from koopman_tools import hankel_dmd, persistence, betti_estimate

KD = 8                       # delay depth inside the loss (Hankel block count)
def delay(Z, K=KD):
    T = Z.shape[0]; return jnp.concatenate([Z[k:T - K + 1 + k] for k in range(K)], axis=1)  # (T-K+1, K*k)

def hl_residual(Z, eps=1e-3):
    """Hankel-Lyapunov residual: || C1 Sigma^-1 C1^T - Sigma ||_F / ||Sigma||_F
    zero  <=>  the delay-covariance is invariant under the best linear (Koopman) shift  (stationary, |lambda|=1)."""
    H = delay(Z); H = H - H.mean(0)
    H0, H1 = H[:-1], H[1:]
    n = H0.shape[0]
    S = H0.T @ H0 / n; C1 = H1.T @ H0 / n
    Sr = S + eps * jnp.trace(S) / S.shape[0] * jnp.eye(S.shape[0])
    M = C1 @ jnp.linalg.solve(Sr, C1.T)
    return jnp.linalg.norm(M - S) / (jnp.linalg.norm(S) + 1e-8)

def loss_fn(params, X, mode, r_star=4):
    W, P = params
    Z = X @ W.T
    pred = jnp.mean((Z[1:] - Z[:-1] @ P.T) ** 2)
    Zc = Z - Z.mean(0); C = Zc.T @ Zc / (Z.shape[0] - 1); k = Z.shape[1]
    reg = 0.0
    if mode in ('vicreg',):
        std = jnp.sqrt(jnp.diag(C) + 1e-4)
        reg = jnp.mean(jnp.maximum(0, 1 - std)) + (jnp.sum(C**2) - jnp.sum(jnp.diag(C)**2)) / k
    if mode in ('koopman', 'koopman_dim'):
        reg = jnp.maximum(0, 1 - jnp.trace(C) / k) ** 2 + hl_residual(Z)
    if mode == 'koopman_dim':      # + dimension floor: participation ratio of latent covariance >= r* - 1/2
        PR = jnp.trace(C) ** 2 / (jnp.sum(C ** 2) + 1e-8)
        reg = reg + jnp.maximum(0, (r_star - 0.5) - PR) ** 2
    return pred + reg, (pred, reg)

def run(mode, X, k=6, steps=1500, lr=1e-2, seed=0, r_star=4):
    key = jax.random.PRNGKey(seed); k1, k2 = jax.random.split(key)
    params = [jax.random.normal(k1, (k, X.shape[1])) / np.sqrt(X.shape[1]), jnp.eye(k) + 0.0 * jax.random.normal(k2, (k, k))]
    m = [jnp.zeros_like(p) for p in params]; v = [jnp.zeros_like(p) for p in params]
    g_and_l = jax.jit(jax.value_and_grad(lambda p: loss_fn(p, X, mode, r_star), has_aux=True))
    hist = []
    for t in range(1, steps + 1):
        (L, (pr, rg)), g = g_and_l(params)
        gn = float(jnp.sqrt(sum(jnp.sum(a**2) for a in g)))
        hist.append((float(L), float(pr), float(rg), gn))
        for i in range(2):
            m[i] = 0.9 * m[i] + 0.1 * g[i]; v[i] = 0.999 * v[i] + 0.001 * g[i]**2
            params[i] = params[i] - lr * (m[i] / (1 - 0.9**t)) / (jnp.sqrt(v[i] / (1 - 0.999**t)) + 1e-8)
    return params, np.array(hist)

def evaluate(W, X, dt, w_true):
    Z = np.asarray(X @ np.asarray(W).T)
    ev = np.sort(np.linalg.eigvalsh(np.cov(Z.T)))[::-1]; p = np.clip(ev, 1e-12, None) / ev.sum()
    erank = float(np.exp(-(p * np.log(p)).sum()))
    A = np.linalg.lstsq(Z[:-1] - Z.mean(0), Z[1:] - Z.mean(0), rcond=None)[0]
    r2 = 1 - np.sum((Z[1:] - Z.mean(0) - (Z[:-1] - Z.mean(0)) @ A)**2) / np.sum((Z[1:] - Z.mean(0))**2)
    energy = float(ev.sum())
    out = dict(trace=energy, erank=erank, r2_next=float(r2), eigs=ev.tolist())
    if energy > 1e-6:
        _, _, _, lam, Xd = hankel_dmd(Z, 20, 4)
        fr = np.abs(np.angle(lam)) / dt; fr = np.sort(fr[fr > 1e-3])
        out['dmd_abs'] = np.sort(np.abs(lam))[::-1].tolist(); out['dmd_omega'] = fr.tolist()
        life = persistence(Xd.T, maxdim=2, n=350)   # Takens readout: same for every method
        out['betti'] = betti_estimate(life)
        out['H1_top'] = life[1][:3].tolist(); out['H2_top'] = life[2][:2].tolist()
    return out

if __name__ == '__main__':
    Lc, Ln, q, w, dt = arm.simulate()
    X = jnp.asarray((Ln - Ln.mean(0)) / Ln.std(0), dtype=jnp.float32)
    from koopman_tools import hankel
    Sx = np.linalg.svd(hankel(np.asarray(X), 60), compute_uv=False); Sx /= Sx[0]
    r_star = int(np.argmax(Sx[:10] / Sx[1:11])) + 1          # data-driven: largest gap in Hankel spectrum
    print('Hankel spectrum gap -> r* =', r_star, flush=True)
    allres = {'r_star': r_star}
    allres['input_reference'] = evaluate(np.eye(6), np.asarray(X), dt, w)
    print('input reference:', {k: allres['input_reference'].get(k) for k in ('erank','betti','dmd_omega')}, flush=True)
    for mode in ['none', 'vicreg', 'koopman', 'koopman_dim']:
        allres[mode] = []
        for seed in range(3):
            (W, P), hist = run(mode, X, seed=seed, r_star=r_star)
            e = evaluate(W, np.asarray(X), dt, w); e['hist_pred'] = hist[::25, 1].tolist(); e['hist_gn'] = hist[::25, 3].tolist()
            e['hist_loss'] = hist[::25, 0].tolist(); allres[mode].append(e)
            print(f"{mode:8s} seed{seed}: tr={e['trace']:.4f} erank={e['erank']:.2f} R2next={e['r2_next']:.3f} "
                  f"betti={e.get('betti')} |lam|={np.round(e.get('dmd_abs',[]),3)} w={np.round(e.get('dmd_omega',[]),3)}", flush=True)
    json.dump(allres, open('results_train.json', 'w'), indent=1)
