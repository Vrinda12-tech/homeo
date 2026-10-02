"""Classical machinery: Hankel/Takens, SVD, exact DMD, Kalman filter, persistent homology."""
import numpy as np
from ripser import ripser

def hankel(X, K):
    """X: (T,m) -> H: (m*K, T-K+1); column t = [x_t; x_{t+1}; ...; x_{t+K-1}] (Takens delay vector)."""
    T, m = X.shape
    return np.vstack([X[k:T - K + 1 + k].T for k in range(K)])

def hankel_dmd(X, K, r):
    """Hankel-DMD. Returns Atilde (r x r), U_r, S, eigenvalues, X-coordinates of delay vectors."""
    H = hankel(X - X.mean(0), K)
    H1, H2 = H[:, :-1], H[:, 1:]
    U, S, Vt = np.linalg.svd(H1, full_matrices=False)
    Ur, Sr, Vr = U[:, :r], S[:r], Vt[:r].T
    Atil = Ur.T @ H2 @ Vr @ np.diag(1 / Sr)          # least-squares Koopman matrix in POD coordinates
    lam = np.linalg.eigvals(Atil)
    return Atil, Ur, S, lam, Ur.T @ H

def kalman(Y, A, C, Q, R, x0, P0):
    """Standard Kalman filter. Y:(T,p). returns filtered states (T,r)."""
    r = A.shape[0]; x, P = x0.copy(), P0.copy(); out = []
    for y in Y:
        x, P = A @ x, A @ P @ A.T + Q
        S = C @ P @ C.T + R
        G = P @ C.T @ np.linalg.inv(S)
        x = x + G @ (y - C @ x)
        P = (np.eye(r) - G @ C) @ P
        out.append(x.copy())
    return np.array(out)

def persistence(Z, maxdim=2, n=450, seed=0):
    """Vietoris-Rips persistent homology of a point cloud (subsampled to n points)."""
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(Z), size=min(n, len(Z)), replace=False))
    Zs = Z[idx] / Z[idx].std(0).mean()
    dgms = ripser(Zs, maxdim=maxdim)['dgms']
    life = [np.sort(d[:, 1] - d[:, 0])[::-1] if len(d) else np.array([]) for d in dgms]
    life = [l[np.isfinite(l)] for l in life]
    return life

def betti_estimate(life, min_gap=2.0, floor=0.3):
    """Betti numbers (b1, b2) from Rips lifespans. Keep bars >= floor (units: mean std of the cloud),
    then count bars before the largest multiplicative gap (last bar is compared with the floor itself).
    No bar >= floor  ->  group declared empty. b0 is 1 by connectedness and not estimated."""
    out = [1]
    for l in life[1:]:
        l = l[l >= floor][:12]
        if len(l) == 0: out.append(0); continue
        l = np.append(l, floor)
        ratios = l[:-1] / l[1:]
        k = int(np.argmax(ratios))
        out.append(k + 1 if ratios[k] >= min_gap else 0)
    return out

def vicreg_terms(Z, gamma=1.0, eps=1e-4):
    """variance hinge + covariance penalty (the two regularizers of VICReg), Z:(N,d)."""
    Zc = Z - Z.mean(0)
    std = np.sqrt(Zc.var(0) + eps)
    v = np.mean(np.maximum(0, gamma - std))
    C = Zc.T @ Zc / (len(Z) - 1)
    c = (np.sum(C**2) - np.sum(np.diag(C)**2)) / Z.shape[1]
    return v, c
