"""Planar 2-joint arm with 6 muscle-tendon actuators (geometric moment arms).
Mirrors the DynSyn 'Motivation' setup: drive joints, record actuator lengths,
ask which actuators move together. Pure numpy, a few MB of RAM."""
import numpy as np

L1 = 0.30
# (origin frame, origin pt, insertion frame, insertion pt); frame 0=ground,1=link1,2=link2
MUSCLES = [
    (0, (-0.02,  0.06), 1, (0.12,  0.03)),   # 0 shoulder flexor     (mono)
    (0, (-0.02, -0.06), 1, (0.12, -0.03)),   # 1 shoulder extensor   (mono)
    (1, ( 0.15,  0.03), 2, (0.06,  0.03)),   # 2 elbow flexor        (mono)
    (1, ( 0.15, -0.03), 2, (0.06, -0.03)),   # 3 elbow extensor      (mono)
    (0, (-0.02,  0.06), 2, (0.10,  0.03)),   # 4 biarticular flexor
    (0, (-0.02, -0.06), 2, (0.10, -0.03)),   # 5 biarticular extensor
]

def _rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])

def _world(frame, pt, q1, q2):
    pt = np.asarray(pt)
    if frame == 0:
        return pt
    if frame == 1:
        return _rot(q1) @ pt
    return _rot(q1) @ np.array([L1, 0.0]) + _rot(q1 + q2) @ pt

def lengths(q1, q2):
    """muscle lengths l_i(q1,q2) for scalar angles"""
    out = np.empty(len(MUSCLES))
    for i, (fa, a, fb, b) in enumerate(MUSCLES):
        out[i] = np.linalg.norm(_world(fa, a, q1, q2) - _world(fb, b, q1, q2))
    return out

def simulate(T=3000, dt=0.05, w1=3.0, w2=3.0 * (np.sqrt(5) - 1) / 2 * 2, A1=0.6, A2=0.6,
             q2_off=1.0, noise=0.15, seed=0, two_freq=True):
    """Quasi-periodic joint motion. two_freq=False -> single frequency (ring).
    Returns clean lengths, noisy lengths, joint angles, true omegas."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) * dt
    q1 = A1 * np.sin(w1 * t)
    q2 = q2_off + A2 * np.sin((w2 if two_freq else w1) * t + 0.7)
    Lc = np.array([lengths(a, b) for a, b in zip(q1, q2)])
    sd = Lc.std(0, keepdims=True)
    Ln = Lc + noise * sd * rng.standard_normal(Lc.shape)
    return Lc, Ln, np.c_[q1, q2], (w1, w2 if two_freq else w1), dt
