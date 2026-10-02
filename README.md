# homeo

**Let the dynamics of the data tell you its dimension and its shape, then use that to stop a network's internal representation from collapsing.**

Small enough for an 8 GB laptop (everything here uses a few MB). Pure `numpy` + `scipy` + `jax` (CPU) + `ripser`.

> One synthetic system, a *linear* encoder, 3 seeds. This repo shows a mechanism with real numbers. It does **not** show that real JEPA / ESN / sequence-model training behaves the same. Section 10 lists every known flaw, including one error in my first version of this README that I corrected here.

---

## 0. Summary in 8 lines

1. Data: a 2-joint arm with 6 muscles. Two joints move at two frequencies whose ratio is irrational, so the 6 muscle lengths trace a **2-torus** (a donut surface).
2. DynSyn's first step (correlating how muscle lengths *change* step to step) cannot see this: differencing drops the signal-to-noise power ratio from 44 to about 0.5–1 (Section 5.2).
3. A Hankel (delay) matrix of the signal has a sharp singular-value gap at **rank 4** (= 2 frequencies × cos/sin). Dynamic mode decomposition (DMD) on it returns the frequencies to 5 digits and all eigenvalues at modulus 1.
4. Persistent homology (a way to count holes) gives Betti numbers **(1, 2, 1)** for the torus, (1, 1, 0) for a ring, (1, 0, 0) for noise with the same covariance.
5. VICReg's variance and covariance terms cannot tell a ring from Gaussian noise. That part is a short exact proof (Section 6.2).
6. Training a small JEPA-style model with only a prediction loss collapses it to zero. VICReg avoids that but forces noise dimensions to full size and keeps the torus in 1 of 3 seeds.
7. A Koopman-style regularizer (built from the Hankel-Lyapunov residual) plus an energy floor fails by *dimension collapse* (keeps 1 oscillator). Adding a dimension floor fixes it: torus (1,2,1) in 3 of 3 seeds, frequencies 3.00 and 3.737, all |λ| = 1.
8. Not proven: "gradients converge easily", behaviour with nonlinear encoders, stop-gradient/EMA targets, real muscle data, other dynamics.

---

## 1. Jargon, mapped to maths

| ML word | What it is as maths |
|---|---|
| **latent vector / hidden state** $z_t$ | the list of numbers a model holds internally at time $t$. Here $z_t = W x_t$, a matrix times the input. |
| **encoder** | the map $x \mapsto z$. Here a $6\times 6$ matrix $W$. |
| **predictor** | a map guessing $z_{t+1}$ from $z_t$. Here a matrix $P$ with $z_{t+1}\approx P z_t$. |
| **JEPA** | Joint-Embedding Predictive Architecture: train encoder + predictor so the predicted $z_{t+1}$ matches the real $z_{t+1}$, with the error measured in latent space, not on raw data. |
| **collapse** | the encoder outputs (nearly) the same thing for everything, e.g. $W\to 0$. Prediction error is then trivially zero. |
| **rank collapse / dimension collapse** | $z$ varies in fewer directions than it could: covariance of $z$ has few non-zero eigenvalues. |
| **VICReg / Barlow Twins** | add penalties on the covariance matrix of $z$ (each coordinate must have variance, coordinates must be uncorrelated) so $z$ cannot collapse. |
| **ESN (echo state network)** | a fixed random recurrent system whose internal state is the "memory"; only a linear readout is trained. |
| **RL** | reinforcement learning: learn a controller by trial and reward. DynSyn uses it; this repo does **not**. |
| **gradient converges easily** | the vector of derivatives of the loss shrinks smoothly toward 0 without wild oscillation. |

---

## 2. Research question and testable claims

**Question.** Models that predict the future in latent space (JEPA, sequence models, ESNs) are known to collapse. Is part of the problem that standard anti-collapse penalties look only at *second-order statistics* (covariance) at *one instant*, while the data's structure is *dynamical* (waves, frequencies, loops, tori)? And if so, does a penalty built from the dynamics itself prevent collapse while keeping the shape?

| # | Claim | How tested |
|---|---|---|
| H1 | Step-to-step correlation (DynSyn-style) cannot recover the dimension of a noisy quasi-periodic system; a Hankel/SVD analysis can. | Sections 5.2, 5.4 |
| H2 | Hankel-DMD recovers the frequencies and shows neutral stability ($\lvert\lambda\rvert = 1$). | 5.7 |
| H3 | VICReg's variance/covariance terms are blind to topology. | exact proof 6.2 + simulation |
| H4 | A dynamics-aware regularizer prevents collapse **and** keeps the topology. | Section 8 |
| H5 | Such a regularizer gives easier gradient convergence. | gradient-norm curve only (weak) |

---

## 3. Where DynSyn fits

DynSyn (He, Zuo, Ma, Sui, ICML 2024, arXiv 2407.11472) controls bodies with many muscles. Muscles whose length changes are correlated are put in one group ("synergy"). A controller outputs one command per group, plus state-dependent corrections per muscle. Its first step is a **snapshot correlation matrix** of length changes.

I keep that idea (`arm.py`: 2 joints, 6 muscle-tendons, true geometric lengths) and test whether *dynamical* statistics (Hankel, DMD, Kalman, persistent homology) recover more than the snapshot statistic. The DynSyn RL training (MyoSuite / MuJoCo) is **not** reproduced.

---

## 4. The system

**Geometry.** Muscle $i$ goes from point $a_i$ (attached to frame $f_a$) to point $b_i$ (frame $f_b$). With joint angles $q=(q_1,q_2)$, link length $L_1 = 0.3$ and rotation $R(\alpha)$, a point on link 1 is $R(q_1)p$, a point on link 2 is $R(q_1)(L_1,0)^\top + R(q_1+q_2)p$. Length:

$$\ell_i(q) = \lVert p^{\text{world}}_{b_i}(q) - p^{\text{world}}_{a_i}(q)\rVert .$$

Six muscles: 2 at the shoulder, 2 at the elbow, 2 crossing both joints.

**Motion.** $q_1(t) = 0.6\sin(\omega_1 t)$, $q_2(t) = 1.0 + 0.6\sin(\omega_2 t + 0.7)$, with $\omega_1 = 3$ rad/s and $\omega_2 = 3(\sqrt5-1) = 3.7082$ rad/s. $\omega_2/\omega_1 = \sqrt5-1$ is irrational. Sampling step $\Delta t = 0.05$ s, $T = 3000$ samples.

**Observation noise.** $y_t = \ell(q_t) + 0.15\, \mathrm{sd}_i\,\varepsilon_t$, $\varepsilon_t\sim\mathcal N(0,I)$, where $\mathrm{sd}_i$ is the standard deviation of muscle $i$'s clean length. Measured RMS noise = 0.00416; predicted $0.15\sqrt{\overline{\mathrm{sd}^2}} = 0.0042$. ✔

---

## 5. Maths part A: reading the shape from data

### 5.1 The system lives on a torus

Define phases $\theta_1 = \omega_1 t$, $\theta_2=\omega_2 t + 0.7$. The map $t\mapsto(\theta_1,\theta_2) \bmod 2\pi$ walks on the torus $\mathbb T^2 = S^1\times S^1$. Because $\omega_1/\omega_2$ is irrational, the orbit is dense and equidistributed (Kronecker/Weyl). The joint angles and the muscle lengths are smooth functions of $(\theta_1,\theta_2)$:

$$\ell(t) = f(\theta_1(t), \theta_2(t)),\qquad f:\mathbb T^2\to\mathbb R^6 .$$

If the Jacobian of $f$ has rank 2 and $f$ is one-to-one, the image of $f$ is a torus sitting inside $\mathbb R^6$. That is the shape.

### 5.2 Why step-to-step correlation fails here (H1)

Muscle lengths change as $\Delta\ell \approx J(q)\,\dot q\,\Delta t$, where $J$ is the $6\times2$ matrix of derivatives $\partial\ell/\partial q$. Two problems:

1. **Differencing amplifies noise.** For a sinusoid with angular frequency $\omega$, $\mathrm{rms}(\Delta s) = 2\sin(\omega\Delta t/2)\cdot \mathrm{rms}(s)\approx \omega\Delta t\cdot\mathrm{rms}(s)$. With $\omega\Delta t = 0.15$ (joint 1) that is about 15% of the signal's rms. White noise of rms $0.15\,\mathrm{sd}$ becomes $\sqrt2\cdot 0.15\,\mathrm{sd} = 21\%$ after differencing. Measured per muscle:

   | quantity | muscles 1–6 |
   |---|---|
   | rms(Δ clean) / sd | 0.151, 0.151, 0.186, 0.225, 0.177, 0.175 |
   | rms(Δ noise) / sd | 0.212, 0.210, 0.211, 0.211, 0.206, 0.212 |
   | power SNR after differencing | 0.51, 0.52, 0.78, 1.14, 0.73, 0.69 |
   | power SNR before ($1/0.15^2$) | 44.4 |

   So differencing drops the signal-to-noise power ratio about 40–80× (44 → 0.5–1.1).

2. **$J$ depends on $q$**, so the covariance of $J(q)\dot q$ is not exactly low-rank even without noise.

Result: eigenvalues of the correlation matrix of $\Delta\ell$ are `1.97, 1.56, 0.69, 0.67, 0.56, 0.55`: no gap, cumulative share 33%, 59%, 70%, 81%, 91%, 100%.

### 5.3 What a function on a torus looks like

Any smooth $f$ on $\mathbb T^2$ has a Fourier series

$$f(\theta_1,\theta_2)=\sum_{m,n\in\mathbb Z} c_{mn}\,e^{i(m\theta_1+n\theta_2)} .$$

Along the orbit each term is a pure oscillation at frequency $m\omega_1+n\omega_2$. For small joint amplitudes the largest terms are $(m,n)=(\pm1,0),(0,\pm1)$; then come $(\pm2,0),(0,\pm2),(\pm1,\pm1)$, created by the nonlinear tendon geometry.

### 5.4 Hankel matrix, rank = number of exponentials (H1)

Delay vector with depth $K$ and the Hankel matrix:

$$h_t=\begin{bmatrix}x_t\\x_{t+1}\\\vdots\\x_{t+K-1}\end{bmatrix}\in\mathbb R^{mK},\qquad H=[h_1\ h_2\ \cdots\ h_N].$$

Here $m=6$, $K=60$, $N=2941$, so $H$ is $360\times2941$.

**Claim.** If $x_t=\sum_{j=1}^{p} c_j z_j^{\,t}$ with $c_j\in\mathbb C^m$ and $p$ distinct $z_j$, then $\operatorname{rank}H\le p$, with equality when $K\ge p$ and the $c_j$ are independent. A real sinusoid of frequency $\omega$ is the sum of two exponentials $z=e^{\pm i\omega\Delta t}$, so each frequency adds **2** to the rank.

**Proof sketch.** $h_t=\sum_j z_j^{\,t}\,(v_j\otimes c_j)$ with $v_j=(1,z_j,\dots,z_j^{K-1})^\top$ (a Vandermonde vector). Hence $H=\sum_j (v_j\otimes c_j)\,(z_j^{1},\dots,z_j^{N})$, a sum of $p$ rank-1 matrices. $\square$

Two frequencies give rank **4**. Measured normalized singular values (noisy data):

`1, 0.943, 0.768, 0.751 | 0.109, 0.108, 0.058, 0.055, 0.027, …`

The gap is $0.751/0.109 = 6.9\times$. So **$r^\*=4$**. Clean-signal values are almost identical (`0.110, 0.109` after the gap), so the second tier (0.109, 0.108) is not noise but real nonlinearity: harmonics from Section 5.3. DMD with $r=8$ confirms this: it returns 6.0005 and 7.407 rad/s, matching $2\omega_1=6.0$ and $2\omega_2=7.416$.

### 5.5 Takens: why delays reconstruct the shape

Takens (1981) and Sauer–Yorke–Casdagli (1991): for a smooth flow on a $d$-dimensional manifold and a generic smooth observation, the delay map is an embedding once the number of delay coordinates exceeds $2d$. Here $d=2$, so 5 delays suffice in principle; we use $6\times 60$ numbers, far more than needed. Therefore the cloud of delay vectors (after projecting onto the top singular directions) is a faithful copy of the torus, and homology computed on it is the torus's homology.

### 5.6 What the noise does to singular values

For an $M\times N$ matrix of iid noise with variance $\sigma^2$ and $c=M/N$ ($M=mK=360$, $N=2941$, so $c=0.122$), singular values fall in the Marchenko–Pastur band $\sigma\sqrt N(1\pm\sqrt c)$, not at one flat level. In this repo the 6 channels have different noise variances, so the real band is wider than the equal-variance formula: noise-only Hankel singular values span 0.05–0.43 (absolute), and the observed tail (0.35–0.42) sits inside that. The signal's top singular value is 16.4, so the noise ceiling is ≈2.6% of $\sigma_1$. Improvement (Section 11): whiten the channels first, then use the Gavish–Donoho optimal hard threshold instead of eyeballing the gap.

### 5.7 DMD and the Koopman operator (H2)

**Koopman operator.** For a map $x_{t+1}=F(x_t)$, $\mathcal K g = g\circ F$ acts on functions and is linear even if $F$ is not. On the torus rotation, $g_{mn}=e^{i(m\theta_1+n\theta_2)}$ satisfies

$$\mathcal K g_{mn}=e^{i(m\omega_1+n\omega_2)\Delta t}\,g_{mn},$$

so eigenvalues are the lattice $e^{i(m\omega_1+n\omega_2)\Delta t}$, all on the unit circle. Neutral stability ($\lvert\lambda\rvert=1$) is what "not collapsing, not exploding" means for a Koopman mode.

**Hankel-DMD.** Least squares for $H_2\approx A H_1$ ($H_1$ = columns $1..N-1$, $H_2$ = columns $2..N$) gives $A=H_2H_1^{+}$. With $H_1\approx U_r\Sigma_rV_r^\top$ we have $H_1^{+}=V_r\Sigma_r^{-1}U_r^\top$ and, projecting onto the $r$ strongest directions,

$$\tilde A=U_r^\top A\,U_r=U_r^\top H_2\,V_r\,\Sigma_r^{-1}\qquad(r\times r).$$

Eigenvalues $\lambda=e^{(\sigma+i\omega)\Delta t}$ give frequency $\omega=\arg\lambda/\Delta t$ and growth rate $\sigma=\ln\lvert\lambda\rvert/\Delta t$.

**Result** ($r=4$): $\lvert\lambda\rvert=1.000$ (all four), $\omega=2.9998$ and $3.7082$ rad/s. True: 3.0000 and 3.7082. Relative errors $7\times10^{-5}$ and $\sim10^{-5}$.

### 5.8 Kalman filter on the Koopman state

Model in the top-4 Hankel coordinates $x_t=U_r^\top h_t$:

$$x_{t+1}=\tilde A x_t+w_t,\ w_t\sim\mathcal N(0,Q);\qquad y_t=C x_t+v_t,\ v_t\sim\mathcal N(0,R),$$

with $C$ = first 6 rows of $U_r$ (the "now" slice), $Q$ = covariance of DMD residuals, $R=\mathrm{diag}((0.15\,\mathrm{sd}_i)^2)$.

$$\begin{aligned}
&\text{predict: } x^-=\tilde A x,\quad P^-=\tilde A P\tilde A^\top+Q\\
&\text{gain: } G=P^-C^\top(CP^-C^\top+R)^{-1}\\
&\text{update: } x=x^-+G(y-Cx^-),\quad P=(I-GC)P^-
\end{aligned}$$

The filter is the minimum-variance estimator for a linear-Gaussian system; the steady-state $P$ solves the discrete algebraic Riccati equation. Fit on the first 60% of the data, filtered on the last 40%:

| | RMSE |
|---|---|
| raw noisy | 0.00416 |
| orthogonal projection onto the 4-D signal subspace (back-of-envelope: $\times\sqrt{4/6}$) | ≈ 0.00340 |
| Kalman–Koopman | **0.00280** (−32.6% vs raw) |

The projection-only bound is about 18% worse than Kalman because the filter also uses the past through $\tilde A$.

### 5.9 Persistent homology (counting holes)

For a point cloud $X$ and scale $\varepsilon$, the Vietoris–Rips complex is the set of simplices whose points are pairwise within $\varepsilon$. Chains $C_k$ and boundary maps $\partial_k:C_k\to C_{k-1}$ give homology $H_k=\ker\partial_k/\operatorname{im}\partial_{k+1}$ and Betti numbers $\beta_k=\dim H_k$: $\beta_0$ = components, $\beta_1$ = loops, $\beta_2$ = voids. As $\varepsilon$ grows, each hole is born at $b$ and dies at $d$; its **lifespan** is $d-b$.

Known answers: circle $(\beta_0,\beta_1)=(1,1)$; 2-torus $(1,2,1)$; a structureless Gaussian cloud has no long bars. The stability theorem (Cohen-Steiner–Edelsbrunner–Harer 2007; Chazal–de Silva–Oudot 2014: bottleneck distance $\le 2\,d_{GH}$) says small perturbations of the cloud move bars only a little, so **long bars are real, short bars are noise**.

**Readout rule used here** (a heuristic, see Section 10): normalize the cloud by its mean per-coordinate std; keep bars with lifespan $\ge0.3$; count bars before the largest ratio-gap if that ratio is $\ge2$, else declare the group empty.

| cloud | top $H_1$ lifespans | top $H_2$ | Betti |
|---|---|---|---|
| ring (1 freq) | **2.234**, 0.015, 0.015 | 0.075 | (1,1,0) |
| torus (2 freq) | **1.985, 1.824**, 0.608 | **1.245**, 0.114 | (1,2,1) |
| Gaussian, same covariance as torus cloud | 0.446, 0.372, 0.358 | 0.290, 0.234 | (1,0,0) |

Ring: 150× jump from the first to the second bar. Torus $H_2$: 11× jump. Gaussian: flat (ratio ~1.2).

---

## 6. Maths part B: collapse

### 6.1 Scale collapse is a symmetry of the loss

Prediction loss $\mathcal L(W,P)=\mathbb E\lVert W x_{t+1}-P\,Wx_t\rVert^2$ is homogeneous of degree 2 in $W$: $\mathcal L(sW,P)=s^2\mathcal L(W,P)$. So $\partial_s\mathcal L=2s\mathcal L\ge0$ and gradient descent shrinks $s$ exponentially: $\dot s\approx-2\hat{\mathcal L}\,s$. The global minimizer is $W=0$ ($z\equiv0$), which predicts itself perfectly. Observed: `none` → latent energy $\mathrm{tr}\,\Sigma_z\approx0$.

### 6.2 VICReg's variance/covariance terms are blind to topology (H3)

With $\Sigma_z=\mathrm{cov}(z)$, VICReg's two regularizers are $\frac1k\sum_i\max(0,\gamma-\sqrt{\Sigma_{ii}+\epsilon})$ and $\frac1k\sum_{i\ne j}\Sigma_{ij}^2$. Both depend on $z$ only through $\Sigma_z$.

**Exact example.** For $\theta\sim U[0,2\pi)$: $\mathbb E\cos^2\theta=\mathbb E\sin^2\theta=\tfrac12$, $\mathbb E\cos\theta\sin\theta=0$, so

$$\mathrm{cov}(\cos\theta,\sin\theta)=\tfrac12 I_2=\mathrm{cov}\big(\mathcal N(0,\tfrac12I_2)\big).$$

A loop and a blob have the identical covariance, hence identical VICReg regularizer values. Any property that is not a function of the first two moments (like $\beta_1$) is invisible to them. Numbers (3000 points each, scaled by $\sqrt2$): ring var-term 0.0039, cov-term $3.6\times10^{-7}$; Gaussian 0.0, $1.7\times10^{-7}$. Same, up to sampling error. Top $H_1$ lifespan: ring **2.289**, Gaussian 0.235.

Scope: this concerns the variance and covariance *regularizers*. It says nothing about VICReg's invariance term between two augmented views, which this repo does not model.

### 6.3 Why "just fix the energy" gives dimension collapse

Suppose the latent is decomposed into modes $i$ with energy $v_i$ and next-step linear predictability $\rho_i^2\in[0,1]$ (fraction of mode $i$'s next-step energy that a linear map can predict). The prediction loss is about $\sum_i v_i(1-\rho_i^2)$. If only the total energy is fixed, $\sum_iv_i=E$, minimizing a linear function over a simplex puts **all** energy on $i^\star=\arg\max_i\rho_i^2$: a vertex solution, rank 1. This is what the `koopman` row shows (effective rank 1.00). Per-coordinate variance floors (VICReg) forbid this by forcing every $v_i\ge$ floor, but then noise coordinates also get full size.

### 6.4 The Hankel–Lyapunov residual (what it really measures)

Let $h_t$ be the delay-stack of the latent $z$ (depth $K_d=8$ in the loss), $\Sigma=\mathbb E[hh^\top]$, $C_1=\mathbb E[h_{t+1}h_t^\top]$. Best linear one-step map: $A=C_1\Sigma^{-1}$. If the process is stationary ($\mathrm{cov}(h_{t+1})=\Sigma$), the one-step prediction error $e=h_{t+1}-Ah_t$ has

$$\mathbb E[ee^\top]=\Sigma-AC_1^\top-C_1A^\top+A\Sigma A^\top=\Sigma-C_1\Sigma^{-1}C_1^\top .$$

Define

$$\boxed{R_{HL}(Z)=\frac{\lVert C_1\Sigma^{-1}C_1^\top-\Sigma\rVert_F}{\lVert\Sigma\rVert_F}=\frac{\lVert\mathrm{cov}(e)\rVert_F}{\lVert\Sigma\rVert_F}}$$

(code adds a ridge $\epsilon=10^{-3}\,\mathrm{tr}\Sigma/\dim$ inside the inverse). So $R_{HL}$ is the **scale-free relative energy of the one-step Koopman prediction error.** It is zero exactly when $\Sigma=A\Sigma A^\top$, the Stein (discrete Lyapunov) equation, i.e. the latent's second-moment is invariant under its own linear dynamics. Special cases:

* pure rotation (all $\lvert\lambda\rvert=1$, no noise): $R_{HL}=0$;
* decaying rotation of isotropic 2-D state, $K_d=1$: $A\Sigma A^\top=\rho^2\Sigma$ so $R_{HL}=1-\rho^2$;
* white noise, $K_d=1$: $C_1=0$ so $R_{HL}=1$;
* **white noise, delay depth $K_d$, shift 1:** consecutive delay vectors share $K_d-1$ blocks, so only the newest block is unpredictable: $R_{HL}=1/\sqrt{K_d}$ (= 0.354 for $K_d=8$).

**Correction.** My first README said "white noise has $C_1\approx0$ so $R_{HL}\approx1$". That is true only for $K_d=1$. With the delay stack used in `train.py` it is $1/\sqrt{K_d}$, a much weaker signal. Checked numerically on 20 000 white-noise samples: 0.353 vs theory 0.354.

Verified on signals (not yet in training), with $K_d=8$:

| signal | shift 1 (current loss) | shift $K_d=8$ (non-overlapping) |
|---|---|---|
| white noise | 0.353 | **0.999** |
| pure rotation | 0.0002 | 0.0004 |
| AR(1)-rotation, $\rho=0.9$, noise-driven | 0.029 | 0.508 |

Shifting by the full window makes the residual separate noise (≈1), damped dynamics (~0.5) and pure oscillation (≈0). This is the first improvement to try (Section 11).

### 6.5 The dimension floor

Participation ratio of the latent covariance with eigenvalues $\mu_1\ge\dots\ge\mu_k$:

$$\mathrm{PR}=\frac{(\sum\mu_i)^2}{\sum\mu_i^2}\in[1,k],$$

equal to $k$ for an isotropic cloud and 1 for rank 1; $r$ equal eigenvalues give $\mathrm{PR}=r$. The floor is $\max(0,\,r^\star-\tfrac12-\mathrm{PR})^2$, with $r^\star=4$ read from the largest gap in the input's Hankel spectrum (`argmax(S[:10]/S[1:11])+1` on the standardized inputs).

---

## 7. Training setup (exact, from `train.py`)

* Data: standardized noisy lengths $x_t\in\mathbb R^6$, 3000 consecutive samples, full batch.
* Encoder $z_t=Wx_t$, $W\in\mathbb R^{6\times6}$, entries $\mathcal N(0,1/6)$. Predictor $P\in\mathbb R^{6\times6}$, init $I$.
* Prediction loss: $\frac{1}{N}\sum\lVert z_{t+1}-Pz_t\rVert^2$ (no stop-gradient, no EMA).
* Optimizer: Adam, lr $10^{-2}$, $\beta=(0.9,0.999)$, 1500 steps, 3 seeds.
* Modes:
  * `none`: prediction loss only.
  * `vicreg`: + $\frac1k\sum_i\max(0,1-\sqrt{\Sigma_{ii}+10^{-4}})$ + $\frac1k\sum_{i\ne j}\Sigma_{ij}^2$.
  * `koopman`: + $\max(0,1-\mathrm{tr}\Sigma_z/k)^2$ + $R_{HL}(Z)$, $K_d=8$.
  * `koopman_dim`: `koopman` + $\max(0,\,r^\star-\tfrac12-\mathrm{PR})^2$.
* All regularizer weights are 1; none were tuned.
* Why no SVD/eigenvalue in the gradient: a ring has *pairs* of equal singular values, and SVD gradients contain $1/(\sigma_i^2-\sigma_j^2)$, which blows up.
* Readout for every method: Hankel-DMD with $K=20$, $r=4$ on the latent, then persistent homology (350 pts) of the 4 Hankel-SVD coordinates.

---

## 8. Results (from `results_train.log`)

Input reference (no training): effective rank 2.32, Betti (1,2,1), $\omega=3.0002,\ 3.7456$.

| method | seed | latent energy tr | eff. rank | next-step R² | Betti | DMD $\omega$ (rad/s) | $\lvert\lambda\rvert$ |
|---|---|---|---|---|---|---|---|
| none | 0 | 0.0000 | 0.00 | 0.957 | n/a | n/a | n/a |
| none | 1 | 0.0000 | 1.13 | 0.955 | (1,0,0)* | 3.11, 4.35* | .999, .994* |
| none | 2 | 0.0000 | 1.14 | 0.957 | n/a | n/a | n/a |
| vicreg | 0 | 3.78 | 4.87 | 0.670 | (1,1,0) | 1.47, 3.03 | .999, .962 |
| vicreg | 1 | 3.86 | 5.04 | 0.656 | (1,1,0) | 1.76, 3.00 | 1.0, .963 |
| vicreg | 2 | 3.52 | 4.61 | 0.702 | (1,2,1) | 3.00, 3.735 | 1.0, 1.0 |
| koopman | 0 | 5.888 | 1.00 | 0.963 | (1,1,0) | 3.002, 6.945 | 1.0, .908 |
| koopman | 1 | 5.888 | 1.00 | 0.963 | (1,1,0) | 3.002, 6.943 | 1.0, .908 |
| koopman | 2 | 5.888 | 1.00 | 0.963 | (1,1,0) | 3.002, 6.948 | 1.0, .908 |
| **koopman_dim** | 0 | 5.364 | 3.87 | 0.789 | **(1,2,1)** | 3.000, 3.737 | 1, 1, 1, 1 |
| **koopman_dim** | 1 | 5.365 | 3.92 | 0.788 | **(1,2,1)** | 3.000, 3.737 | 1, 1, 1, 1 |
| **koopman_dim** | 2 | 5.363 | 3.88 | 0.788 | **(1,2,1)** | 3.000, 3.737 | 1, 1, 1, 1 |

\* Latent energy is ≈0, so these are readouts on a numerically-zero cloud; ignore everything in `none` rows except energy and effective rank. The rows differ slightly between runs because the collapsed values are floating-point dust.

### How to read it

* **`none` collapses**, exactly as 6.1 predicts.
* **R² is not "better".** `koopman` has the highest R² (0.963) because it kept one clean oscillator and threw the rest away (6.3). Predictability is bought by deleting dimensions. Judge by effective rank + Betti + frequencies together.
* **`koopman`'s second mode is fake.** It has $\lvert\lambda\rvert=0.908<1$ at $\omega\approx6.94$, which matches none of $2\omega_1=6.0$, $\omega_1+\omega_2=6.708$, $2\omega_2=7.416$. A DMD with $r=4$ forced onto a latent with only about 2 real dimensions fits noise; the decay gives it away.
* **VICReg** keeps rank 4.6–5.0 (it also enlarges noise dimensions, since every coordinate must reach variance 1), loses predictability (R² ≈ 0.66–0.70), and the torus survives in 1 of 3 seeds.
* **`koopman_dim`** matches the input: same Betti, same frequencies (3.000, 3.737; the input itself reads 3.746 so the 0.8% offset from the true 3.708 is a DMD-readout bias, not the regularizer), $\lvert\lambda\rvert=1$, effective rank 3.9 against the true 4.
* Gradient norm (`summary.png`, panel 5): `koopman_dim` falls smoothly to about $2\times10^{-3}$; `vicreg` hovers near $10^{-1}$ with jitter. Suggestive, but the losses are different objectives, there was no step-size sweep, and 3 seeds is thin. H5 is **not** established.

---

## 9. Verdicts

| claim | verdict |
|---|---|
| H1 snapshot correlation fails, Hankel works | **Supported** on this system (SNR 44 → ~0.5–1; spectrum gap ×6.9) |
| H2 Hankel-DMD recovers frequencies, $\lvert\lambda\rvert=1$ | **Supported** (errors ~$10^{-5}$ relative) |
| H3 VICReg var/cov blind to topology | **Proven** for the covariance terms; training evidence is weak (1/3 vs 3/3) |
| H4 dynamics-aware regularizer prevents collapse and keeps topology | **Partly supported**: needs the dimension floor, which needs $r^\star$ from data; linear encoder, one system |
| H5 easier gradient convergence | **Not established** |

---

## 10. Known flaws (read this before citing anything)

1. One synthetic system; a linear encoder with no nonlinearity and no EMA/stop-gradient target (real JEPA has both). 3 seeds, no confidence intervals.
2. $R_{HL}$ is a **normalized prediction error** (6.4). Its topological meaning is indirect: low prediction error + stationarity ⇒ eigenvalues near the unit circle. It is not a differentiable topology measure. Shape is read out afterwards with persistent homology; it is not optimized.
3. With shift 1 the white-noise residual is only $1/\sqrt{K_d}$ (6.4), so the penalty on noise is weak. The first README overstated it as ≈1. The non-overlapping shift version fixes this on signals but is **not yet tested in training**.
4. `koopman` alone fails by dimension collapse (6.3, 8). The dimension floor works but takes $r^\star$ from the input's Hankel spectrum by hand.
5. The Betti counter is a heuristic; VICReg seed 2 also passed it.
6. DMD frequency for $\omega_2$ is biased by 0.8% in latents (3.737 vs 3.708).
7. VICReg's invariance term, Barlow Twins and a real JEPA's stop-gradient/EMA were not modelled. No MyoSuite data, no RL.
8. Regularizer weights were not tuned (all 1).

---

## 11. How to improve it (ordered by expected payoff)

1. **Non-overlapping shift in $R_{HL}$** (predict $h_{t+K_d}$ from $h_t$). On signals it moves white noise from 0.35 to 1.00 and separates damped from pure oscillation. Test in the training loop first.
2. **Remove the hand-fed $r^\star$.** Whiten channels with the noise covariance, then apply the Gavish–Donoho optimal hard threshold to the Hankel spectrum; or use ESPRIT / matrix-pencil on the Hankel matrix to count frequencies directly.
3. **Nonlinear encoder + EMA target.** A 2-layer tanh encoder, stop-gradient on the target, compare `none` / `vicreg` / `koopman_dim`. Without this, claims about JEPA are not earned.
4. **More noise levels and seeds** (noise 0.15 → 0.6; 20 seeds; report mean ± interval). Check where $r^\star$ detection breaks.
5. **Other dynamics.** Van der Pol (one limit cycle, a ring), a damped system (should be penalized), and Lorenz (chaotic: continuous Koopman spectrum, so "$\lvert\lambda\rvert=1$" is the wrong target). The last is the most likely place the regularizer misleads.
6. **Add Barlow Twins and a "VICReg + dimension floor" baseline**, so the gain from the dimension floor is separated from the gain from the Koopman term.
7. **A differentiable topology term** (e.g. persistence-based loss on a small subsample) as an optional extra, to optimize shape directly instead of reading it out.
8. **Real data:** MyoSuite muscle lengths from the environments DynSyn uses; this is the true test of the DynSyn connection.
9. **Convergence study for H5:** same optimizer, learning-rate sweep, steps-to-threshold on a common metric (e.g. $\lVert\nabla\rVert<10^{-3}$) instead of comparing different losses.

---

## 12. Files and reproduction

```
arm.py            planar arm, 6 muscle lengths, quasi-periodic simulation
koopman_tools.py  hankel(), hankel_dmd(), kalman(), persistence(), betti_estimate(), vicreg_terms()
analysis.py       Sections 5.2, 5.4, 5.7, 5.8, 5.9, 6.2  ->  results_analysis.json
train.py          Section 7-8 (jax, CPU)                  ->  results_train.json
fig.py            summary.png
```

```
pip install numpy scipy matplotlib ripser jax
python analysis.py | tee results_analysis.log
python train.py    | tee results_train.log
python fig.py
```

Memory: a few MB. Runtime: about a minute on a laptop CPU. Re-runs match to ~3 decimals; rows with numerically-zero latent energy are not reproducible.

---

## 13. References

* Takens, *Detecting strange attractors in turbulence*, 1981. Sauer, Yorke, Casdagli, *Embedology*, J. Stat. Phys. 1991.
* Koopman, 1931; Mezić, *Spectral properties of dynamical systems…*, 2005; Schmid, *Dynamic mode decomposition…*, JFM 2010; Tu et al., *On dynamic mode decomposition*, 2014; Arbabi & Mezić, *Ergodic theory, DMD and computation of spectral properties of the Koopman operator*, 2017.
* Kalman, *A new approach to linear filtering and prediction problems*, 1960.
* Edelsbrunner, Letscher, Zomorodian, *Topological persistence and simplification*, 2002; Cohen-Steiner, Edelsbrunner, Harer, *Stability of persistence diagrams*, 2007; Chazal, de Silva, Oudot, *Persistence stability for geometric complexes*, 2014. Ripser: Bauer, 2021.
* Marchenko & Pastur, 1967; Gavish & Donoho, *The optimal hard threshold for singular values is $4/\sqrt3$*, 2014.
* Bardes, Ponce, LeCun, *VICReg*, 2022; Zbontar et al., *Barlow Twins*, 2021; Assran et al., *I-JEPA*, 2023.
* He, Zuo, Ma, Sui, *DynSyn: Dynamical Synergistic Representation for Efficient Learning and Control in Overactuated Embodied Systems*, ICML 2024, arXiv 2407.11472.