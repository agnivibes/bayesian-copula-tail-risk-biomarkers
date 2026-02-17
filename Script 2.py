# ============================================================
# NHANES GLU + GHB + Simulation (BAYESIAN ONLY)
# Clayton + Gumbel copulas (bivariate)
#
# THEORY-FAITHFUL tail risks:
#   RL(alpha) = C(alpha, alpha)
#   RU(alpha) = P(U>=1-alpha, V>=1-alpha) = 2*alpha - 1 + C(1-alpha, 1-alpha)
#   RC(alpha) = RL(alpha) / alpha
#
# INFERENCE (Bayesian, matches theory):
#   - Pseudo-observations (ranks/(n+1))
#   - Copula log-likelihood via Archimedean density formula
#   - Restricted Jeffreys prior pi(theta) ∝ sqrt(I(theta)) on [theta_min, theta_max]
#       * I(theta) precomputed ONCE per family via MC (Genest-Rivest sampler)
#   - Posterior over theta on a grid; posterior summaries for RL/RU/RC
#
# SIMULATION (Bayesian):
#   - Generate (U,V) via Genest-Rivest Algorithm I
#   - For each replicate: posterior grid + induced CI for RL/RU/RC
#   - Report posterior mean bias + Bayesian CI coverage
#
# FIGURES:
#   Fig1: Scatter + marginal hists (raw scale)
#   Fig2: Posterior distributions of RL and RU (Clayton vs Gumbel)
# ============================================================

import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings("ignore")

from scipy.optimize import brentq, minimize_scalar
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# -----------------------------
# 0) USER SETTINGS
# -----------------------------
DATA_CSV = "NHANES_2017_2018_glu_ghb.csv"
XCOL = "glu"
YCOL = "ghb"

ALPHA = 0.05

# Simulation
SIM_N = 500
SIM_REPS = 50
SIM_THETAS = [2.0, 5.0, 10.0]

# Bayesian / grid settings
THETA_MAX = 50.0
POST_GRID_SIZE = 1800        # posterior theta grid size (speed vs resolution)
POST_SAMPLES = 30000         # posterior draws (for induced RL/RU/RC CIs)
CI_LEVEL = 0.95              # 0.95 -> 95% credible intervals

# Jeffreys prior precompute (ONCE per family)
JEFFREYS_MC = 2500           # MC sample size per theta point (increase if you want smoother prior)
JEFFREYS_H = 1e-4            # finite diff step for score
JEFFREYS_GRID_SIZE = 220     # theta points for prior precompute (interpolated to posterior grid)
CACHE_PRIOR = True
PRIOR_CACHE_FILE = "jeffreys_prior_cache_clayton_gumbel.npz"

RNG_SEED = 123
np.random.seed(RNG_SEED)

EPS = 1e-12

# -----------------------------
# 1) LOAD DATA
# -----------------------------
df = pd.read_csv(DATA_CSV)
df = df[[XCOL, YCOL]].dropna().reset_index(drop=True)

print("Data file:", DATA_CSV)
print("Merged shape:", df.shape)
print(df[[XCOL, YCOL]].describe())

x = df[XCOL].to_numpy()
y = df[YCOL].to_numpy()

# -----------------------------
# 2) PSEUDO-OBSERVATIONS
# -----------------------------
def pseudo_obs(z):
    """Rank-based pseudo-observations r/(n+1) with simple tie-averaging."""
    z = np.asarray(z)
    n = len(z)
    order = np.argsort(z, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)

    z_sorted = z[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and z_sorted[j + 1] == z_sorted[i]:
            j += 1
        if j > i:
            avg = 0.5 * (i + 1 + j + 1)
            ranks[order[i:j + 1]] = avg
        i = j + 1

    u = ranks / (n + 1.0)
    return np.clip(u, EPS, 1.0 - EPS)

u_data = pseudo_obs(x)
v_data = pseudo_obs(y)

# -----------------------------
# 3) COPULA CDFs + tail risks
# -----------------------------
def clayton_cdf(u, v, theta):
    u = np.asarray(u); v = np.asarray(v)
    t = np.maximum(u**(-theta) + v**(-theta) - 1.0, 0.0)
    return t ** (-1.0/theta)

def gumbel_cdf(u, v, theta):
    u = np.asarray(u); v = np.asarray(v)
    X = (-np.log(u))**theta
    Y = (-np.log(v))**theta
    a = X + Y
    return np.exp(-(a**(1.0/theta)))

def C_uv(fam, u, v, theta):
    if fam == "clayton":
        return clayton_cdf(u, v, theta)
    if fam == "gumbel":
        return gumbel_cdf(u, v, theta)
    raise ValueError("fam must be 'clayton' or 'gumbel'")

def tail_risks_from_theta(fam, theta, alpha):
    a = float(np.clip(alpha, EPS, 1.0 - EPS))
    Caa = float(C_uv(fam, a, a, theta))
    Cuu = float(C_uv(fam, 1.0 - a, 1.0 - a, theta))
    RL = Caa
    RU = 2.0*a - 1.0 + Cuu
    RC = RL / a
    return float(RL), float(RU), float(RC)

# -----------------------------
# 4) ARCHIMEDEAN density via generator formula
# c(u,v) = - phi''(C) * phi'(u) * phi'(v) / (phi'(C))^3
# -----------------------------
def phi_prime(fam, t, theta):
    t = np.asarray(t)
    t = np.clip(t, EPS, 1.0 - EPS)
    if fam == "clayton":
        return -t**(-theta - 1.0)
    if fam == "gumbel":
        return -(theta * (-np.log(t))**(theta - 1.0)) / t
    raise ValueError("fam must be 'clayton' or 'gumbel'")

def phi_double_prime(fam, t, theta):
    t = np.asarray(t)
    t = np.clip(t, EPS, 1.0 - EPS)
    if fam == "clayton":
        return (theta + 1.0) * t**(-theta - 2.0)
    if fam == "gumbel":
        L = -np.log(t)
        return (theta * (L**(theta - 2.0)) * ((theta - 1.0) + L)) / (t**2)
    raise ValueError("fam must be 'clayton' or 'gumbel'")

def copula_density_archimedean(fam, u, v, theta):
    u = np.asarray(u); v = np.asarray(v)
    u = np.clip(u, EPS, 1.0 - EPS)
    v = np.clip(v, EPS, 1.0 - EPS)

    C = C_uv(fam, u, v, theta)
    C = np.clip(C, EPS, 1.0 - EPS)

    num = -phi_double_prime(fam, C, theta) * phi_prime(fam, u, theta) * phi_prime(fam, v, theta)
    den = (phi_prime(fam, C, theta))**3
    c = num / den
    return np.clip(c, EPS, np.inf)

def loglik_theta(fam, theta, u, v):
    theta = float(theta)
    if fam == "clayton" and theta <= 0:
        return -np.inf
    if fam == "gumbel" and theta < 1:
        return -np.inf
    c = copula_density_archimedean(fam, u, v, theta)
    return float(np.sum(np.log(c)))

# -----------------------------
# 5) MLE (optional, used as diagnostic / sanity)
# -----------------------------
def mle_theta(fam, u, v, theta_max=THETA_MAX):
    if fam == "clayton":
        lo, hi = 1e-4, theta_max
    else:
        lo, hi = 1.0 + 1e-6, theta_max

    def obj(th):
        ll = loglik_theta(fam, th, u, v)
        return -ll if np.isfinite(ll) else 1e100

    res = minimize_scalar(obj, bounds=(lo, hi), method="bounded", options={"xatol": 1e-6})
    th_hat = float(res.x)
    ll_hat = -float(res.fun)
    return th_hat, ll_hat

# -----------------------------
# 6) Genest-Rivest Algorithm I sampler (Archimedean)
# -----------------------------
def phi_clayton(u, theta):
    return (u**(-theta) - 1.0) / theta

def phi_inv_clayton(x, theta):
    return (1.0 + theta * x) ** (-1.0/theta)

def K_clayton(u, theta):
    return u + (u - u**(theta + 1.0)) / theta

def phi_gumbel(u, theta):
    return (-np.log(u))**theta

def phi_inv_gumbel(x, theta):
    return np.exp(-(x**(1.0/theta)))

def K_gumbel(u, theta):
    return u * (1.0 + (-np.log(u))/theta)

def K_inv_scalar(t, fam, theta):
    t = float(np.clip(t, EPS, 1.0 - EPS))
    if fam == "clayton":
        f = lambda w: K_clayton(w, theta) - t
    else:
        f = lambda w: K_gumbel(w, theta) - t

    lo, hi = EPS, 1.0 - EPS
    flo, fhi = f(lo), f(hi)
    if flo > 0:
        return lo
    if fhi < 0:
        return hi
    return float(brentq(f, lo, hi, maxiter=200))

def rarch_genest1(n, fam, theta, rng):
    s = rng.uniform(0.0, 1.0, size=n)
    t = rng.uniform(0.0, 1.0, size=n)
    w = np.array([K_inv_scalar(tt, fam, theta) for tt in t], dtype=float)

    if fam == "clayton":
        phiw = phi_clayton(w, theta)
        u = phi_inv_clayton(s * phiw, theta)
        v = phi_inv_clayton((1.0 - s) * phiw, theta)
    else:
        phiw = phi_gumbel(w, theta)
        u = phi_inv_gumbel(s * phiw, theta)
        v = phi_inv_gumbel((1.0 - s) * phiw, theta)

    u = np.clip(u, EPS, 1.0 - EPS)
    v = np.clip(v, EPS, 1.0 - EPS)
    return u, v

# -----------------------------
# 7) Jeffreys prior precompute: pi(theta) ∝ sqrt(I(theta))
# I(theta) = E[(d/dtheta log c_theta(U,V))^2]
# Score approximated by central finite differences
# -----------------------------
def score_theta_fd(fam, theta, u, v, h):
    th1 = theta + h
    th0 = theta - h

    if fam == "clayton" and th0 <= 0:
        th0 = theta
    if fam == "gumbel" and th0 < 1.0:
        th0 = theta

    if th1 == th0:
        th1 = theta + h

    logc1 = np.log(copula_density_archimedean(fam, u, v, th1))
    logc0 = np.log(copula_density_archimedean(fam, u, v, th0))
    return (logc1 - logc0) / (th1 - th0)

def fisher_info_mc(fam, theta, m=JEFFREYS_MC, h=JEFFREYS_H, seed=0):
    rng = np.random.default_rng(seed)
    u, v = rarch_genest1(m, fam, theta, rng)
    s = score_theta_fd(fam, theta, u, v, h)
    I = float(np.mean(s**2))
    if (not np.isfinite(I)) or (I <= 0):
        I = 1e-12
    return I

def make_theta_grid(fam, size, theta_max):
    if fam == "clayton":
        return np.exp(np.linspace(np.log(1e-4), np.log(theta_max), size))
    else:
        # theta in [1, theta_max]
        return 1.0 + np.exp(np.linspace(np.log(1e-6), np.log(theta_max - 1.0 + 1e-6), size))

def precompute_jeffreys_prior():
    thC = make_theta_grid("clayton", JEFFREYS_GRID_SIZE, THETA_MAX)
    thG = make_theta_grid("gumbel",  JEFFREYS_GRID_SIZE, THETA_MAX)

    I_C = np.array([fisher_info_mc("clayton", th, seed=10_000 + i) for i, th in enumerate(thC)], dtype=float)
    I_G = np.array([fisher_info_mc("gumbel",  th, seed=20_000 + i) for i, th in enumerate(thG)], dtype=float)

    priorC = np.sqrt(np.clip(I_C, 1e-300, np.inf))
    priorG = np.sqrt(np.clip(I_G, 1e-300, np.inf))

    return thC, priorC, thG, priorG

def load_or_build_prior():
    if CACHE_PRIOR:
        try:
            z = np.load(PRIOR_CACHE_FILE, allow_pickle=False)
            thC = z["thC"]; priorC = z["priorC"]
            thG = z["thG"]; priorG = z["priorG"]
            print(f"\nLoaded Jeffreys prior cache: {PRIOR_CACHE_FILE}")
            return thC, priorC, thG, priorG
        except Exception:
            pass

    print("\nPrecomputing Jeffreys prior (one-time)...")
    thC, priorC, thG, priorG = precompute_jeffreys_prior()

    if CACHE_PRIOR:
        np.savez(PRIOR_CACHE_FILE, thC=thC, priorC=priorC, thG=thG, priorG=priorG)
        print(f"Saved Jeffreys prior cache: {PRIOR_CACHE_FILE}")

    return thC, priorC, thG, priorG

thC_prior_grid, priorC_vals, thG_prior_grid, priorG_vals = load_or_build_prior()

# -----------------------------
# 8) Posterior over theta on a grid
# posterior(theta) ∝ L(theta; data) * prior(theta)
# prior(theta) obtained by interpolation of Jeffreys-precompute
# -----------------------------
def interp_prior(theta_grid, fam):
    if fam == "clayton":
        x = thC_prior_grid
        y = priorC_vals
    else:
        x = thG_prior_grid
        y = priorG_vals

    # Interpolate in log-space for stability
    logy = np.log(np.clip(y, 1e-300, np.inf))
    logp = np.interp(theta_grid, x, logy, left=logy[0], right=logy[-1])
    return np.exp(logp)

def posterior_theta_grid(fam, u, v, theta_max=THETA_MAX):
    theta_grid = make_theta_grid(fam, POST_GRID_SIZE, theta_max)

    logliks = np.array([loglik_theta(fam, th, u, v) for th in theta_grid], dtype=float)

    prior = interp_prior(theta_grid, fam)
    prior = np.clip(prior, 1e-300, np.inf)
    logpost_unn = logliks + np.log(prior)

    logpost_unn[~np.isfinite(logpost_unn)] = -np.inf
    logZ = logsumexp(logpost_unn)
    if not np.isfinite(logZ):
        raise ValueError("Posterior normalization failed (logZ not finite).")

    w = np.exp(logpost_unn - logZ)
    w[~np.isfinite(w)] = 0.0
    s = float(np.sum(w))
    if (not np.isfinite(s)) or (s <= 0):
        raise ValueError("Posterior weights invalid (sum <= 0).")
    w /= s

    map_idx = int(np.argmax(w))
    theta_map = float(theta_grid[map_idx])
    theta_mean = float(np.sum(theta_grid * w))
    return theta_grid, w, theta_map, theta_mean, float(logZ)

def posterior_draws(theta_grid, w, size=POST_SAMPLES, seed=12345):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(theta_grid), size=size, replace=True, p=w)
    return theta_grid[idx]

def induced_posterior_tailrisks(fam, theta_draws, alpha):
    RL = np.empty_like(theta_draws, dtype=float)
    RU = np.empty_like(theta_draws, dtype=float)
    RC = np.empty_like(theta_draws, dtype=float)
    for i, th in enumerate(theta_draws):
        rl, ru, rc = tail_risks_from_theta(fam, float(th), alpha)
        RL[i], RU[i], RC[i] = rl, ru, rc
    return RL, RU, RC

def summarize_posterior(theta_grid, w, fam, alpha, seed=0):
    th_draws = posterior_draws(theta_grid, w, size=POST_SAMPLES, seed=seed)
    RLd, RUd, RCd = induced_posterior_tailrisks(fam, th_draws, alpha)

    a = 1.0 - CI_LEVEL
    loq, hiq = a/2.0, 1.0 - a/2.0

    def ci(arr):
        return (float(np.quantile(arr, loq)), float(np.quantile(arr, hiq)))

    return {
        "theta_post_mean": float(np.mean(th_draws)),
        "theta_post_ci": ci(th_draws),
        "RL_post_mean": float(np.mean(RLd)),
        "RL_post_ci": ci(RLd),
        "RU_post_mean": float(np.mean(RUd)),
        "RU_post_ci": ci(RUd),
        "RC_post_mean": float(np.mean(RCd)),
        "RC_post_ci": ci(RCd),
        "_draws": {"theta": th_draws, "RL": RLd, "RU": RUd, "RC": RCd}
    }

# -----------------------------
# 9) REAL DATA: Bayesian posterior + (optional) MLE diagnostics
# -----------------------------
print("\nMLE FITS (PSEUDO-OBS, REAL DATA) [diagnostic only]")
thetaC_mle, llC = mle_theta("clayton", u_data, v_data)
thetaG_mle, llG = mle_theta("gumbel",  u_data, v_data)
print("Clayton theta_hat (MLE):", thetaC_mle, "| loglik:", llC)
print("Gumbel  theta_hat (MLE):", thetaG_mle, "| loglik:", llG)

print("\nBAYESIAN POSTERIOR (Restricted Jeffreys, grid-based)")
thgC, wC, thetaC_map, thetaC_mean, logZC = posterior_theta_grid("clayton", u_data, v_data)
postC = summarize_posterior(thgC, wC, "clayton", ALPHA, seed=777)
print("Clayton: theta_MAP =", thetaC_map, "| theta_mean =", thetaC_mean)
print("Clayton posterior summary:", {k: v for k, v in postC.items() if k != "_draws"})

thgG, wG, thetaG_map, thetaG_mean, logZG = posterior_theta_grid("gumbel", u_data, v_data)
postG = summarize_posterior(thgG, wG, "gumbel", ALPHA, seed=888)
print("Gumbel : theta_MAP =", thetaG_map, "| theta_mean =", thetaG_mean)
print("Gumbel posterior summary:", {k: v for k, v in postG.items() if k != "_draws"})

# Independence baseline for RU at alpha: alpha^2
indep_joint = ALPHA**2

print(f"\nREAL DATA interpretation sanity check @ alpha={ALPHA}")
print("Independence baseline (alpha^2) =", indep_joint)
print("Bayes RU Clayton mean =", postC["RU_post_mean"], "| multiple of indep =", postC["RU_post_mean"]/indep_joint)
print("Bayes RU Gumbel  mean =", postG["RU_post_mean"], "| multiple of indep =", postG["RU_post_mean"]/indep_joint)

real = {
    "pair": "GLU_GHB",
    "n": int(len(df)),
    "alpha": float(ALPHA),
    "independence_alpha2": float(indep_joint),
    "mle_diagnostic": {
        "clayton": {"theta_mle": float(thetaC_mle), "loglik": float(llC)},
        "gumbel":  {"theta_mle": float(thetaG_mle), "loglik": float(llG)}
    },
    "bayes_restricted_jeffreys": {
        "theta_max": float(THETA_MAX),
        "clayton": {
            "theta_map": float(thetaC_map),
            "theta_mean": float(thetaC_mean),
            "theta_ci": postC["theta_post_ci"],
            "RL_mean": postC["RL_post_mean"], "RL_ci": postC["RL_post_ci"],
            "RU_mean": postC["RU_post_mean"], "RU_ci": postC["RU_post_ci"],
            "RC_mean": postC["RC_post_mean"], "RC_ci": postC["RC_post_ci"],
        },
        "gumbel": {
            "theta_map": float(thetaG_map),
            "theta_mean": float(thetaG_mean),
            "theta_ci": postG["theta_post_ci"],
            "RL_mean": postG["RL_post_mean"], "RL_ci": postG["RL_post_ci"],
            "RU_mean": postG["RU_post_mean"], "RU_ci": postG["RU_post_ci"],
            "RC_mean": postG["RC_post_mean"], "RC_ci": postG["RC_post_ci"],
        }
    }
}

# -----------------------------
# 10) FIGURE 1: Scatter + marginal hists (raw scale)
# -----------------------------
def fig1_scatter_marginals(x, y, out_png="Figure1_scatter_marginals.png"):
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(4, 4, figure=fig)

    ax_scatter = fig.add_subplot(gs[1:4, 0:3])
    ax_histx = fig.add_subplot(gs[0, 0:3], sharex=ax_scatter)
    ax_histy = fig.add_subplot(gs[1:4, 3], sharey=ax_scatter)

    ax_scatter.scatter(x, y, s=10, alpha=0.4)
    ax_scatter.set_xlabel(XCOL.upper())
    ax_scatter.set_ylabel(YCOL.upper())
    ax_scatter.set_title("NHANES GLU vs GHB (raw scale)")

    ax_histx.hist(x, bins=40, density=True, alpha=0.6)
    ax_histy.hist(y, bins=40, density=True, orientation="horizontal", alpha=0.6)

    plt.setp(ax_histx.get_xticklabels(), visible=False)
    plt.setp(ax_histy.get_yticklabels(), visible=False)

    ax_histx.set_ylabel("Density")
    ax_histy.set_xlabel("Density")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    print("Saved:", out_png)

fig1_scatter_marginals(x, y)

# -----------------------------
# 11) FIGURE 2: Posterior distributions of RL and RU (Bayesian)
# -----------------------------
def fig2_posterior_tailrisks(postC, postG, out_png="Figure2_posterior_tailrisks.png"):
    RLc = postC["_draws"]["RL"]; RUc = postC["_draws"]["RU"]
    RLg = postG["_draws"]["RL"]; RUg = postG["_draws"]["RU"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Panel A: RL
    ax = axes[0]
    ax.hist(RLc, bins=60, density=True, alpha=0.5, label="Clayton")
    ax.hist(RLg, bins=60, density=True, alpha=0.5, label="Gumbel")
    ax.axvline(postC["RL_post_mean"], linewidth=2)
    ax.axvline(postG["RL_post_mean"], linewidth=2)
    ax.axvspan(postC["RL_post_ci"][0], postC["RL_post_ci"][1], alpha=0.15)
    ax.axvspan(postG["RL_post_ci"][0], postG["RL_post_ci"][1], alpha=0.15)
    ax.set_title(f"Posterior: $R_L(\\alpha)$ with $\\alpha={ALPHA}$")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()

    # Panel B: RU
    ax = axes[1]
    ax.hist(RUc, bins=60, density=True, alpha=0.5, label="Clayton")
    ax.hist(RUg, bins=60, density=True, alpha=0.5, label="Gumbel")
    ax.axvline(postC["RU_post_mean"], linewidth=2)
    ax.axvline(postG["RU_post_mean"], linewidth=2)
    ax.axvspan(postC["RU_post_ci"][0], postC["RU_post_ci"][1], alpha=0.15)
    ax.axvspan(postG["RU_post_ci"][0], postG["RU_post_ci"][1], alpha=0.15)
    ax.axvline(indep_joint, linestyle="--", linewidth=2)  # independence baseline
    ax.set_title(f"Posterior: $R_U(\\alpha)$ with $\\alpha={ALPHA}$")
    ax.set_xlabel("Value")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    print("Saved:", out_png)

fig2_posterior_tailrisks(postC, postG)

# -----------------------------
# 12) SIMULATION STUDY (Bayesian)
# coverage = fraction of times true RL/RU/RC lies in posterior CI
# -----------------------------
def sim_study_bayes(fam, theta_true, n, reps, alpha, seed):
    rng = np.random.default_rng(seed)

    RL_true, RU_true, RC_true = tail_risks_from_theta(fam, theta_true, alpha)

    cover_RL = cover_RU = cover_RC = 0
    kept = 0

    # store posterior means (to report bias-like summaries)
    RL_means = []
    RU_means = []
    RC_means = []
    th_means = []

    for r in range(reps):
        u, v = rarch_genest1(n, fam, theta_true, rng)

        thg, w, th_map, th_mean, _ = posterior_theta_grid(fam, u, v)
        post = summarize_posterior(thg, w, fam, alpha, seed=1000 + r)

        loRL, hiRL = post["RL_post_ci"]
        loRU, hiRU = post["RU_post_ci"]
        loRC, hiRC = post["RC_post_ci"]

        cover_RL += int(loRL <= RL_true <= hiRL)
        cover_RU += int(loRU <= RU_true <= hiRU)
        cover_RC += int(loRC <= RC_true <= hiRC)

        RL_means.append(post["RL_post_mean"])
        RU_means.append(post["RU_post_mean"])
        RC_means.append(post["RC_post_mean"])
        th_means.append(post["theta_post_mean"])

        kept += 1

    return {
        "fam": fam,
        "theta_true": float(theta_true),
        "n": int(n),
        "reps": int(kept),
        "RL_true": float(RL_true),
        "RU_true": float(RU_true),
        "RC_true": float(RC_true),
        "theta_post_mean_avg": float(np.mean(th_means)) if kept else np.nan,
        "RL_post_mean_avg": float(np.mean(RL_means)) if kept else np.nan,
        "RU_post_mean_avg": float(np.mean(RU_means)) if kept else np.nan,
        "RC_post_mean_avg": float(np.mean(RC_means)) if kept else np.nan,
        "coverage_RL": float(cover_RL / kept) if kept else np.nan,
        "coverage_RU": float(cover_RU / kept) if kept else np.nan,
        "coverage_RC": float(cover_RC / kept) if kept else np.nan,
    }

print("\nSIMULATION STUDY (Bayesian posterior credible-interval coverage)")
sim_results = {"clayton": {}, "gumbel": {}}
for th in SIM_THETAS:
    sim_results["clayton"][str(th)] = sim_study_bayes(
        fam="clayton", theta_true=th,
        n=SIM_N, reps=SIM_REPS, alpha=ALPHA,
        seed=RNG_SEED + 100 + int(th*10)
    )
    sim_results["gumbel"][str(th)] = sim_study_bayes(
        fam="gumbel", theta_true=th,
        n=SIM_N, reps=SIM_REPS, alpha=ALPHA,
        seed=RNG_SEED + 200 + int(th*10)
    )

print("\nSIM RESULTS (Clayton)")
for k, v in sim_results["clayton"].items():
    print(k, "->", v)

print("\nSIM RESULTS (Gumbel)")
for k, v in sim_results["gumbel"].items():
    print(k, "->", v)

# -----------------------------
# 13) SAVE RESULTS (JSON)
# -----------------------------
results = {
    "realdata": real,
    "simulation_bayes": sim_results,
    "settings": {
        "ALPHA": float(ALPHA),
        "THETA_MAX": float(THETA_MAX),
        "POST_GRID_SIZE": int(POST_GRID_SIZE),
        "POST_SAMPLES": int(POST_SAMPLES),
        "CI_LEVEL": float(CI_LEVEL),
        "SIM_N": int(SIM_N),
        "SIM_REPS": int(SIM_REPS),
        "SIM_THETAS": SIM_THETAS,
        "JEFFREYS_MC": int(JEFFREYS_MC),
        "JEFFREYS_GRID_SIZE": int(JEFFREYS_GRID_SIZE),
        "JEFFREYS_H": float(JEFFREYS_H),
        "PRIOR_CACHE_FILE": PRIOR_CACHE_FILE if CACHE_PRIOR else None
    }
}

out_json = "nhanes_glu_ghb_bayes_only_results.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved:", out_json)
print("Saved: Figure1_scatter_marginals.png")
print("Saved: Figure2_posterior_tailrisks.png")
