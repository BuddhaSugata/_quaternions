
"""
Unimetry SN Fit — rigid template f(z) = J(z; Ωm) * sqrt(z)
(HD "instant turn + immediate renormalization" phenomenology)

What it does
------------
1) Loads Pantheon+SH0ES.dat (robust column autodetect).
2) Optional low-z cut (to suppress peculiar-velocity systematics).
3) Baseline ΛCDM fit (Ωm, M) by grid over Ωm with analytic M.
4) 1-parameter template fit Δμ(z) = A * f(z) with
   f(z) = J(z;Ωm) * sqrt(z),   J = ∂ ln D_L / ∂ ln(1+z) = 1 + ((1+z)/E)/χ
   Two modes:
     (A) Ωm fixed at the baseline ΛCDM best-fit (FREEZE_OM_TO_LCDM=True)
     (B) Ωm refit jointly with A (grid over Ωm) (FREEZE_OM_TO_LCDM=False)
   In both modes we center f by the weighted mean to remove degeneracy with M.
5) Prints χ², dof, χ²_red, AIC/BIC, Δχ² vs ΛCDM, A±σ(A).
6) Saves simple plots: residuals and Δμ(z) = A f(z).

Usage
-----
python unimetry_fit_jsqrtz.py

Configuration block below has:
DATA_PATH, Z_MIN, PREFER_ZHD, ADD_PEC_ERR, PEC_Z_CLIP,
FREEZE_OM_TO_LCDM.

Outputs
-------
- jsqrtz_template_dmu.png      (Δμ vs z)
- jsqrtz_template_resid.png    (residuals after ΛCDM+template)
- console with best-fit numbers

"""

import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from pathlib import Path

# -------- Config --------
DATA_PATH = Path("./Pantheon+SH0ES.dat")
Z_MIN = 0.03         # low-z cut (change to 0.02 if you like)
PREFER_ZHD = False   # use zHD instead of zCMB if True
ADD_PEC_ERR = True   # add peculiar-velocity error when available
PEC_Z_CLIP = 0.003   # to avoid blow-up at z~0
SIGMA_FLOOR = 0.12   # minimal σ_μ if missing

# Template fitting options
FREEZE_OM_TO_LCDM = True        # True: fix Ωm to baseline; False: refit Ωm jointly with A
OM_GRID = np.linspace(0.05, 0.6, 240)
H0 = 70.0                       # only sets overall scale in μ via M; not critical here

SUPPRESS_WARNINGS = True
if SUPPRESS_WARNINGS:
    warnings.simplefilter("ignore", category=DeprecationWarning)

c_km_s = 299792.458

# -------- I/O helpers --------
def _read_any(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_table(path, comment='#', sep=r"\s+", engine="python")
        if len(df.columns) == 1:
            df = pd.read_csv(path, comment='#')
        return df
    except Exception:
        try:
            return pd.read_csv(path, comment='#')
        except Exception:
            return pd.read_csv(path, delim_whitespace=True, comment='#')

def load_sn_table(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = _read_any(path)
    if len(df.columns) == 1:
        df = pd.read_table(path, comment='#', sep=r"\s+", engine="python")

    cols_lower = {c.lower().strip(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n.lower() in cols_lower:
                return cols_lower[n.lower()]
        return None

    # choose z
    if PREFER_ZHD:
        zcol = pick('zhd','zcmb','z')
    else:
        zcol = pick('zcmb','zhd','z')
    mucol  = pick('mu','mu_sh0es','distmod','m_b_corr','mb_corr','mb','m_b')
    sigcol = pick('mu_err','sigma_mu','mu_sh0es_err_diag','m_b_corr_err_diag','dmu','dmb','merr')
    vperr_col = pick('vpecerr','vpec_err','sigma_vpec')

    if zcol is None or mucol is None:
        raise ValueError(f"Missing essential columns. Have: {list(df.columns)}")

    out = pd.DataFrame()
    out['z'] = df[zcol].astype(float)
    out['mu_like'] = df[mucol].astype(float)

    if sigcol is not None:
        out['sigma_mu'] = df[sigcol].astype(float).clip(lower=SIGMA_FLOOR)
        sigma_src = sigcol
    else:
        out['sigma_mu'] = SIGMA_FLOOR
        sigma_src = f"floor={SIGMA_FLOOR}"

    if ADD_PEC_ERR and (vperr_col is not None):
        zv = np.clip(out['z'].values, PEC_Z_CLIP, None)
        sigma_v = df[vperr_col].astype(float).values
        sigma_mu_pec = (5.0/np.log(10.0)) * (sigma_v / (c_km_s * zv))
        out['sigma_mu'] = np.sqrt(out['sigma_mu'].values**2 + sigma_mu_pec**2)

    out = out.replace([np.inf,-np.inf], np.nan).dropna(subset=['z','mu_like','sigma_mu'])
    out = out[(out['z']>=Z_MIN) & (out['z']<=2.0)].sort_values('z').reset_index(drop=True)

    print(f"Loaded {len(out)} SNe from {path.name}")
    print(f"Using z from '{zcol}', mu-like from '{mucol}', sigma from '{sigma_src}'")
    if ADD_PEC_ERR and (vperr_col is not None):
        print(f"Including peculiar-velocity error from '{vperr_col}', z clip {PEC_Z_CLIP}")
    return out

# -------- Cosmology & sensitivities --------
def E_z(z, Om, Ol):
    Ok = 1.0 - Om - Ol
    zp1 = 1.0 + np.asarray(z, float)
    return np.sqrt(Om*zp1**3 + Ok*zp1**2 + Ol)

def comoving_distance(z, Om, Ol, Nint=800):
    z = float(z)
    if z <= 0: return 0.0
    zg = np.linspace(0.0, z, Nint)
    Ez = E_z(zg, Om, Ol)
    return (c_km_s) * np.trapz(1.0/np.maximum(Ez,1e-12), zg)

def lum_distance(z, Om, Ol, H0=H0):
    chi = comoving_distance(z, Om, Ol)
    return (1.0 + z) * chi / H0

def mu_theory(z, Om, Ol, M0):
    Dl = np.array([lum_distance(zi, Om, Ol) for zi in np.atleast_1d(z)])
    return 5.0*np.log10(np.clip(Dl,1e-9,None)) + M0

def J_sensitivity(z, Om, Ol):
    """J = d ln D_L / d ln(1+z) = 1 + ((1+z)/E)/χ  (flat-like form with Ok accounted in E, χ)"""
    z = np.asarray(z, float)
    # χ(z):
    chi = np.array([comoving_distance(zi, Om, Ol) for zi in z])
    Ez  = E_z(z, Om, Ol)
    with np.errstate(divide='ignore', invalid='ignore'):
        J = 1.0 + (1.0+z)/(np.maximum(Ez,1e-12)) / np.maximum(chi,1e-12)
    # For z→0, χ→0: the analytic limit tends to 2; clip to avoid infs in practice
    J = np.where(np.isfinite(J), J, 2.0)
    return J

# -------- Baseline ΛCDM fit --------
def fit_lcdm(sn, Om_grid=OM_GRID):
    z = sn['z'].values
    y = sn['mu_like'].values
    w = 1.0/(sn['sigma_mu'].values**2)
    best=None
    for Om in Om_grid:
        # Ol is 1-Om (flat); Ok=0 assumed in the template; small curvature would generalize analogously
        Ol = 1.0 - Om
        mu0 = np.array([5.0*np.log10(lum_distance(zi, Om, Ol)) for zi in z])
        M = np.sum(w*(y-mu0))/np.sum(w)
        r = y - (mu0 + M)
        chi2 = np.sum(w*r*r)
        if best is None or chi2<best['chi2']:
            best={'Om':float(Om),'Ol':float(Ol),'M':float(M),'chi2':float(chi2)}
    N = len(y); k = 2
    best['dof']=int(N-k); best['chi2_red']=best['chi2']/max(1,best['dof'])
    best['AIC'] = best['chi2'] + 2*k
    best['BIC'] = best['chi2'] + k*np.log(N)
    return best

# -------- Template fit: Δμ = A * (J * sqrt(z)) --------
def fit_jsqrtz(sn, Om_grid=OM_GRID, freeze_Om_to=None):
    z = sn['z'].values
    y = sn['mu_like'].values
    w = 1.0/(sn['sigma_mu'].values**2)
    best=None
    if freeze_Om_to is not None:
        Om_list = [float(freeze_Om_to)]
    else:
        Om_list = Om_grid

    for Om in Om_list:
        Ol = 1.0 - Om
        mu0 = np.array([5.0*np.log10(lum_distance(zi, Om, Ol)) for zi in z])
        # Centered f(z) to avoid degeneracy with M
        J = J_sensitivity(z, Om, Ol)
        f = J * np.sqrt(np.clip(z,0,None))
        f = f - np.average(f, weights=w)

        # Linear solve for [M, A]
        X = np.column_stack([np.ones_like(z), f])  # [M, A]
        W = np.diag(w)
        A_mat = X.T @ W @ X
        b_vec = X.T @ W @ (y - mu0)
        theta = np.linalg.solve(A_mat, b_vec)
        M_hat, A_hat = float(theta[0]), float(theta[1])
        resid = y - (mu0 + M_hat + A_hat*f)
        chi2  = float(np.sum(w*resid*resid))
        cov   = np.linalg.inv(A_mat)
        sigA  = float(np.sqrt(cov[1,1]))

        if (best is None) or (chi2<best['chi2']):
            best={'Om':float(Om),'Ol':float(Ol),'M':M_hat,'A':A_hat,'sigma_A':sigA,
                  'chi2':chi2,'f':f,'resid':resid}

    N = len(y); k = 3  # Om chosen by grid, but parameter count in χ² for [M,A,Om?] we keep 3 for AIC fairness
    best['dof']=int(N-k); best['chi2_red']=best['chi2']/max(1,best['dof'])
    best['AIC'] = best['chi2'] + 2*k
    best['BIC'] = best['chi2'] + k*np.log(N)
    return best

# -------- Plots --------
def plot_jsqrtz(sn, lcdm, templ, prefix="jsqrtz_template"):
    z = sn['z'].values; y = sn['mu_like'].values
    Om = templ['Om']; Ol = templ['Ol']
    J  = J_sensitivity(z, Om, Ol)
    f  = J * np.sqrt(np.clip(z,0,None))
    f  = f - np.average(f, weights=1.0/(sn['sigma_mu'].values**2))

    mu0 = np.array([5.0*np.log10(lum_distance(zi, Om, Ol)) for zi in z])
    muT = mu0 + templ['M'] + templ['A']*f

    # Δμ(z)
    idx = np.argsort(z)
    plt.figure(figsize=(8,4.6))
    plt.plot(z[idx], (templ['A']*f)[idx])
    plt.xlabel("z"); plt.ylabel("Δμ(z) [mag]")
    plt.title(f"Δμ = A·J(z)·√z with Ωm={Om:.3f},  A={templ['A']:+.4f}±{templ['sigma_A']:.4f}")
    plt.tight_layout(); plt.savefig(f"{prefix}_dmu.png", dpi=160)

    # Residuals after template
    plt.figure(figsize=(8,5))
    plt.scatter(z, y - muT, s=6, alpha=0.5)
    plt.xlabel("z"); plt.ylabel("μ_obs - μ_model")
    plt.title("Residuals after ΛCDM + A·J·√z")
    plt.tight_layout(); plt.savefig(f"{prefix}_resid.png", dpi=160)

# -------- Main --------
def main():
    sn = load_sn_table(DATA_PATH)
    lcdm = fit_lcdm(sn)

    if FREEZE_OM_TO_LCDM:
        templ = fit_jsqrtz(sn, freeze_Om_to=lcdm['Om'])
        mode = "Ωm fixed to ΛCDM best-fit"
    else:
        templ = fit_jsqrtz(sn)
        mode = "Ωm refit jointly with A"

    print("\n=== RESULTS ===")
    print(f"z-min cut: {Z_MIN:.3f}, ADD_PEC_ERR={ADD_PEC_ERR}, PREFER_ZHD={PREFER_ZHD}")
    print(f"LCDM:   chi2={lcdm['chi2']:.1f}, dof={lcdm['dof']}, chi2_red={lcdm['chi2_red']:.3f}, AIC={lcdm['AIC']:.1f}, BIC={lcdm['BIC']:.1f}")
    print(f"J√z ({mode}):")
    print(f"  Om={templ['Om']:.3f},  A={templ['A']:+.4f} ± {templ['sigma_A']:.4f} mag")
    print(f"  chi2={templ['chi2']:.1f}, dof={templ['dof']}, chi2_red={templ['chi2_red']:.3f}, AIC={templ['AIC']:.1f}, BIC={templ['BIC']:.1f}")
    print(f"Δχ² (ΛCDM→J√z) = {lcdm['chi2']-templ['chi2']:.2f} for {templ['dof']-lcdm['dof']:+d} params")

    plot_jsqrtz(sn, lcdm, templ, prefix="jsqrtz_template")
    print("Saved: jsqrtz_template_dmu.png, jsqrtz_template_resid.png")

if __name__ == "__main__":
    main()
