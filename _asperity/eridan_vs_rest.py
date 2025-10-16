#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eridan_vs_rest.py — "Эридан vs остальное небо" + A-fit
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- optional SciPy ----------
try:
    from scipy.stats import chi2
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ---------- small utils ----------
def trapz_safe(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)

# ---------- geometry ----------
def radec_to_gal(ra_deg, dec_deg):
    ra = np.deg2rad(ra_deg); dec = np.deg2rad(dec_deg)
    x = np.cos(dec)*np.cos(ra); y = np.cos(dec)*np.sin(ra); z = np.sin(dec)
    R = np.array([[-0.0548755604, -0.8734370902, -0.4838350155],
                  [ 0.4941094279, -0.4448296300,  0.7469822445],
                  [-0.8676661490, -0.1980763734,  0.4559837762]])
    xg, yg, zg = (R @ np.vstack([x,y,z]))
    b = np.arcsin(np.clip(zg, -1, 1))
    l = (np.arctan2(yg, xg)) % (2*np.pi)
    return np.rad2deg(l), np.rad2deg(b)

def angsep_deg(lon1, lat1, lon2, lat2):
    lam1 = np.deg2rad(lon1); phi1 = np.deg2rad(lat1)
    lam2 = np.deg2rad(lon2); phi2 = np.deg2rad(lat2)
    dlam = lam2 - lam1; dphi = phi2 - phi1
    sin2 = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2.0)**2
    return np.rad2deg(2*np.arcsin(np.minimum(1.0, np.sqrt(sin2))))

# ---------- cosmology & mappings ----------
c_km_s = 299792.458

def z_to_zeta(z):
    z = np.clip(z, 0, None)
    return 2.0*np.arctan(np.sqrt(z))

def E_z(z, Om, Ol):
    return np.sqrt(Om*(1+z)**3 + Ol)

def dL_Mpc(z, H0, Om):
    Ol = 1.0 - Om
    out = np.empty_like(z, dtype=float)
    for i, zi in enumerate(z):
        zg = np.linspace(0, zi, max(100, int(zi*400)+1))
        Ez = E_z(zg, Om, Ol)
        Dc = (c_km_s/H0) * trapz_safe(1.0/np.maximum(Ez,1e-12), zg)
        out[i] = (1+zi)*Dc
    return out

def mu_of_z(z, H0=70.0, Om=0.3):
    dL = dL_Mpc(z, H0, Om)
    return 5.0*np.log10(np.maximum(dL,1e-9)) + 25.0

# ---------- loading ----------
GUESS_COLS = dict(
    z = ['zHD','zCMB','zcmb','z'],
    mu = ['MU_SH0ES','mu','m_b_corr','m_b'],
    sig=['MU_SH0ES_ERR_DIAG','mu_err','dmu','m_b_corr_err_DIAG','sigma_mu'],
    ra = ['RA','ra'],
    dec= ['DEC','dec']
)

def pick_col(df, keys):
    for k in keys:
        if k in df.columns: return k
    return None

def load_sn(path, prefer='zhd'):
    df = pd.read_csv(path, sep=r"\s+", comment='#', engine='python')
    zcol = None
    if prefer.lower()=='zhd' and 'zHD' in df.columns: zcol='zHD'
    elif prefer.lower()=='zcmb' and 'zCMB' in df.columns: zcol='zCMB'
    if zcol is None: zcol = pick_col(df, GUESS_COLS['z'])
    mucol = pick_col(df, GUESS_COLS['mu'])
    sigcol= pick_col(df, GUESS_COLS['sig'])
    racol = pick_col(df, GUESS_COLS['ra'])
    decol = pick_col(df, GUESS_COLS['dec'])
    if any(c is None for c in [zcol, mucol, sigcol, racol, decol]):
        raise ValueError("Нет нужных колонок (z, mu, mu_err, RA, DEC). Есть: "+str(list(df.columns)))
    out = pd.DataFrame(dict(
        z = df[zcol].astype(float).values,
        mu= df[mucol].astype(float).values,
        sig= df[sigcol].astype(float).values,
        RA= df[racol].astype(float).values,
        DEC= df[decol].astype(float).values
    ))
    m = np.isfinite(out[['z','mu','sig','RA','DEC']]).all(axis=1).values
    out = out[m].copy()
    out = out[out['z']>=0.01].reset_index(drop=True)
    return out

# ---------- binning & stats ----------
def bin_stats(x, y, nbins):
    x = np.asarray(x); y = np.asarray(y)
    be = np.linspace(x.min(), x.max(), nbins+1)
    bc = 0.5*(be[:-1] + be[1:])
    cnt = np.zeros(nbins, dtype=int)
    mu  = np.full(nbins, np.nan)
    var = np.full(nbins, np.nan)
    for i in range(nbins):
        m = (x>=be[i]) & (x<be[i+1])
        yi = y[m]
        cnt[i]= yi.size
        if yi.size>0:
            mu[i] = yi.mean()
            var[i]= yi.var(ddof=1) if yi.size>1 else 0.0
    frac = 100.0*cnt/np.maximum(1,cnt.sum())
    return be, bc, cnt, mu, var, frac

def acf_1d(resid, nlags=20):
    r = np.asarray(resid, float)
    r = r[np.isfinite(r)]
    r = r - r.mean()
    n = r.size
    if n<2: return np.zeros(nlags+1)
    ac = np.correlate(r, r, mode='full')[n-1:n+nlags]
    denom = np.arange(n, n-nlags-1, -1, dtype=float)
    ac = ac/denom
    ac0 = ac[0] if ac[0]!=0 else 1.0
    return ac/ac0

def ljung_box_pvalues(acf, n):
    hmax = len(acf)-1
    if hmax<1:
        return np.array([])
    Q = np.zeros(hmax)
    for h in range(1,hmax+1):
        k = np.arange(1,h+1)
        Q[h-1] = n*(n+2)*np.sum(acf[1:h+1]**2/(n-k))
    if HAVE_SCIPY:
        p = chi2.sf(Q, df=np.arange(1,hmax+1))
    else:
        p = np.full_like(Q, np.nan, dtype=float)
    return p

# ---------- A-projection helpers ----------
def J_factor(z, kind='1pz'):
    z = np.asarray(z, float)
    if kind == 'none':
        return np.ones_like(z)
    if kind == '1pz':
        return 1.0/(1.0+z)
    if kind == 'sec':
        zeta = z_to_zeta(z)
        val = 1.0/np.cos(zeta)
        return np.clip(val, 1.0, 10.0)  # safety near z~1
    return np.ones_like(z)

def estimate_A(resid, sigma, z, J_kind='1pz'):
    z = np.asarray(z, float)
    f = J_factor(z, J_kind) * np.sqrt(np.clip(z, 0, None))
    w = 1.0/np.maximum(np.asarray(sigma, float), 1e-9)**2
    S1 = np.sum(w * f * resid)
    S2 = np.sum(w * f * f)
    if S2 <= 0:
        return dict(A=np.nan, A_err=np.nan, chi2=np.nan, chi2_null=np.nan, dchi2=np.nan, p=np.nan, N=len(resid))
    A = S1 / S2
    chi2_fit  = np.sum(w * (resid - A * f)**2)
    chi2_null = np.sum(w * (resid)**2)
    dchi2 = chi2_null - chi2_fit
    if HAVE_SCIPY:
        from scipy.stats import chi2 as chi2dist
        p = chi2dist.sf(dchi2, df=1)
    else:
        p = np.nan
    A_err = 1.0/np.sqrt(S2)
    return dict(A=A, A_err=A_err, chi2=chi2_fit, chi2_null=chi2_null, dchi2=dchi2, p=p, N=len(resid))

def plot_A_projection(resid_in, z_in, resid_out, z_out, J_kind, path):
    f_in  = J_factor(z_in,  J_kind) * np.sqrt(np.clip(z_in,  0, None))
    f_out = J_factor(z_out, J_kind) * np.sqrt(np.clip(z_out, 0, None))
    plt.figure(figsize=(7,5))
    plt.scatter(f_out, resid_out, s=6, alpha=0.35, label="outside")
    plt.scatter(f_in,  resid_in,  s=8, alpha=0.80, label="inside")
    # unweighted line fits for visualization
    def lineA(f, r):
        if f.size<2 or np.allclose(np.dot(f,f),0): return 0.0
        return float(np.dot(f, r)/np.dot(f, f))
    Ain, Aout = lineA(f_in, resid_in), lineA(f_out, resid_out)
    X = np.linspace(0, max(f_in.max() if f_in.size else 0, f_out.max() if f_out.size else 0), 200)
    plt.plot(X, Aout*X, color='C0', lw=2, alpha=0.8)
    plt.plot(X, Ain *X, color='C1', lw=2, alpha=0.8)
    plt.xlabel(f"f(z) = √z · J_{J_kind}(z)"); plt.ylabel("residual μ (mag)")
    plt.title("Projection onto f(z) = J·√z")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()

# ---------- plotting wrappers ----------
def save_scatter(x_in, y_in, x_out, y_out, xlabel, ylabel, title, path):
    plt.figure(figsize=(7,5))
    plt.scatter(x_out, y_out, s=6, alpha=0.35, label="outside")
    plt.scatter(x_in,  y_in,  s=8, alpha=0.8,  label="inside")
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()

def save_density(be, cnt_in, cnt_out, labelx, title, path):
    plt.figure(figsize=(7,4))
    plt.step(be[:-1], cnt_out, where='post', label='outside')
    plt.step(be[:-1], cnt_in,  where='post', label='inside')
    plt.xlabel(labelx); plt.ylabel("count per bin"); plt.title(title)
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()

def save_meanvar(bc, mu_in, var_in, mu_out, var_out, xlab, path_prefix):
    plt.figure(figsize=(7,4))
    plt.plot(bc, mu_out, marker='o', ms=3, lw=1, label='outside')
    plt.plot(bc, mu_in,  marker='o', ms=3, lw=1, label='inside')
    plt.xlabel(xlab); plt.ylabel("mean residual (mag)")
    plt.title("Mean residual per bin")
    plt.legend(); plt.tight_layout(); plt.savefig(path_prefix+"_mean.png", dpi=160); plt.close()

    plt.figure(figsize=(7,4))
    plt.plot(bc, var_out, marker='s', ms=3, lw=1, label='outside')
    plt.plot(bc, var_in,  marker='s', ms=3, lw=1, label='inside')
    plt.xlabel(xlab); plt.ylabel("variance (mag^2)")
    plt.title("Residual variance per bin")
    plt.legend(); plt.tight_layout(); plt.savefig(path_prefix+"_var.png", dpi=160); plt.close()

def save_acf_lb(resid_in, resid_out, nlags, path):
    ac_in  = acf_1d(resid_in, nlags)
    ac_out = acf_1d(resid_out, nlags)
    n_in, n_out = len(resid_in), len(resid_out)
    p_in  = ljung_box_pvalues(ac_in,  n_in)
    p_out = ljung_box_pvalues(ac_out, n_out)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    lags = np.arange(nlags + 1)
    ax[0].stem(lags, ac_out, label="outside", basefmt=" ")
    ax[0].stem(lags + 0.1, ac_in, label="inside", basefmt=" ")
    ax[0].axhline(0, color='k', lw=0.7, alpha=0.6)
    ax[0].set_xlabel("lag h"); ax[0].set_ylabel("ACF"); ax[0].set_title("Autocorrelation")
    ax[0].legend()

    if p_in.size > 0 and p_out.size > 0:
        ax[1].plot(np.arange(1, nlags + 1), -np.log10(p_out), marker='o', ms=3, lw=1, label="outside")
        ax[1].plot(np.arange(1, nlags + 1), -np.log10(p_in),  marker='o', ms=3, lw=1, label="inside")
        ax[1].set_ylabel("-log10 p(Ljung-Box)")
    else:
        ax[1].plot([], [])
        ax[1].text(0.5, 0.5, "SciPy not found — p-values skipped",
                   ha='center', va='center', transform=ax[1].transAxes)
    ax[1].set_xlabel("lag h"); ax[1].set_title("Ljung–Box whiteness"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)

# ---------- linear fit ----------
def linfit(y, x, w=None):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    W = (np.ones_like(x) if w is None else np.asarray(w, float)[m])
    if x.size<2: return np.nan, np.nan, np.nan, np.nan
    X = np.vstack([x, np.ones_like(x)]).T
    WX = X*W[:,None]
    beta, *_ = np.linalg.lstsq(WX, y*W, rcond=None)
    slope, intercept = beta[0], beta[1]
    yhat = X@beta; dof = max(1, x.size-2)
    sigma2 = np.sum(W*(y-yhat)**2)/dof
    cov = sigma2 * np.linalg.inv(WX.T@WX)
    stderr = np.sqrt(cov[0,0])
    if HAVE_SCIPY:
        from scipy.stats import t as tdist
        tval = slope/max(stderr,1e-12)
        p = 2*tdist.sf(abs(tval), df=dof)
    else:
        p = np.nan
    return slope, intercept, stderr, p

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default='Pantheon+SH0ES.dat')
    ap.add_argument('--prefer', choices=['zhd','zcmb'], default='zhd')
    ap.add_argument('--mask-frame', choices=['equ','gal'], default='gal')
    ap.add_argument('--center-ra', type=float, default=None)
    ap.add_argument('--center-dec', type=float, default=None)
    ap.add_argument('--center-l', type=float, default=209.0)
    ap.add_argument('--center-b', type=float, default=-57.0)
    ap.add_argument('--radius', type=float, default=20.0, help='deg')
    ap.add_argument('--nbins', type=int, default=36)
    ap.add_argument('--nlags', type=int, default=20)
    ap.add_argument('--H0', type=float, default=70.0)
    ap.add_argument('--Om', type=float, default=0.3)
    ap.add_argument('--zmin', type=float, default=0.03)
    ap.add_argument('--zmax', type=float, default=0.8)
    ap.add_argument('--A-J', dest='A_J', choices=['none','1pz','sec'], default='1pz',
                    help="J-factor for A-projection: none | 1pz (default) | sec (=sec ζ, clipped)")
    ap.add_argument('--outdir', type=str, default='eridan_scan')
    args, _ = ap.parse_known_args()

    os.makedirs(args.outdir, exist_ok=True)

    sn = load_sn(args.data, prefer=args.prefer)
    mrange = (sn['z']>=args.zmin) & (sn['z']<=args.zmax)
    sn = sn[mrange].reset_index(drop=True)

    mu_model = mu_of_z(sn['z'].values, H0=args.H0, Om=args.Om)
    resid = sn['mu'].values - mu_model
    zeta  = z_to_zeta(sn['z'].values)

    # Build mask
    if args.mask_frame == 'gal':
        l, b = radec_to_gal(sn['RA'].values, sn['DEC'].values)
        sep = angsep_deg(l, b, args.center_l, args.center_b)
    else:
        if args.center_ra is None or args.center_dec is None:
            print("Укажи --center-ra и --center-dec для mask-frame=equ", file=sys.stderr); sys.exit(1)
        sep = angsep_deg(sn['RA'].values, sn['DEC'].values, args.center_ra, args.center_dec)
    inside = sep <= args.radius
    outside= ~inside

    # Sorted-by-ζ for ACF
    order = np.argsort(zeta)
    r_in_seq   = resid[order][inside[order]]
    r_out_seq  = resid[order][outside[order]]

    # Figures: scatter & densities
    save_scatter(zeta[inside],  resid[inside], zeta[outside], resid[outside],
                 "zeta (rad)", "mu_obs - mu_LCDM (mag)",
                 "Residuals vs zeta: inside circle vs outside",
                 os.path.join(args.outdir, "scatter_resid_vs_zeta_inside_vs_outside.png"))
    save_scatter(sn['z'].values[inside],  resid[inside], sn['z'].values[outside], resid[outside],
                 "z", "mu_obs - mu_LCDM (mag)",
                 "Residuals vs z: inside circle vs outside",
                 os.path.join(args.outdir, "scatter_resid_vs_z_inside_vs_outside.png"))

    be, bc, cnt_in, mu_in, var_in, _  = bin_stats(zeta[inside],  resid[inside], args.nbins)
    _,  _, cnt_out, mu_out, var_out, _ = bin_stats(zeta[outside], resid[outside], args.nbins)
    save_density(be, cnt_in, cnt_out, "zeta (rad)", "Counts per zeta bin",
                 os.path.join(args.outdir, "zeta_density_inside_vs_outside.png"))
    save_meanvar(bc, mu_in, var_in, mu_out, var_out, "zeta (rad)",
                 os.path.join(args.outdir, "zeta"))
    save_acf_lb(r_in_seq, r_out_seq, args.nlags,
                os.path.join(args.outdir, "zeta_acf_ljungbox_inside_vs_outside.png"))

    be_z, bc_z, cnt_in_z, mu_in_z, var_in_z, _ = bin_stats(sn['z'].values[inside],  resid[inside], args.nbins)
    _,    _,    cnt_out_z,mu_out_z,var_out_z,_ = bin_stats(sn['z'].values[outside], resid[outside], args.nbins)
    save_density(be_z, cnt_in_z, cnt_out_z, "z", "Counts per z bin",
                 os.path.join(args.outdir, "z_density_inside_vs_outside.png"))
    save_meanvar(bc_z, mu_in_z, var_in_z, mu_out_z, var_out_z, "z",
                 os.path.join(args.outdir, "z"))
    ordz = np.argsort(sn['z'].values)
    save_acf_lb(resid[ordz][inside[ordz]], resid[ordz][outside[ordz]], args.nlags,
                os.path.join(args.outdir, "z_acf_ljungbox_inside_vs_outside.png"))

    # ---------- NEW: A-projection inside vs outside ----------
    A_in  = estimate_A(resid[inside],  sn['sig'].values[inside],  sn['z'].values[inside],  J_kind=args.A_J)
    A_out = estimate_A(resid[outside], sn['sig'].values[outside], sn['z'].values[outside], J_kind=args.A_J)
    plot_A_projection(resid[inside], sn['z'].values[inside],
                      resid[outside], sn['z'].values[outside],
                      args.A_J, os.path.join(args.outdir, f"A_projection_{args.A_J}.png"))
    pd.DataFrame([dict(region='inside',  **A_in),
                  dict(region='outside', **A_out)]).to_csv(
        os.path.join(args.outdir, f"A_fit_summary_{args.A_J}.csv"), index=False)

    print(f"\nA-fit with f(z)=√z·J_{args.A_J}(z):")
    for tag, Ares in [('inside', A_in), ('outside', A_out)]:
        A, Ae, dchi2, p = Ares['A'], Ares['A_err'], Ares['dchi2'], Ares['p']
        if np.isfinite(A) and np.isfinite(Ae):
            sig = A/Ae
            print(f"  {tag:7s}: A = {A:+.5f} ± {Ae:.5f} mag  (S/N={sig:.2f}),  Δχ²={dchi2:.2f},  p={p if p==p else 'NA'}")
        else:
            print(f"  {tag:7s}: not enough data")

    print("\nSaved figures & CSV in:", args.outdir)

if __name__ == "__main__":
    main()
