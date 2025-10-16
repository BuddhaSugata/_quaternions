#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan_beam_A.py — пройтись "лучом" и найти центр с максимально значимым A.

• Луч (beam) задаётся режимом:
    --beam gal_lon   (l от 0..360 при фиксированном b0)
    --beam gal_lat   (b от -90..+90 при фиксированном l0)
    --beam equ_ra    (RA от 0..360 при фиксированном Dec0)
    --beam equ_dec   (Dec от -90..+90 при фиксированном RA0)

• В каждой точке луча берётся круг радиуса --radius (deg), внутри которого
  оценивается A в модели resid = c + A * [ sqrt(z) * J(z) ], J ∈ {none, 1/(1+z), secζ}.

• Для устойчивости остатки считаются после глобальной калибровки (по умолчанию)
  ΛCDM: подгоняем Ωm по сетке и M (аналитически).

Выход:
  - CSV: results_beam_scan.csv (одна строка на центр луча)
  - PNG: beam_SNR_vs_coord.png, beam_Dchi2_vs_coord.png
  - Текстовый summary с топ-k

Примеры:
  python scan_beam_A.py --data Pantheon+SH0ES.dat --prefer zhd \
    --beam gal_lon --b0 -57 --radius 15 --step 2 \
    --zmin 0.03 --zmax 0.8 --A-J 1pz --outdir beam_scan_l_b-57_r15

  python scan_beam_A.py --data Pantheon+SH0ES.dat --prefer zhd \
    --beam equ_dec --ra0 50 --radius 15 --step 2 --A-J sec --outdir beam_equ_dec
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

# ---------- numerics ----------
def trapz_safe(y, x):
    if hasattr(np, "trapezoid"): return np.trapezoid(y, x)
    return np.trapz(y, x)

# ---------- coords ----------
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

# ---------- cosmology ----------
c_km_s = 299792.458

def z_to_zeta(z):
    z = np.clip(z, 0, None)
    return 2.0*np.arctan(np.sqrt(z))

def E_z(z, Om, Ol): return np.sqrt(Om*(1+z)**3 + Ol)

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

# ---------- IO ----------
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
    return out

# ---------- LCDM calibration ----------
def fit_lcdm(sn, H0=70.0, OM_GRID=np.linspace(0.05,0.6,241)):
    z = sn["z"].values; y = sn["mu"].values; w = 1.0/np.maximum(sn["sig"].values, 1e-9)**2
    best = None
    for Om in OM_GRID:
        mu0 = mu_of_z(z, H0=H0, Om=Om)  # with M=0
        M = np.sum(w*(y-mu0))/np.sum(w) # optimal intercept
        r = y - (mu0 + M)
        chi2 = np.sum(w*r*r)
        if (best is None) or (chi2 < best["chi2"]):
            best = dict(Om=float(Om), M=float(M), chi2=float(chi2), resid=r, mu_model=(mu0+M))
    N=len(z); k=2
    best["dof"]=int(N-k); best["chi2_red"]=best["chi2"]/max(1,best["dof"])
    return best

# ---------- A-projection ----------
def J_factor(z, kind='1pz'):
    z = np.asarray(z, float)
    if kind == 'none': return np.ones_like(z)
    if kind == '1pz':  return 1.0/(1.0+z)
    if kind == 'sec':
        zeta = z_to_zeta(z); val = 1.0/np.cos(zeta)
        return np.clip(val, 1.0, 10.0)  # guard near z~1
    return np.ones_like(z)

def estimate_A(resid, sigma, z, J_kind='1pz'):
    """
    Взвешенная регрессия: resid = c + A * f(z),  f = sqrt(z) * J(z).
    Используем корректную постановку с sqrt(w):
        Xw = X * sqrt(w), yw = y * sqrt(w)
    Ковариация параметров: Cov = (X^T W X)^{-1} = (Xw^T Xw)^{-1}  (без sigma2).
    Возвращает A, A_err, c, c_err, chi2_full, chi2_null(c-only), dchi2, p, N.
    """
    z = np.asarray(z, float)
    f = J_factor(z, J_kind) * np.sqrt(np.clip(z, 0, None))
    y = np.asarray(resid, float)
    w = 1.0 / np.maximum(np.asarray(sigma, float), 1e-12)**2

    m = np.isfinite(f) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    f, y, w = f[m], y[m], w[m]
    N = len(y)
    if N < 3:
        return dict(A=np.nan, A_err=np.nan, c=np.nan, c_err=np.nan,
                    chi2=np.nan, chi2_null=np.nan, dchi2=np.nan, p=np.nan, N=N)

    # Дизайн: y = c + A f
    X = np.vstack([np.ones_like(f), f]).T
    sqrtw = np.sqrt(w)
    Xw = X * sqrtw[:, None]
    yw = y * sqrtw

    # Взвешенная ОЛС через lstsq на (√w X, √w y)
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)  # [c, A]
    c, A = float(beta[0]), float(beta[1])

    # χ² полной модели и нулевой по A (с c свободным)
    yhat = X @ beta
    chi2_full = float(np.sum(w * (y - yhat)**2))
    c0 = float(np.sum(w * y) / np.sum(w))
    chi2_null = float(np.sum(w * (y - c0)**2))
    dchi2 = chi2_null - chi2_full

    # Ковариация параметров при известных σ_i: Cov = (X^T W X)^(-1) = (Xw^T Xw)^(-1)
    XTWX = Xw.T @ Xw
    cov = np.linalg.inv(XTWX)
    c_err = float(np.sqrt(max(cov[0, 0], 0.0)))
    A_err = float(np.sqrt(max(cov[1, 1], 0.0)))

    if HAVE_SCIPY:
        from scipy.stats import chi2 as chi2dist
        p = float(chi2dist.sf(dchi2, df=1))
    else:
        p = np.nan

    return dict(A=A, A_err=A_err, c=c, c_err=c_err,
                chi2=chi2_full, chi2_null=chi2_null, dchi2=dchi2, p=p, N=N)


# ---------- plotting ----------
def plot_metric_vs_coord(coord, snr, dchi2, xlabel, outdir):
    x = np.asarray(coord)
    plt.figure(figsize=(8,4))
    plt.plot(x, snr, '-o', ms=3)
    plt.xlabel(xlabel); plt.ylabel('S/N of A'); plt.title('A significance along beam')
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "beam_SNR_vs_coord.png"), dpi=160); plt.close()
    plt.figure(figsize=(8,4))
    plt.plot(x, dchi2, '-o', ms=3)
    plt.xlabel(xlabel); plt.ylabel('Δχ² (vs A=0, c free)'); plt.title('Δχ² along beam')
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "beam_Dchi2_vs_coord.png"), dpi=160); plt.close()

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default='Pantheon+SH0ES.dat')
    ap.add_argument('--prefer', choices=['zhd','zcmb'], default='zhd')
    ap.add_argument('--H0', type=float, default=70.0)
    ap.add_argument('--Om_grid_min', type=float, default=0.05)
    ap.add_argument('--Om_grid_max', type=float, default=0.60)
    ap.add_argument('--Om_grid_N',   type=int,   default=241)

    ap.add_argument('--beam', choices=['gal_lon','gal_lat','equ_ra','equ_dec'], default='gal_lon')
    ap.add_argument('--l0', type=float, default=0.0, help='fixed galactic longitude for gal_lat beam')
    ap.add_argument('--b0', type=float, default=0.0, help='fixed galactic latitude  for gal_lon beam')
    ap.add_argument('--ra0', type=float, default=0.0, help='fixed RA (deg) for equ_dec beam')
    ap.add_argument('--dec0', type=float, default=0.0, help='fixed Dec (deg) for equ_ra beam')

    ap.add_argument('--start', type=float, default=None, help='beam start coordinate (deg)')
    ap.add_argument('--end',   type=float, default=None, help='beam end coordinate (deg)')
    ap.add_argument('--step',  type=float, default=2.0,  help='step in degrees along beam')

    ap.add_argument('--radius', type=float, default=15.0, help='circle radius (deg)')
    ap.add_argument('--A-J', dest='A_J', choices=['none','1pz','sec'], default='1pz')

    ap.add_argument('--zmin', type=float, default=0.03)
    ap.add_argument('--zmax', type=float, default=0.8)
    ap.add_argument('--minN', type=int,   default=25, help='min SNe inside circle to accept point')
    ap.add_argument('--outdir', type=str, default='beam_scan_out')
    ap.add_argument('--topk', type=int, default=10)
    args, _ = ap.parse_known_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load & filter
    sn = load_sn(args.data, prefer=args.prefer)
    sn = sn[(sn['z']>=args.zmin) & (sn['z']<=args.zmax)].reset_index(drop=True)

    # Global ΛCDM calibration
    OM_GRID = np.linspace(args.Om_grid_min, args.Om_grid_max, max(3, args.Om_grid_N))
    best = fit_lcdm(sn, H0=args.H0, OM_GRID=OM_GRID)
    mu_model = best['mu_model']
    resid = sn['mu'].values - mu_model
    print(f"Global LCDM fit: Ωm={best['Om']:.3f}, M={best['M']:+.3f}, χ²_red={best['chi2_red']:.3f}")

    # Precompute coords
    l, b = radec_to_gal(sn['RA'].values, sn['DEC'].values)
    RA = sn['RA'].values; DEC = sn['DEC'].values

    # Build beam coordinates
    if args.beam == 'gal_lon':
        b0 = args.b0
        start = 0.0 if args.start is None else args.start
        end   = 360.0 if args.end is None else args.end
        coords = np.arange(start, end+1e-9, args.step)  # l along beam
        centers = [(float(L%360.0), float(b0)) for L in coords]
        xlabel = f"galactic longitude l (deg) @ b={b0:.1f}°"
        def sep_mask(c):
            Lc, Bc = c; return angsep_deg(l, b, Lc, Bc) <= args.radius

    elif args.beam == 'gal_lat':
        l0 = args.l0 % 360.0
        start = -90.0 if args.start is None else args.start
        end   = +90.0 if args.end is None else args.end
        coords = np.arange(start, end+1e-9, args.step)  # b along beam
        centers = [(float(l0), float(B)) for B in coords]
        xlabel = f"galactic latitude b (deg) @ l={l0:.1f}°"
        def sep_mask(c):
            Lc, Bc = c; return angsep_deg(l, b, Lc, Bc) <= args.radius

    elif args.beam == 'equ_ra':
        dec0 = args.dec0
        start = 0.0 if args.start is None else args.start
        end   = 360.0 if args.end is None else args.end
        coords = np.arange(start, end+1e-9, args.step)  # RA along beam
        centers = [(float(R%360.0), float(dec0)) for R in coords]
        xlabel = f"RA (deg) @ Dec={dec0:.1f}°"
        def sep_mask(c):
            RAc, DECc = c; return angsep_deg(RA, DEC, RAc, DECc) <= args.radius

    else:  # equ_dec
        ra0 = args.ra0 % 360.0
        start = -90.0 if args.start is None else args.start
        end   = +90.0 if args.end is None else args.end
        coords = np.arange(start, end+1e-9, args.step)  # Dec along beam
        centers = [(float(ra0), float(DD)) for DD in coords]
        xlabel = f"Dec (deg) @ RA={ra0:.1f}°"
        def sep_mask(c):
            RAc, DECc = c; return angsep_deg(RA, DEC, RAc, DECc) <= args.radius

    # Scan
    rows = []
    for k, c in enumerate(centers):
        mask = sep_mask(c)
        Ni = int(mask.sum())
        if Ni < args.minN:
            rows.append(dict(coord=coords[k], N=Ni, A=np.nan, A_err=np.nan,
                             SNR=np.nan, dchi2=np.nan, p=np.nan,
                             center_1=c[0], center_2=c[1]))
            continue
        Ares = estimate_A(resid[mask], sn['sig'].values[mask], sn['z'].values[mask], J_kind=args.A_J)
        A, Ae = Ares['A'], Ares['A_err']
        snr = (A/Ae) if (np.isfinite(A) and np.isfinite(Ae) and Ae>0) else np.nan
        rows.append(dict(coord=coords[k], N=Ni, A=A, A_err=Ae,
                         SNR=snr, dchi2=Ares['dchi2'], p=Ares['p'],
                         c=Ares['c'], c_err=Ares['c_err'],
                         chi2=Ares['chi2'], chi2_null=Ares['chi2_null'],
                         center_1=c[0], center_2=c[1]))

    df = pd.DataFrame(rows)
    # ranking
    dfs = df.copy()
    dfs['rank_snr'] = (-np.abs(dfs['SNR'])).argsort().argsort()+1
    dfs['rank_dchi2'] = (-dfs['dchi2'].fillna(-np.inf)).argsort().argsort()+1

    out_csv = os.path.join(args.outdir, "results_beam_scan.csv")
    dfs.to_csv(out_csv, index=False)
    print(f"Saved table: {out_csv}")

    # Plots
    plot_metric_vs_coord(dfs['coord'].values, dfs['SNR'].values, dfs['dchi2'].values, xlabel, args.outdir)

    # Summary top-k
    topk = max(1, int(args.topk))
    top_by_snr = dfs.sort_values(by='SNR', key=lambda s: np.abs(s), ascending=False).head(topk)
    top_by_dch = dfs.sort_values(by='dchi2', ascending=False).head(topk)

    with open(os.path.join(args.outdir, "best_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Global fit: Om={best['Om']:.4f}, M={best['M']:+.4f}, chi2_red={best['chi2_red']:.3f}\n")
        f.write(f"Beam mode: {args.beam}, radius={args.radius} deg, A-J={args.A_J}\n\n")
        f.write("Top by |S/N(A)|:\n")
        for _, r in top_by_snr.iterrows():
            f.write(f"  coord={r['coord']:.2f}  N={int(r['N'])}  A={r['A']:+.5f}±{r['A_err']:.5f}  "
                    f"S/N={r['SNR']:.2f}  Δχ²={r['dchi2']:.2f}  p={r['p']}\n")
        f.write("\nTop by Δχ²:\n")
        for _, r in top_by_dch.iterrows():
            f.write(f"  coord={r['coord']:.2f}  N={int(r['N'])}  A={r['A']:+.5f}±{r['A_err']:.5f}  "
                    f"S/N={r['SNR']:.2f}  Δχ²={r['dchi2']:.2f}  p={r['p']}\n")

    print("Top by |S/N|:")
    print(top_by_snr[['coord','N','A','A_err','SNR','dchi2','p']].to_string(index=False))
    print("\nTop by Δχ²:")
    print(top_by_dch[['coord','N','A','A_err','SNR','dchi2','p']].to_string(index=False))
    print("Saved plots and summary in:", args.outdir)

if __name__ == "__main__":
    main()
