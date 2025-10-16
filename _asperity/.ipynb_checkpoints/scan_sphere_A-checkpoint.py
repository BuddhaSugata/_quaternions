#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan_sphere_A.py — полный 2D-скан по сфере для поиска максимальной значимости A.

• Базовая модель: глобально подгоняем ΛCDM (Ωm по сетке + M аналитически).
• Для каждого центра на сфере (HEALPix / Фибоначчи / lon-lat сетка) берём круг радиуса R
  и оцениваем A в модели:
      resid(μ) = c + A * f(z),   f(z) = sqrt(z) * J(z),   J ∈ {none, 1/(1+z), sec(ζ)}.
  Весa из σ_μ: корректная постановка с sqrt(w). Выводим A, σ_A, Δχ², p, SNR_lr=√Δχ².

Выход:
  - results_sphere_scan.csv  (центр, N, A±σ, SNR_wald, SNR_lr, Δχ², p, ...)
  - maps: sphere_SNR.png, sphere_A.png, sphere_N.png  (Mollweide)
  - best_summary.txt  (топ-пики по SNR_lr и Δχ²)

Примеры:
  python scan_sphere_A.py --data Pantheon+SH0ES.dat --prefer zhd \
    --frame gal --grid healpix --nside 16 --radius 15 --A-J 1pz \
    --zmin 0.03 --zmax 0.8 --outdir sphere_hp16_r15

  python scan_sphere_A.py --data Pantheon+SH0ES.dat --prefer zhd \
    --frame gal --grid fibonacci --npix 3072 --radius 15 --A-J 1pz \
    --outdir sphere_fibo3072_r15
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

# ---------- optional HEALPix ----------
HAVE_HEALPY = False
try:
    import healpy as hp
    HAVE_HEALPY = True
except Exception:
    HAVE_HEALPY = False

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

def sph_to_cart(lon_deg, lat_deg):
    L = np.deg2rad(lon_deg); B = np.deg2rad(lat_deg)
    cB = np.cos(B)
    return np.stack([cB*np.cos(L), cB*np.sin(L), np.sin(B)], axis=-1)

def cart_to_sph(xyz):
    x,y,z = xyz[...,0], xyz[...,1], xyz[...,2]
    B = np.arcsin(np.clip(z, -1.0, 1.0))
    L = np.arctan2(y, x) % (2*np.pi)
    return np.rad2deg(L), np.rad2deg(B)

# ---------- grids on the sphere ----------
def grid_healpix(nside, frame='gal'):
    """Return centers (lon,lat) and unit vectors for HEALPix centers."""
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix), nest=False)  # theta=colat, phi=lon
    lon = np.rad2deg(phi); lat = 90.0 - np.rad2deg(theta)
    xyz = sph_to_cart(lon, lat)
    return lon, lat, xyz

def grid_fibonacci(npix):
    """Near-uniform Fibonacci sphere grid, returns lon, lat, xyz."""
    # Use 'step' golden angle; center points equally on z
    i = np.arange(npix, dtype=float)
    z = 1 - 2*(i+0.5)/npix
    r = np.sqrt(np.maximum(0.0, 1 - z*z))
    phi = (np.pi * (1 + 5**0.5)) * i  # golden angle
    x = r*np.cos(phi); y = r*np.sin(phi)
    lon = (np.rad2deg(np.arctan2(y, x)) % 360.0)
    lat = np.rad2deg(np.arcsin(z))
    xyz = np.stack([x, y, z], axis=1)
    return lon, lat, xyz

def grid_lonlat(step_lon=5.0, step_lat=5.0):
    lons = np.arange(0.0, 360.0, step_lon)
    lats = np.arange(-90.0, 90.0+1e-9, step_lat)
    LL, BB = np.meshgrid(lons, lats)
    lon = LL.ravel(); lat = BB.ravel()
    xyz = sph_to_cart(lon, lat)
    return lon, lat, xyz

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
    # choose z
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
        mu0 = mu_of_z(z, H0=H0, Om=Om)  # M=0
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
        return np.clip(val, 1.0, 10.0)
    return np.ones_like(z)

def estimate_A(resid, sigma, z, J_kind='1pz'):
    """
    Взвешенная регрессия: resid = c + A * f(z),  f = sqrt(z) * J(z).
    Корректные веса через sqrt(w). Возвращает A, σ_A, c, σ_c, χ²_full, χ²_null (c-only), Δχ², p, N.
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

    X = np.vstack([np.ones_like(f), f]).T
    sqrtw = np.sqrt(w); Xw = X*sqrtw[:,None]; yw = y*sqrtw
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)  # [c, A]
    c, A = float(beta[0]), float(beta[1])

    yhat = X @ beta
    chi2_full = float(np.sum(w * (y - yhat)**2))
    c0 = float(np.sum(w * y) / np.sum(w))
    chi2_null = float(np.sum(w * (y - c0)**2))
    dchi2 = chi2_null - chi2_full

    cov = np.linalg.inv(Xw.T @ Xw)  # при известных σ_i
    c_err = float(np.sqrt(max(cov[0,0],0.0)))
    A_err = float(np.sqrt(max(cov[1,1],0.0)))

    if HAVE_SCIPY:
        from scipy.stats import chi2 as chi2dist
        p = float(chi2dist.sf(dchi2, df=1))
    else:
        p = np.nan

    return dict(A=A, A_err=A_err, c=c, c_err=c_err,
                chi2=chi2_full, chi2_null=chi2_null, dchi2=dchi2, p=p, N=N)

# ---------- plotting ----------
def mollweide_scatter(lon_deg, lat_deg, val, title, fname, vcenter=None, cmap='coolwarm'):
    """Simple Mollweide scatter (matplotlib)."""
    # shift lon to [-180,180] and to radians
    L = (np.asarray(lon_deg) - 180.0) * np.pi/180.0
    B = np.asarray(lat_deg) * np.pi/180.0
    V = np.asarray(val, float)

    fig = plt.figure(figsize=(9,5))
    ax = fig.add_subplot(111, projection='mollweide')
    # symmetric vlim around 0 by default
    if vcenter == 'sym':
        vmax = np.nanpercentile(np.abs(V), 98)
        vmin, vmax = -vmax, +vmax
    else:
        vmin = np.nanpercentile(V, 2)
        vmax = np.nanpercentile(V, 98)
    sc = ax.scatter(L, B, c=V, s=8, alpha=0.9, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.grid(True, lw=0.5, alpha=0.4)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.08)
    cb.ax.set_ylabel(title, rotation=90)
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(fname, dpi=160); plt.close(fig)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default='Pantheon+SH0ES.dat')
    ap.add_argument('--prefer', choices=['zhd','zcmb'], default='zhd')
    ap.add_argument('--frame', choices=['gal','equ'], default='gal', help='в какой системе строить сетку/круги')
    ap.add_argument('--H0', type=float, default=70.0)
    ap.add_argument('--Om_grid_min', type=float, default=0.05)
    ap.add_argument('--Om_grid_max', type=float, default=0.60)
    ap.add_argument('--Om_grid_N',   type=int,   default=241)

    ap.add_argument('--grid', choices=['healpix','fibonacci','lonlat'], default='healpix')
    ap.add_argument('--nside', type=int, default=16, help='для HEALPix (npix=12*nside^2)')
    ap.add_argument('--npix',  type=int, default=3072, help='для Fibonacci grid')
    ap.add_argument('--step-lon', type=float, default=5.0, help='для lonlat-сетки')
    ap.add_argument('--step-lat', type=float, default=5.0, help='для lonlat-сетки')

    ap.add_argument('--radius', type=float, default=15.0, help='deg')
    ap.add_argument('--A-J', dest='A_J', choices=['none','1pz','sec'], default='1pz')

    ap.add_argument('--zmin', type=float, default=0.03)
    ap.add_argument('--zmax', type=float, default=0.8)
    ap.add_argument('--minN', type=int,   default=25)

    ap.add_argument('--outdir', type=str, default='sphere_scan_out')
    ap.add_argument('--topk', type=int, default=15)
    args, _ = ap.parse_known_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load & z-cut
    sn = load_sn(args.data, prefer=args.prefer)
    zcut = (sn['z']>=args.zmin) & (sn['z']<=args.zmax)
    sn = sn[zcut].reset_index(drop=True)

    # Global LCDM calibration
    OM_GRID = np.linspace(args.Om_grid_min, args.Om_grid_max, max(3, args.Om_grid_N))
    best = fit_lcdm(sn, H0=args.H0, OM_GRID=OM_GRID)
    mu_model = best['mu_model']
    resid = sn['mu'].values - mu_model
    print(f"Global LCDM fit: Ωm={best['Om']:.4f}, M={best['M']:+.4f}, chi2_red={best['chi2_red']:.3f}")

    # Choose frame and get source unit vectors
    if args.frame == 'gal':
        lon_sn, lat_sn = radec_to_gal(sn['RA'].values, sn['DEC'].values)
    else:
        lon_sn, lat_sn = sn['RA'].values, sn['DEC'].values
    xyz_sn = sph_to_cart(lon_sn, lat_sn)   # shape (N,3)
    cosR = np.cos(np.deg2rad(args.radius))

    # Build centers
    if args.grid == 'healpix':
        if not HAVE_HEALPY:
            print("healpy не найден — переключаюсь на Fibonacci grid", file=sys.stderr)
            lon_c, lat_c, xyz_c = grid_fibonacci(args.npix)
        else:
            lon_c, lat_c, xyz_c = grid_healpix(args.nside)
    elif args.grid == 'fibonacci':
        lon_c, lat_c, xyz_c = grid_fibonacci(args.npix)
    else:
        lon_c, lat_c, xyz_c = grid_lonlat(args.step_lon, args.step_lat)

    # Scan all centers
    rows = []
    z = sn['z'].values; sig = sn['sig'].values
    for i in range(xyz_c.shape[0]):
        c = xyz_c[i]
        # inside cap via dot product: dot >= cosR
        dots = xyz_sn @ c
        mask = dots >= cosR
        Ni = int(mask.sum())
        if Ni < args.minN:
            rows.append(dict(lon=lon_c[i], lat=lat_c[i], N=Ni, A=np.nan, A_err=np.nan,
                             SNR_wald=np.nan, SNR_lr=np.nan, dchi2=np.nan, p=np.nan,
                             c0=np.nan, c0_err=np.nan))
            continue
        Ares = estimate_A(resid[mask], sig[mask], z[mask], J_kind=args.A_J)
        A, Ae = Ares['A'], Ares['A_err']
        dchi2 = Ares['dchi2']
        snr_wald = (A/Ae) if (np.isfinite(A) and np.isfinite(Ae) and Ae>0) else np.nan
        snr_lr = np.sqrt(dchi2) if np.isfinite(dchi2) and dchi2>=0 else np.nan
        rows.append(dict(lon=lon_c[i], lat=lat_c[i], N=Ni, A=A, A_err=Ae,
                         SNR_wald=snr_wald, SNR_lr=snr_lr, dchi2=dchi2, p=Ares['p'],
                         c0=Ares['c'], c0_err=Ares['c_err']))

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.outdir, "results_sphere_scan.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved table: {out_csv}")

    # Top-k by SNR_lr and by Δχ²
    valid = df.dropna(subset=['SNR_lr'])
    topk = max(1, int(args.topk))
    top_by_snr = valid.reindex(valid['SNR_lr'].abs().sort_values(ascending=False).index).head(topk)
    top_by_dch = valid.sort_values(by='dchi2', ascending=False).head(topk)

    with open(os.path.join(args.outdir, "best_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Global fit: Om={best['Om']:.4f}, M={best['M']:+.4f}, chi2_red={best['chi2_red']:.3f}\n")
        f.write(f"Frame={args.frame}, grid={args.grid}, R={args.radius} deg, J={args.A_J}, z∈[{args.zmin},{args.zmax}], minN={args.minN}\n\n")
        f.write("Top by SNR_lr = sqrt(Δχ²):\n")
        for _, r in top_by_snr.iterrows():
            f.write(f"  center=({r['lon']:.2f},{r['lat']:.2f})  N={int(r['N'])}  "
                    f"A={r['A']:+.5f}±{r['A_err']:.5f}  SNR_lr={r['SNR_lr']:.2f}  "
                    f"Δχ²={r['dchi2']:.2f}  p={r['p']}\n")
        f.write("\nTop by Δχ²:\n")
        for _, r in top_by_dch.iterrows():
            f.write(f"  center=({r['lon']:.2f},{r['lat']:.2f})  N={int(r['N'])}  "
                    f"A={r['A']:+.5f}±{r['A_err']:.5f}  SNR_lr={r['SNR_lr']:.2f}  "
                    f"Δχ²={r['dchi2']:.2f}  p={r['p']}\n")

    print("Top by SNR_lr:")
    print(top_by_snr[['lon','lat','N','A','A_err','SNR_lr','dchi2','p']].to_string(index=False))
    print("\nTop by Δχ²:")
    print(top_by_dch[['lon','lat','N','A','A_err','SNR_lr','dchi2','p']].to_string(index=False))

    # Maps
    mollweide_scatter(df['lon'].values, df['lat'].values, df['SNR_lr'].values,
                      title=f"SNR_lr = sqrt(Δχ²), R={args.radius}°, frame={args.frame}",
                      fname=os.path.join(args.outdir, "sphere_SNR.png"),
                      vcenter='sym', cmap='coolwarm')
    mollweide_scatter(df['lon'].values, df['lat'].values, df['A'].values,
                      title=f"A (mag), R={args.radius}°, frame={args.frame}",
                      fname=os.path.join(args.outdir, "sphere_A.png"),
                      vcenter='sym', cmap='coolwarm')
    mollweide_scatter(df['lon'].values, df['lat'].values, df['N'].values,
                      title=f"N per cap (R={args.radius}°), frame={args.frame}",
                      fname=os.path.join(args.outdir, "sphere_N.png"),
                      vcenter=None, cmap='viridis')

    print("Saved maps and summary in:", args.outdir)

if __name__ == "__main__":
    main()
