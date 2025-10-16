#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan_sphere_deltaA.py — полный 2D-скан по сфере для ΔA = A_in - A_out
и множественных проверок (Bonferroni, FDR).

• Базовая модель: глобально подгоняем ΛCDM (Ωm по сетке + M аналитически).
• Для каждого центра на сфере (HEALPix / Фибоначчи / lon-lat) берём круг радиуса R:
      A_in:   resid = c + A * [ sqrt(z) * J(z) ]  внутри круга
      A_out:  то же для "снаружи" по режиму contrast:
               - complement: весь комплимент круга
               - global:     все SNe (без исключения)   [использовать аккуратно]
               - annulus:    кольцо R1..R2 (зададим --outside-r1 --outside-r2)
• ΔA = A_in - A_out,  σ_Δ = sqrt(σ_Ain^2 + σ_Aout^2),  SNR_Δ = ΔA/σ_Δ.
  p_local — двусторонняя нормальная, затем p_Bonf. и q_FDR по Benjamini–Hochberg.
• Выход: CSV с метриками, карты (Mollweide), список локальных максимумов.

Примеры:
  python scan_sphere_deltaA.py --data Pantheon+SH0ES.dat --prefer zhd \
    --frame gal --grid fibonacci --npix 3072 \
    --radius 15 --contrast complement --A-J 1pz \
    --zmin 0.03 --zmax 0.8 --minN 25 --minN-out 100 \
    --outdir sphere_deltaA_r15

  python scan_sphere_deltaA.py --data Pantheon+SH0ES.dat --prefer zhd \
    --frame gal --grid healpix --nside 16 \
    --radius 15 --contrast annulus --outside-r1 25 --outside-r2 60 \
    --A-J 1pz --outdir sphere_deltaA_annulus
"""
import argparse, os, sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- optional SciPy ----------
try:
    from scipy.stats import chi2, norm
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

# ---------- grids ----------
def grid_healpix(nside):
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix), nest=False)  # theta=colat, phi=lon
    lon = np.rad2deg(phi); lat = 90.0 - np.rad2deg(theta)
    xyz = sph_to_cart(lon, lat)
    return lon, lat, xyz

def grid_fibonacci(npix):
    i = np.arange(npix, dtype=float)
    z = 1 - 2*(i+0.5)/npix
    r = np.sqrt(np.maximum(0.0, 1 - z*z))
    phi = (np.pi * (1 + 5**0.5)) * i
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

# ---------- stats helpers ----------
def p_two_sided_from_z(z):
    """двусторонняя нормальная p из Z=SNR."""
    z = abs(float(z))
    if HAVE_SCIPY:
        return float(2*norm.sf(z))
    # без SciPy — через erfc
    return float(math.erfc(z / math.sqrt(2.0)))

def fdr_bh(pvals):
    """Benjamini–Hochberg q-values (FDR) для массива p (NaN → NaN)."""
    p = np.asarray(pvals, float)
    n = np.sum(np.isfinite(p))
    q = np.full_like(p, np.nan)
    if n == 0: return q
    idx = np.argsort(p, kind='mergesort')
    rank = np.empty_like(idx); rank[idx] = np.arange(1, len(p)+1)
    # только конечные
    finite_idx = idx[np.isfinite(p[idx])]
    m = len(finite_idx)
    if m==0: return q
    pi = p[finite_idx]
    ranks = rank[finite_idx].astype(float)
    qvals = pi * m / ranks
    # make monotone
    qvals = np.minimum.accumulate(qvals[::-1])[::-1]
    q[finite_idx] = np.minimum(qvals, 1.0)
    return q

# ---------- plotting ----------
def mollweide_scatter(lon_deg, lat_deg, val, title, fname, vcenter='sym', cmap='coolwarm'):
    # shift lon to [-180,180] and to radians
    L = (np.asarray(lon_deg) - 180.0) * np.pi/180.0
    B = np.asarray(lat_deg) * np.pi/180.0
    V = np.asarray(val, float)
    fig = plt.figure(figsize=(9,5))
    ax = fig.add_subplot(111, projection='mollweide')
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

# ---------- local maxima on sphere ----------
def estimate_spacing_deg(npts):
    # типовый угловой шаг ~ sqrt(4π/N) рад
    return float(np.sqrt(4*np.pi/npts) * 180/np.pi)

def local_maxima_spherical(lon, lat, metric, radius_deg):
    lon = np.asarray(lon, float); lat = np.asarray(lat, float); v = np.asarray(metric, float)
    xyz = sph_to_cart(lon, lat)
    cosR = np.cos(np.deg2rad(radius_deg))
    peaks = []
    N = len(lon)
    for i in range(N):
        if not np.isfinite(v[i]): continue
        dots = xyz @ xyz[i]
        neigh = dots >= cosR
        if not np.any(neigh): continue
        vi = abs(v[i]); vn = np.abs(v[neigh])
        if vi >= np.nanmax(vn) - 1e-12:
            peaks.append(i)
    return np.array(peaks, dtype=int)

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
    ap.add_argument('--nside', type=int, default=16)
    ap.add_argument('--npix',  type=int, default=3072)
    ap.add_argument('--step-lon', type=float, default=6.0)
    ap.add_argument('--step-lat', type=float, default=4.0)

    ap.add_argument('--radius', type=float, default=15.0, help='deg (inside cap)')
    ap.add_argument('--contrast', choices=['complement','global','annulus'], default='complement')
    ap.add_argument('--outside-r1', type=float, default=None, help='annulus inner radius (deg)')
    ap.add_argument('--outside-r2', type=float, default=None, help='annulus outer radius (deg)')

    ap.add_argument('--A-J', dest='A_J', choices=['none','1pz','sec'], default='1pz')
    ap.add_argument('--zmin', type=float, default=0.03)
    ap.add_argument('--zmax', type=float, default=0.8)
    ap.add_argument('--minN', type=int,   default=25,  help='min inside')
    ap.add_argument('--minN-out', type=int, default=100, help='min outside')

    ap.add_argument('--outdir', type=str, default='sphere_deltaA_out')
    ap.add_argument('--topk', type=int, default=15)
    args, _ = ap.parse_known_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load & z-cut
    sn = load_sn(args.data, prefer=args.prefer)
    zmask = (sn['z']>=args.zmin) & (sn['z']<=args.zmax)
    sn = sn[zmask].reset_index(drop=True)

    # Global LCDM calibration
    OM_GRID = np.linspace(args.Om_grid_min, args.Om_grid_max, max(3, args.Om_grid_N))
    best = fit_lcdm(sn, H0=args.H0, OM_GRID=OM_GRID)
    mu_model = best['mu_model']
    resid_all = sn['mu'].values - mu_model
    print(f"Global LCDM fit: Ωm={best['Om']:.4f}, M={best['M']:+.4f}, chi2_red={best['chi2_red']:.3f}")

    # Coords in chosen frame
    if args.frame == 'gal':
        lon_sn, lat_sn = radec_to_gal(sn['RA'].values, sn['DEC'].values)
    else:
        lon_sn, lat_sn = sn['RA'].values, sn['DEC'].values
    xyz_sn = sph_to_cart(lon_sn, lat_sn)
    z = sn['z'].values; sig = sn['sig'].values

    # Centers grid
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

    cosR_in = np.cos(np.deg2rad(args.radius))
    # annulus defaults if needed
    r1 = args.outside_r1 if args.outside_r1 is not None else (args.radius + 10.0)
    r2 = args.outside_r2 if args.outside_r2 is not None else max(r1+10.0, args.radius + 25.0)
    cosR1 = np.cos(np.deg2rad(r1)); cosR2 = np.cos(np.deg2rad(r2))

    rows = []
    for i in range(xyz_c.shape[0]):
        c = xyz_c[i]
        dots = xyz_sn @ c
        inside = dots >= cosR_in
        Nin = int(inside.sum())
        if Nin < args.minN:
            rows.append(dict(lon=lon_c[i], lat=lat_c[i], N_in=Nin, N_out=0,
                             A_in=np.nan, A_in_err=np.nan, A_out=np.nan, A_out_err=np.nan,
                             dA=np.nan, dA_err=np.nan, SNR_dA=np.nan,
                             p_local=np.nan, p_bonf=np.nan, q_fdr=np.nan))
            continue

        if args.contrast == 'complement':
            outside = ~inside
        elif args.contrast == 'global':
            outside = np.ones_like(inside, dtype=bool)
        else:  # annulus
            # in a spherical cap: cos angle between c and point is dots
            # annulus R1..R2 => angle in [R1, R2] => cos in [cosR2, cosR1]
            outside = (dots <= cosR1) & (dots >= cosR2)

        Nout = int(outside.sum())
        if Nout < args.minN_out:
            rows.append(dict(lon=lon_c[i], lat=lat_c[i], N_in=Nin, N_out=Nout,
                             A_in=np.nan, A_in_err=np.nan, A_out=np.nan, A_out_err=np.nan,
                             dA=np.nan, dA_err=np.nan, SNR_dA=np.nan,
                             p_local=np.nan, p_bonf=np.nan, q_fdr=np.nan))
            continue

        Ain = estimate_A(resid_all[inside],  sig[inside],  z[inside],  J_kind=args.A_J)
        Aout= estimate_A(resid_all[outside], sig[outside], z[outside], J_kind=args.A_J)

        dA = Ain['A'] - Aout['A']
        dA_err = math.sqrt(max(Ain['A_err']**2 + Aout['A_err']**2, 1e-18))
        snr_dA = dA / dA_err

        p_loc = p_two_sided_from_z(snr_dA)

        rows.append(dict(lon=lon_c[i], lat=lat_c[i],
                         N_in=Nin, N_out=Nout,
                         A_in=Ain['A'], A_in_err=Ain['A_err'],
                         A_out=Aout['A'], A_out_err=Aout['A_err'],
                         dA=dA, dA_err=dA_err, SNR_dA=snr_dA,
                         p_local=p_loc, p_bonf=np.nan, q_fdr=np.nan))

    df = pd.DataFrame(rows)
    # коррекции множественных проверок
    valid = df['p_local'].notna()
    M = int(valid.sum())
    df.loc[valid, 'p_bonf'] = np.minimum(1.0, df.loc[valid, 'p_local'] * M)
    df['q_fdr'] = fdr_bh(df['p_local'].values)

    out_csv = os.path.join(args.outdir, "results_sphere_contrast.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved table: {out_csv}")

    # Карты
    mollweide_scatter(df['lon'].values, df['lat'].values, df['dA'].values,
                      title=f"ΔA = A_in - A_out (mag), R={args.radius}°, contrast={args.contrast}",
                      fname=os.path.join(args.outdir, "sphere_dA.png"),
                      vcenter='sym', cmap='coolwarm')
    mollweide_scatter(df['lon'].values, df['lat'].values, df['SNR_dA'].values,
                      title=f"SNR_Δ = ΔA/σ_Δ, R={args.radius}°, contrast={args.contrast}",
                      fname=os.path.join(args.outdir, "sphere_SNR_delta.png"),
                      vcenter='sym', cmap='coolwarm')
    mollweide_scatter(df['lon'].values, df['lat'].values, df['N_in'].values,
                      title=f"N_in per cap (R={args.radius}°)",
                      fname=os.path.join(args.outdir, "sphere_Nin.png"),
                      vcenter=None, cmap='viridis')
    mollweide_scatter(df['lon'].values, df['lat'].values, df['N_out'].values,
                      title=f"N_out per outside set",
                      fname=os.path.join(args.outdir, "sphere_Nout.png"),
                      vcenter=None, cmap='viridis')

    # Локальные максимумы по |SNR_Δ|
    spacing = estimate_spacing_deg(len(df))
    nbhd = 1.25*spacing  # радиус окрестности для поиска пиков
    peaks_idx = local_maxima_spherical(df['lon'].values, df['lat'].values,
                                       df['SNR_dA'].values, radius_deg=nbhd)
    peaks = df.iloc[peaks_idx].copy()
    peaks = peaks.sort_values(by='SNR_dA', key=lambda s: np.abs(s), ascending=False)

    topk = max(1, int(args.topk))
    top_peaks = peaks.head(topk)

    # summary с поправками
    with open(os.path.join(args.outdir, "best_peaks_deltaA.txt"), "w", encoding="utf-8") as f:
        f.write(f"Global fit: Om={best['Om']:.4f}, M={best['M']:+.4f}, chi2_red={best['chi2_red']:.3f}\n")
        f.write(f"Frame={args.frame}, grid={args.grid}, R={args.radius}°, contrast={args.contrast}, J={args.A_J}\n")
        f.write(f"z∈[{args.zmin},{args.zmax}], minN_in={args.minN}, minN_out={args.minN_out}, tests M={M}\n\n")
        f.write("Top local peaks by |SNR_Δ| (with multiplicity corrections):\n")
        for _, r in top_peaks.iterrows():
            f.write(f"  center=({r['lon']:.2f},{r['lat']:.2f})  "
                    f"N_in={int(r['N_in'])}  N_out={int(r['N_out'])}  "
                    f"ΔA={r['dA']:+.5f}±{r['dA_err']:.5f}  "
                    f"SNR_Δ={r['SNR_dA']:.2f}  "
                    f"p_local={r['p_local']:.3g}  p_bonf={r['p_bonf']:.3g}  q_fdr={r['q_fdr']:.3g}\n")

    print("\nTop local peaks by |SNR_Δ|:")
    print(top_peaks[['lon','lat','N_in','N_out','dA','dA_err','SNR_dA','p_local','p_bonf','q_fdr']].to_string(index=False))
    print("Saved maps and summary in:", args.outdir)

if __name__ == "__main__":
    main()
