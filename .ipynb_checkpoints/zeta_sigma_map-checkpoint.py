#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zeta_sigma_map.py — оценка зернистости σ_ζ (рад) только по (RA,DEC,z),
без «истинного» z и без светимостей.

Идея:
  Для фиксированного направления (кэпс радиуса R) и набора ширин окна Δζ_k
  скользим по ζ, в каждом окне считаем выборочную дисперсию s_z^2, вычитаем
  медианную дисперсию измерений (если дана), нормируем на (dz/dζ)^2 в центре окна
  и берём медиану по окнам: Y_k = median_win[ (s_z^2 - σ_meas^2) / (dz/dζ)^2 ].
  Тогда Y_k ≈ σ_ζ^2 + (Δζ_k^2)/12. Линейная регрессия Y_k = a + b*(Δζ_k^2)
  даёт оценку a≈σ_ζ^2 и b≈1/12. Перехват → σ_ζ, наклон → sanity check.

Вход: таблица с колонками RA, DEC, Z (и опц. ZERR).
Выход:
  • results_sigma_zeta.csv (на кэпс): lon,lat,Ncap, sigma_zeta_rad, sigma_zeta_deg,
    sigma2_zeta, sigma2_zeta_err, slope_b, slope_b_expected(=1/12), R2, ...
  • Карты Mollweide: sphere_sigma_zeta_deg.png, sphere_slope_b.png, sphere_Ncap.png
  • (опц.) диагностические графики Y(Δζ^2) для нескольких кэпсов.

Пример:
  python zeta_sigma_map.py --cat DESI_DR1_like.tsv --cols RA DEC Z \
    --frame gal --grid fibonacci --npix 4096 --radius 15 \
    --zmin 0.02 --zmax 1.0 --dzetas 0.06 0.08 0.10 0.12 --step 0.02 \
    --minNcap 1000 --minNwin 120 --subtract-zerr \
    --outdir zeta_sigma_out
"""
import argparse, os, sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- basic transforms ----------------
def z_to_zeta(z):
    z = np.clip(np.asarray(z, float), 0, None)
    return 2.0*np.arctan(np.sqrt(z))

def dz_dzeta(z):
    z = np.clip(np.asarray(z, float), 0, None)
    return np.sqrt(z)*(1.0+z)

# ---------------- coords & sphere grids ----------------
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
    return np.stack([cB*np.cos(L), cB*np.sin(L), np.sin(B)], axis=1)

def cart_to_sph(xyz):
    x,y,z = xyz[...,0], xyz[...,1], xyz[...,2]
    B = np.arcsin(np.clip(z, -1.0, 1.0))
    L = (np.arctan2(y, x)) % (2*np.pi)
    return np.rad2deg(L), np.rad2deg(B)

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

def grid_lonlat(step_lon=6.0, step_lat=4.0):
    lons = np.arange(0.0, 360.0, step_lon)
    lats = np.arange(-90.0, 90.0+1e-9, step_lat)
    LL, BB = np.meshgrid(lons, lats)
    lon = LL.ravel(); lat = BB.ravel()
    xyz = sph_to_cart(lon, lat)
    return lon, lat, xyz

# ---------------- sliding windows in ζ ----------------
def sliding_windows_stats(z, zeta, z_err, dzeta, step, zmin, zmax, minNwin, subtract_zerr):
    """Return arrays for windows: zc, Nwin, varz_use, denom=(dz/dζ)^2 at zc."""
    m = np.isfinite(z) & np.isfinite(zeta) & (z>=zmin) & (z<=zmax)
    if z_err is not None:
        m &= np.isfinite(z_err)
    z, zeta = z[m], zeta[m]
    z_err = z_err[m] if z_err is not None else None

    if z.size < minNwin:
        return np.array([]), np.array([]), np.array([]), np.array([])

    order = np.argsort(zeta)
    z, zeta = z[order], zeta[order]
    z_err = z_err[order] if z_err is not None else None

    zeta_min, zeta_max = zeta.min(), zeta.max()
    centers = np.arange(zeta_min + dzeta/2, zeta_max - dzeta/2 + 1e-12, step)

    out_zc, out_N, out_varuse, out_denom = [], [], [], []
    L = zeta.size
    j0 = 0
    for c in centers:
        a, b = c - dzeta/2, c + dzeta/2
        while j0 < L and zeta[j0] < a: j0 += 1
        j1 = j0
        while j1 < L and zeta[j1] <= b: j1 += 1
        n = j1 - j0
        if n < minNwin:
            continue
        chunk = z[j0:j1]
        v = float(np.var(chunk, ddof=1)) if n>1 else np.nan
        if subtract_zerr and (z_err is not None):
            m2 = float(np.median(np.square(z_err[j0:j1])))
            v_use = max(v - m2, 0.0) if np.isfinite(m2) else v
        else:
            v_use = v
        zc = float(np.median(chunk))
        denom = float(dz_dzeta(zc)**2) if np.isfinite(zc) else np.nan
        if (not np.isfinite(denom)) or denom <= 0: 
            continue
        out_zc.append(zc); out_N.append(int(n)); out_varuse.append(v_use); out_denom.append(denom)

    return np.array(out_zc), np.array(out_N), np.array(out_varuse), np.array(out_denom)

def robust_stat_per_width(zc, Nw, varuse, denom):
    """Compute Y = median( varuse/denom ) and its robust SE via MAD/sqrt(n)."""
    y_win = np.asarray(varuse, float)/np.asarray(denom, float)
    m = np.isfinite(y_win)
    y = y_win[m]
    if y.size == 0:
        return np.nan, np.nan, 0
    med = float(np.median(y))
    mad = float(np.median(np.abs(y - med))) if y.size>1 else 0.0
    # normal consistency: σ ≈ 1.4826 * MAD; SE ≈ σ / sqrt(n_eff)
    sigma = 1.4826 * mad
    se = sigma / math.sqrt(max(1, y.size))
    return med, se, int(y.size)

# ---------------- regression Y(Δζ^2) = a + b X ----------------
def fit_line_x_to_y(X, Y, SE=None):
    """Weighted least squares. Return a,b, SE_a, SE_b, R2, n."""
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    m = np.isfinite(X) & np.isfinite(Y)
    if np.any(SE is not None):
        se = np.asarray(SE, float)
        m &= np.isfinite(se) & (se>0)
    X, Y = X[m], Y[m]
    if X.size < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan, int(X.size)
    if SE is None:
        w = np.ones_like(X)
    else:
        w = 1.0/np.square(se[m])
    W = np.diag(w)
    A = np.vstack([np.ones_like(X), X]).T
    # beta = (A^T W A)^{-1} A^T W Y
    AtW = A.T * w
    M = AtW @ A
    try:
        Minv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, np.nan, np.nan, int(X.size)
    beta = Minv @ (AtW @ Y)
    a, b = float(beta[0]), float(beta[1])
    # residuals, weighted SSE
    Yhat = A @ beta
    resid = Y - Yhat
    # effective dof ~ n-2
    dof = max(1, X.size - 2)
    s2 = float(np.sum(w*resid*resid) / dof)
    cov = Minv * s2
    SE_a = float(np.sqrt(max(cov[0,0], 0.0)))
    SE_b = float(np.sqrt(max(cov[1,1], 0.0)))
    # R^2 (weighted)
    ybar = float(np.sum(w*Y)/np.sum(w))
    ss_tot = float(np.sum(w*(Y - ybar)**2))
    ss_res = float(np.sum(w*resid*resid))
    R2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return a, b, SE_a, SE_b, R2, int(X.size)

# ---------------- plotting ----------------
def mollweide_scatter(lon_deg, lat_deg, val, title, fname, vcenter='sym', cmap='coolwarm'):
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

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cat', type=str, required=True, help='путь к каталогу (csv/tsv/whitespace)')
    ap.add_argument('--sep', type=str, default=None, help="разделитель: ',', '\\t' или 'auto'(по умолчанию auto/whitespace)")
    ap.add_argument('--cols', nargs='+', required=True, help='имена колонок: RA DEC Z [ZERR]')

    ap.add_argument('--frame', choices=['gal','equ'], default='gal')
    ap.add_argument('--grid', choices=['fibonacci','lonlat'], default='fibonacci')
    ap.add_argument('--npix', type=int, default=4096)
    ap.add_argument('--step-lon', type=float, default=6.0)
    ap.add_argument('--step-lat', type=float, default=4.0)
    ap.add_argument('--radius', type=float, default=15.0)

    ap.add_argument('--zmin', type=float, default=0.02)
    ap.add_argument('--zmax', type=float, default=1.0)

    ap.add_argument('--dzetas', nargs='+', type=float, default=[0.06,0.08,0.10,0.12], help='набор Δζ (рад)')
    ap.add_argument('--step', type=float, default=0.02, help='шаг окна по ζ (рад)')
    ap.add_argument('--minNcap', type=int, default=1000)
    ap.add_argument('--minNwin', type=int, default=120)
    ap.add_argument('--subtract-zerr', action='store_true')

    ap.add_argument('--outdir', type=str, default='zeta_sigma_out')
    ap.add_argument('--save-examples', type=int, default=0, help='сколько кэпсов сохранить с диаграммой Y vs Δζ^2')
    args, _ = ap.parse_known_args()
    os.makedirs(args.outdir, exist_ok=True)

    # ---- load ----
    if args.sep is None or args.sep.lower()=='auto':
        df = pd.read_csv(args.cat, sep=r'\s+', engine='python', comment='#')
    else:
        df = pd.read_csv(args.cat, sep=args.sep, engine='python', comment='#')

    if len(args.cols) < 3:
        print("Нужно минимум 3 колонки: RA DEC Z [ZERR]", file=sys.stderr); sys.exit(1)
    RA  = df[args.cols[0]].astype(float).values
    DEC = df[args.cols[1]].astype(float).values
    Z   = df[args.cols[2]].astype(float).values
    ZERR = df[args.cols[3]].astype(float).values if (len(args.cols)>3 and args.cols[3] in df.columns) else None

    m = np.isfinite(RA) & np.isfinite(DEC) & np.isfinite(Z) & (Z>=args.zmin) & (Z<=args.zmax)
    if ZERR is not None: m &= np.isfinite(ZERR)
    RA,DEC,Z = RA[m],DEC[m],Z[m]; ZERR = ZERR[m] if ZERR is not None else None

    # coords
    if args.frame=='gal':
        lon_sn, lat_sn = radec_to_gal(RA, DEC)
    else:
        lon_sn, lat_sn = RA, DEC
    xyz_sn = sph_to_cart(lon_sn, lat_sn)
    zeta = z_to_zeta(Z)

    # centers grid
    if args.grid=='fibonacci':
        lon_c, lat_c, xyz_c = grid_fibonacci(args.npix)
    else:
        lon_c, lat_c, xyz_c = grid_lonlat(args.step_lon, args.step_lat)

    cosR = np.cos(np.deg2rad(args.radius))

    # main loop over caps
    rows = []
    example_count = 0
    for i in range(xyz_c.shape[0]):
        c = xyz_c[i]
        inside = (xyz_sn @ c) >= cosR
        Ncap = int(np.sum(inside))
        if Ncap < args.minNcap:
            rows.append(dict(lon=lon_c[i], lat=lat_c[i], Ncap=Ncap,
                             sigma2_zeta=np.nan, sigma2_zeta_err=np.nan,
                             sigma_zeta_rad=np.nan, sigma_zeta_deg=np.nan,
                             slope_b=np.nan, R2=np.nan, n_widths=0, n_windows=0))
            continue

        z_cap = Z[inside]; zeta_cap = zeta[inside]
        zerr_cap = ZERR[inside] if ZERR is not None else None

        X = []   # Δζ^2
        Y = []   # median over windows of (varuse / denom)
        SE = []  # SE of that median (robust)
        total_windows = 0

        for dz in args.dzetas:
            zc, Nw, varuse, denom = sliding_windows_stats(
                z_cap, zeta_cap, zerr_cap, dzeta=dz, step=args.step,
                zmin=args.zmin, zmax=args.zmax, minNwin=args.minNwin,
                subtract_zerr=args.subtract_zerr
            )
            if zc.size == 0:
                continue
            y_med, y_se, nwin = robust_stat_per_width(zc, Nw, varuse, denom)
            if np.isfinite(y_med):
                X.append(dz*dz); Y.append(y_med); SE.append(max(y_se, 1e-12))
                total_windows += nwin

        if len(X) < 2:
            rows.append(dict(lon=lon_c[i], lat=lat_c[i], Ncap=Ncap,
                             sigma2_zeta=np.nan, sigma2_zeta_err=np.nan,
                             sigma_zeta_rad=np.nan, sigma_zeta_deg=np.nan,
                             slope_b=np.nan, R2=np.nan, n_widths=len(X), n_windows=total_windows))
            continue

        a, b, SE_a, SE_b, R2, nused = fit_line_x_to_y(np.array(X), np.array(Y), np.array(SE))
        # a ≈ σ_ζ^2  (clip to >=0)
        sigma2 = max(a, 0.0) if np.isfinite(a) else np.nan
        sigma2_err = SE_a if np.isfinite(SE_a) else np.nan
        sigma = math.sqrt(sigma2) if np.isfinite(sigma2) else np.nan
        sigma_deg = sigma * 180.0/np.pi if np.isfinite(sigma) else np.nan

        rows.append(dict(lon=lon_c[i], lat=lat_c[i], Ncap=Ncap,
                         sigma2_zeta=sigma2, sigma2_zeta_err=sigma2_err,
                         sigma_zeta_rad=sigma, sigma_zeta_deg=sigma_deg,
                         slope_b=b, slope_b_expected=1.0/12.0, R2=R2,
                         n_widths=nused, n_windows=total_windows))

        # optional: save Y vs Δζ^2 diagnostic for a few caps
        if args.save_examples and example_count < args.save_examples:
            example_count += 1
            xx = np.array(X); yy = np.array(Y); ee = np.array(SE)
            xfit = np.linspace(0, xx.max()*1.05, 100)
            yfit = (a + b*xfit) if np.all(np.isfinite([a,b])) else None
            plt.figure(figsize=(5.2,4))
            plt.errorbar(xx, yy, yerr=ee, fmt='o', ms=4, capsize=3, label='median over ζ-windows')
            if yfit is not None:
                plt.plot(xfit, yfit, '-', label=f'fit: a={a:.4e}, b={b:.4f} (exp 1/12≈{1/12:.4f})')
            plt.xlabel(r'$\Delta\zeta^2$ (rad$^2$)')
            plt.ylabel(r'$Y=\mathrm{median}\left[\,(s_z^2-\tilde\sigma_{z,meas}^2)/(dz/d\zeta)^2\,\right]$')
            plt.title(f'Cap #{i}: lon={lon_c[i]:.1f}, lat={lat_c[i]:.1f}, N={Ncap}')
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, f'cap_{i:05d}_Y_vs_dzeta2.png'), dpi=160)
            plt.close()

    # save table
    out = pd.DataFrame(rows)
    out_csv = os.path.join(args.outdir, "results_sigma_zeta.csv")
    out.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

    # maps
    mollweide_scatter(out['lon'].values, out['lat'].values, out['sigma_zeta_deg'].values,
                      title=r'$\hat\sigma_\zeta$ (deg)  —  extrapolated at $\Delta\zeta\to0$',
                      fname=os.path.join(args.outdir, "sphere_sigma_zeta_deg.png"),
                      vcenter=None, cmap='plasma')
    mollweide_scatter(out['lon'].values, out['lat'].values, out['slope_b'].values,
                      title=r'slope $b$ in $Y=a+b\,\Delta\zeta^2$  (expect $\approx 1/12$)',
                      fname=os.path.join(args.outdir, "sphere_slope_b.png"),
                      vcenter='sym', cmap='coolwarm')
    mollweide_scatter(out['lon'].values, out['lat'].values, out['Ncap'].values,
                      title=f'N per cap (R={args.radius}°)',
                      fname=os.path.join(args.outdir, "sphere_Ncap.png"),
                      vcenter=None, cmap='viridis')
    print("Saved maps in:", args.outdir)

if __name__ == "__main__":
    main()
