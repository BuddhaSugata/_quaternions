
# desi_zeta_noise_check.py
import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def add_zeta(df, col_z):
    z = df[col_z].to_numpy(dtype=float)
    x = np.clip(1.0/(1.0+z), 0.0, 1.0)
    df["zeta"] = np.arccos(x)
    return df

def bins_equal_count(x, nbins):
    qs = np.linspace(0,1,nbins+1)
    e = np.quantile(x, qs)
    for i in range(1,len(e)):
        if e[i] <= e[i-1]:
            e[i] = np.nextafter(e[i-1], np.inf)
    return e

def binned_var(x, y, edges):
    idx = np.digitize(x, edges) - 1
    centers, counts, vars_ = [], [], []
    for b in range(len(edges)-1):
        m = (idx==b)
        yy = y[m]
        v = np.var(yy, ddof=1) if yy.size>=2 else np.nan
        centers.append(0.5*(edges[b]+edges[b+1]))
        counts.append(int(np.sum(m)))
        vars_.append(v)
    return np.array(centers), np.array(counts), np.array(vars_)

def bp_like(x, y):
    yy = y**2
    X = np.vstack([np.ones_like(x), x]).T
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    yhat = X @ beta
    ss_tot = np.sum((yy - yy.mean())**2)
    ss_reg = np.sum((yhat - yy.mean())**2)
    r2 = (ss_reg/ss_tot) if ss_tot>0 else 0.0
    stat = len(yy)*r2
    return float(stat), float(r2)

def sky_bin_stats(ra, dec, val, ra_bins=36, dec_bins=18):
    ra = np.mod(ra, 360.0)
    ra_edges  = np.linspace(0, 360, ra_bins+1)
    dec_edges = np.linspace(-90, 90, dec_bins+1)
    ra_idx  = np.digitize(ra, ra_edges) - 1
    dec_idx = np.digitize(dec, dec_edges) - 1
    var_grid = np.full((dec_bins, ra_bins), np.nan)
    cnt_grid = np.zeros((dec_bins, ra_bins), dtype=int)
    for i in range(dec_bins):
        for j in range(ra_bins):
            m = (dec_idx==i) & (ra_idx==j)
            if np.sum(m) >= 2:
                var_grid[i,j] = np.var(val[m], ddof=1)
                cnt_grid[i,j] = int(np.sum(m))
    ra_ctr  = 0.5*(ra_edges[:-1] + ra_edges[1:])
    dec_ctr = 0.5*(dec_edges[:-1]+ dec_edges[1:])
    return ra_ctr, dec_ctr, var_grid, cnt_grid

def main():
    ap = argparse.ArgumentParser(description="DESI ζ-noise check (RA,DEC,Z only)")
    ap.add_argument("--input", required=True, help="CSV with columns RA,DEC,Z (names configurable)")
    ap.add_argument("--sep", default=",", help="CSV separator (default ,)")
    ap.add_argument("--col-ra", default="RA")
    ap.add_argument("--col-dec", default="DEC")
    ap.add_argument("--col-z",   default="Z")
    ap.add_argument("--nbins", type=int, default=20, help="equal-count bins along axis")
    ap.add_argument("--ra_bins", type=int, default=36, help="RA bins for sky map")
    ap.add_argument("--dec_bins", type=int, default=18, help="DEC bins for sky map")
    ap.add_argument("--outdir", default="desi_zeta_noise_out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.input, sep=args.sep)

    for c in [args.col_ra, args.col_dec, args.col_z]:
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not found. Available: {list(df.columns)[:15]}...")

    df = df[[args.col_ra, args.col_dec, args.col_z]].rename(
        columns={args.col_ra:"RA", args.col_dec:"DEC", args.col_z:"Z"}).copy()

    df = add_zeta(df, "Z")

    Z  = df["Z"].to_numpy(dtype=float)
    ZE = df["zeta"].to_numpy(dtype=float)

    edges_z  = bins_equal_count(Z,  args.nbins)
    edges_ze = bins_equal_count(ZE, args.nbins)

    c_z,  n_z,  v_z  = binned_var(Z,  Z,  edges_z)
    c_ze, n_ze, v_ze = binned_var(ZE, Z,  edges_ze)

    pd.DataFrame({
        "center_z": c_z,   "count_z": n_z,   "var_Z_vs_z": v_z,
        "center_ze": c_ze, "count_ze": n_ze, "var_Z_vs_zeta": v_ze
    }).to_csv(os.path.join(args.outdir,"variance_profiles.csv"), index=False)

    plt.figure()
    plt.plot(c_z,  v_z,  marker="o", label="Var[Z | z-bin]")
    plt.plot(c_ze, v_ze, marker="s", label="Var[Z | ζ-bin]")
    plt.xlabel("axis value")
    plt.ylabel("variance of Z")
    plt.legend()
    plt.title("Binned variance profiles (DESI-like RA,DEC,Z)")
    plt.savefig(os.path.join(args.outdir, "variance_profiles.png"), dpi=160, bbox_inches="tight")
    plt.close()

    stat_z,  r2_z  = bp_like(Z,  Z)
    stat_ze, r2_ze = bp_like(ZE, Z)

    shells = np.quantile(Z, [0.0, 0.25, 0.5, 0.75, 1.0])
    shell_stats = []
    for i in range(len(shells)-1):
        zmin, zmax = shells[i], shells[i+1]
        m = (Z >= zmin) & (Z < zmax)
        if np.sum(m) < args.ra_bins:
            continue
        ra_ctr, dec_ctr, var_grid, cnt_grid = sky_bin_stats(
            df["RA"].to_numpy()[m],
            df["DEC"].to_numpy()[m],
            df["zeta"].to_numpy()[m],
            ra_bins=args.ra_bins,
            dec_bins=args.dec_bins
        )
        plt.figure()
        plt.imshow(var_grid, origin="lower",
                   extent=[ra_ctr.min(), ra_ctr.max(), dec_ctr.min(), dec_ctr.max()],
                   aspect="auto")
        plt.colorbar()
        plt.xlabel("RA (deg)")
        plt.ylabel("DEC (deg)")
        plt.title(f"Var[ζ] on sky, shell {zmin:.3f} ≤ z < {zmax:.3f}")
        fname = f"sky_var_zeta_shell_{i+1}.png"
        plt.savefig(os.path.join(args.outdir, fname), dpi=160, bbox_inches="tight")
        plt.close()
        shell_stats.append({
            "zmin": float(zmin), "zmax": float(zmax),
            "mean_var_zeta": float(np.nanmean(var_grid)),
            "nonempty_cells": int(np.sum(~np.isnan(var_grid)))
        })

    summary = {
        "bp_like": {
            "z":   {"stat": stat_z,  "R2": r2_z},
            "zeta":{"stat": stat_ze, "R2": r2_ze}
        },
        "config": vars(args),
        "shell_stats": shell_stats
    }
    with open(os.path.join(args.outdir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
