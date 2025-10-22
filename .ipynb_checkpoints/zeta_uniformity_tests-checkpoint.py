
# zeta_uniformity_tests.py
import os, json, argparse, math, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import scipy.stats as spstats
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

rng_global = np.random.default_rng(42)

def compute_zeta(z):
    x = np.clip(1.0/(1.0+z.astype(float)), 0.0, 1.0)
    return np.arccos(x)

def tricube(u):
    a = np.clip(1 - np.abs(u)**3, 0, None)
    return a**3

def lowess(x, y, frac=0.2, it=1):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    r = int(np.clip(np.ceil(frac * n), 2, n))
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    y_hat = np.zeros(n, float)
    bw = np.empty(n, float)
    for i in range(n):
        lo = max(0, i - r)
        hi = min(n - 1, i + r)
        idx = np.arange(lo, hi + 1)
        bw[i] = np.max(np.abs(xs[i] - xs[idx])) or 1.0
    rw = np.ones(n, float)
    for itn in range(it+1):
        for i in range(n):
            d = np.abs(xs - xs[i]) / (bw[i] if bw[i] > 0 else 1.0)
            w = tricube(d) * rw
            if np.sum(w) <= 0:
                y_hat[i] = ys[i]
                continue
            W = np.diag(w)
            X = np.vstack([np.ones(n), xs - xs[i]]).T
            try:
                beta = np.linalg.lstsq(W @ X, W @ ys, rcond=None)[0]
                y_hat[i] = beta[0]
            except Exception:
                y_hat[i] = np.average(ys, weights=w)
        resid = ys - y_hat
        if itn < it:
            s = np.median(np.abs(resid)) or 1.0
            u = resid / (6.0 * s)
            rw = (1 - u**2)**2
            rw[np.abs(u) >= 1] = 0.0
    y_fit = np.empty(n, float)
    y_fit[order] = y_hat
    return y_fit

def equal_count_edges(x, nbins):
    qs = np.linspace(0, 1, nbins+1)
    e = np.quantile(x, qs)
    for i in range(1, len(e)):
        if e[i] <= e[i-1]:
            e[i] = np.nextafter(e[i-1], np.inf)
    return e

def group_stats_by_bins(x, val, edges):
    idx = np.digitize(x, edges) - 1
    nb = len(edges) - 1
    centers, groups = [], []
    for b in range(nb):
        m = (idx == b)
        if np.sum(m) == 0:
            centers.append(0.5*(edges[b]+edges[b+1]))
            groups.append(np.array([], float))
            continue
        centers.append(0.5*(edges[b]+edges[b+1]))
        groups.append(val[m])
    return np.array(centers), groups

def binwise_sigma(groups):
    sigmas = []
    for g in groups:
        if len(g) >= 2:
            sigmas.append(np.std(g, ddof=1))
        else:
            sigmas.append(np.nan)
    sigmas = np.array(sigmas, float)
    cv = float(np.nanstd(sigmas) / np.nanmean(sigmas)) if np.nanmean(sigmas) > 0 else np.nan
    rnge = float(np.nanmax(sigmas) - np.nanmin(sigmas)) if np.isfinite(sigmas).any() else np.nan
    return sigmas, cv, rnge

def levene_like(groups, center='mean'):
    gs = [g[~np.isnan(g)] for g in groups if len(g) > 1]
    k = len(gs)
    ni = np.array([len(g) for g in gs])
    if k < 2 or np.any(ni < 2):
        return np.nan, np.nan
    if center == 'median':
        ci = np.array([np.median(g) for g in gs])
    else:
        ci = np.array([np.mean(g) for g in gs])
    zij = []
    for i, g in enumerate(gs):
        zij.append(np.abs(g - ci[i]))
    zij_all = np.concatenate(zij)
    grand = np.mean(zij_all)
    ss_between = 0.0
    for i in range(k):
        mi = np.mean(zij[i])
        ss_between += ni[i] * (mi - grand)**2
    ss_within = 0.0
    for i in range(k):
        ss_within += np.sum((zij[i] - np.mean(zij[i]))**2)
    df1 = k - 1
    df2 = np.sum(ni) - k
    F = (ss_between/df1) / (ss_within/df2) if ss_within > 0 else np.inf
    if SCIPY_OK:
        p = spstats.f.sf(F, df1, df2)
    else:
        p = np.nan
    return float(F), float(p)

def bp_test(x, r, degree=1):
    y = r**2
    X = np.vstack([x**d for d in range(degree+1)]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_tot = np.sum((y - y.mean())**2)
    ss_reg = np.sum((yhat - y.mean())**2)
    R2 = ss_reg / ss_tot if ss_tot > 0 else 0.0
    stat = len(y) * R2
    df = degree
    p = spstats.chi2.sf(stat, df) if SCIPY_OK else np.nan
    return float(stat), float(R2), float(p), df

def kb_test(x, r, degree=1):
    y = np.abs(r)
    X = np.vstack([x**d for d in range(degree+1)]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_tot = np.sum((y - y.mean())**2)
    ss_reg = np.sum((yhat - y.mean())**2)  # <-- intentional error to test write
    R2 = ss_reg / ss_tot if ss_tot > 0 else 0.0
    stat = len(y) * R2
    df = degree
    p = spstats.chi2.sf(stat, df) if SCIPY_OK else np.nan
    return float(stat), float(R2), float(p), df

def spearman_abs_rho(x, r):
    x = np.asarray(x)
    y = np.abs(np.asarray(r))
    if SCIPY_OK:
        rho, p = spstats.spearmanr(x, y)
        return float(rho), float(p)
    # custom rank correlation + permutation p-value
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rho = np.corrcoef(rx, ry)[0,1]
    return float(rho), np.nan

def cusumsq_stat(r):
    r = np.asarray(r)
    r = (r - np.mean(r)) / (np.std(r, ddof=1) or 1.0)
    s2 = r**2
    cs = np.cumsum(s2) / np.sum(s2)
    t = np.linspace(1/len(r), 1.0, len(r))
    dev = np.max(np.abs(cs - t))
    return float(dev)

def compare_axis(x, y, label, nbins=20, lowess_frac=0.2, perms=2000, outdir="out"):
    yhat = lowess(x, y, frac=lowess_frac, it=1)
    r = y - yhat
    edges = equal_count_edges(x, nbins)
    centers, groups = group_stats_by_bins(x, r, edges)
    sigmas, cv, rnge = binwise_sigma(groups)
    F_lev, p_lev  = levene_like(groups, center='mean')
    F_bf,  p_bf   = levene_like(groups, center='median')

    if SCIPY_OK:
        gs = [g[~np.isnan(g)] for g in groups if len(g) > 1]
        if len(gs) >= 2:
            try:
                fk_stat, fk_p = spstats.fligner(*gs)
            except Exception:
                fk_stat, fk_p = np.nan, np.nan
        else:
            fk_stat, fk_p = np.nan, np.nan
    else:
        fk_stat, fk_p = np.nan, np.nan

    bp_stat, bp_R2, bp_p, bp_df = bp_test(x, r, degree=1)
    kb_stat, kb_R2, kb_p, kb_df = kb_test(x, r, degree=1)

    if SCIPY_OK:
        rho, rho_p = spstats.spearmanr(x, np.abs(r))
    else:
        rho, rho_p = spearman_abs_rho(x, r)

    order = np.argsort(x)
    dev_obs = cusumsq_stat(r[order])
    cnt = 0
    for _ in range(perms):
        rp = rng_global.permutation(r)
        d = cusumsq_stat(rp[order])
        if d >= dev_obs: cnt += 1
    cusum_p = (cnt + 1) / (perms + 1)

    plt.figure()
    plt.plot(centers, sigmas, marker='o')
    plt.xlabel(label)
    plt.ylabel("binwise std of residuals")
    plt.title(f"Std profile vs {label}")
    plt.savefig(os.path.join(outdir, f"std_profile_vs_{'zeta' if label=='ζ' else 'z'}.png"), dpi=160, bbox_inches='tight')
    plt.close()

    return {
        "label": label,
        "nbins": nbins,
        "lowess_frac": lowess_frac,
        "cv_bin_std": float(cv),
        "range_bin_std": float(rnge),
        "levene_F": float(F_lev), "levene_p": (None if np.isnan(p_lev) else float(p_lev)),
        "brown_forsythe_F": float(F_bf), "brown_forsythe_p": (None if np.isnan(p_bf) else float(p_bf)),
        "fligner_stat": (None if np.isnan(fk_stat) else float(fk_stat)),
        "fligner_p": (None if np.isnan(fk_p) else float(fk_p)),
        "bp_stat": float(bp_stat), "bp_R2": float(bp_R2), "bp_df": int(bp_df),
        "bp_p": (None if np.isnan(bp_p) else float(bp_p)),
        "kb_stat": float(kb_stat), "kb_R2": float(kb_R2), "kb_df": int(kb_df),
        "kb_p": (None if np.isnan(kb_p) else float(kb_p)),
        "spearman_abs_r": float(rho), "spearman_p": (None if np.isnan(rho_p) else float(rho_p)),
        "cusumsq_dev": float(dev_obs), "cusumsq_perm_p": float(cusum_p)
    }

def main():
    ap = argparse.ArgumentParser(description="Uniformity-of-noise tests in z vs ζ (model-free)")
    ap.add_argument("--input", required=True, help="CSV with columns (at least) Z; RA/DEC optional")
    ap.add_argument("--sep", default=",")
    ap.add_argument("--col-z", default="Z")
    ap.add_argument("--nbins", type=int, default=20)
    ap.add_argument("--lowess_frac", type=float, default=0.2)
    ap.add_argument("--perms", type=int, default=2000)
    ap.add_argument("--outdir", default="zeta_uniformity_out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.input, sep=args.sep)
    if args.col_z not in df.columns:
        raise KeyError(f"Column '{args.col_z}' not in columns: {list(df.columns)[:20]} ...")

    Z = df[args.col_z].to_numpy(dtype=float)
    mask = np.isfinite(Z)
    Z = Z[mask]
    Zeta = compute_zeta(Z)

    res_z = compare_axis(Z, Z, label='z', nbins=args.nbins, lowess_frac=args.lowess_frac, perms=args.perms, outdir=args.outdir)
    res_ze = compare_axis(Zeta, Z, label='ζ', nbins=args.nbins, lowess_frac=args.lowess_frac, perms=args.perms, outdir=args.outdir)

    summary = {"z": res_z, "zeta": res_ze, "config": vars(args)}
    with open(os.path.join(args.outdir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    rows = []
    for k in ["cv_bin_std","range_bin_std","levene_F","levene_p","brown_forsythe_F","brown_forsythe_p",
              "fligner_stat","fligner_p","bp_stat","bp_R2","bp_df","bp_p","kb_stat","kb_R2","kb_df","kb_p",
              "spearman_abs_r","spearman_p","cusumsq_dev","cusumsq_perm_p"]:
        rows.append({"metric": k, "z": res_z.get(k), "zeta": res_ze.get(k)})
    pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "summary_side_by_side.csv"), index=False)

    print("Done. See:", args.outdir)

if __name__ == "__main__":
    main()
