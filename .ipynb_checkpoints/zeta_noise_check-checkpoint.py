# zeta_noise_check.py  — сравнение гетероскедастичности в z vs ζ
import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import scipy.stats as spstats
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

def add_zeta(df, col_z):
    z = df[col_z].to_numpy(dtype=float)
    x = np.clip(1.0/(1.0+z), 0.0, 1.0)
    df["zeta"] = np.arccos(x)
    return df

def flat_lcdm_mu(z, H0=73.0, Om=0.3, Ok=0.0, Moff=0.0):
    c = 299792.458
    Ol = 1.0 - Om - Ok
    def dC(zi):
        n = max(100, int(400*(zi+0.1)))
        zz = np.linspace(0.0, zi, n+1)
        Ez = np.sqrt(Om*(1+zz)**3 + Ok*(1+zz)**2 + Ol)
        return (c/H0) * np.trapz(1.0/Ez, zz)
    dC_arr = np.array([dC(zi) for zi in z])
    dL = (1+z)*dC_arr
    return 5.0*np.log10(np.maximum(dL, 1e-12)) + 25.0 + Moff

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
        counts.append(int(yy.size))
        vars_.append(v)
    return np.array(centers), np.array(counts), np.array(vars_)

def bp_like(x, resid):
    y = resid**2
    X = np.vstack([np.ones_like(x), x]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_tot = np.sum((y - y.mean())**2)
    ss_reg = np.sum((yhat - y.mean())**2)
    r2 = (ss_reg/ss_tot) if ss_tot>0 else 0.0
    stat = len(y)*r2
    pval = float(spstats.chi2.sf(stat, df=1)) if SCIPY_OK else None
    return float(stat), float(r2), pval

def pick_col(df, user_col, candidates, label):
    if user_col in df.columns:
        print(f"[ok] {label}: using '{user_col}'")
        return user_col
    for c in candidates:
        if c in df.columns:
            print(f"[auto] {label}: '{user_col}' not found, using '{c}'")
            return c
    raise KeyError(f"{label}: none of {candidates} in columns. Got: {list(df.columns)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--sep", default=r"\s+", help="CSV sep/regex; Pantheon+ обычно '\\s+'")
    ap.add_argument("--col-z", dest="col_z", default="zCMB")
    ap.add_argument("--col-mu", dest="col_mu", default="MU_SH0ES")
    ap.add_argument("--drop_calibrators", action="store_true")
    ap.add_argument("--only_hubble_flow", action="store_true")
    ap.add_argument("--H0", type=float, default=73.0)
    ap.add_argument("--Om", type=float, default=0.3)
    ap.add_argument("--Moff", type=float, default=0.0)
    ap.add_argument("--nbins", type=int, default=20)
    ap.add_argument("--outdir", default="zeta_noise_out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.input, sep=args.sep, engine="python")
    print("[columns]", list(df.columns))  # диагностика

    # авто-выбор имён колонок
    z_candidates  = ["zCMB", "zHD", "zHEL", "z"]
    mu_candidates = ["MU_SH0ES", "MU", "m_b_corr"]
    args.col_z  = pick_col(df, args.col_z,  z_candidates,  "redshift")
    args.col_mu = pick_col(df, args.col_mu, mu_candidates, "distance modulus")

    # Фильтры для Pantheon+SH0ES
    if args.drop_calibrators and "IS_CALIBRATOR" in df.columns:
        df = df[df["IS_CALIBRATOR"]==0].copy()
    if args.only_hubble_flow and "USED_IN_SH0ES_HF" in df.columns:
        df = df[df["USED_IN_SH0ES_HF"]==1].copy()

    df = add_zeta(df, args.col_z)

    # Остатки относительно flat ΛCDM
    z = df[args.col_z].to_numpy(dtype=float)
    mu_obs = df[args.col_mu].to_numpy(dtype=float)
    mu_th = flat_lcdm_mu(z, H0=args.H0, Om=args.Om, Ok=0.0, Moff=args.Moff)
    resid = mu_obs - mu_th
    df["mu_model"] = mu_th
    df["resid"] = resid

    # Биннинг и дисперсии
    edges_z = bins_equal_count(z, args.nbins)
    edges_ze = bins_equal_count(df["zeta"].to_numpy(), args.nbins)
    c_z, n_z, v_z   = binned_var(z, resid, edges_z)
    c_ze, n_ze, v_ze= binned_var(df["zeta"].to_numpy(), resid, edges_ze)

    # График профилей дисперсий
    plt.figure()
    plt.plot(c_z, v_z, marker="o", label="var(resid) vs z")
    plt.plot(c_ze, v_ze, marker="s", label="var(resid) vs ζ")
    plt.xlabel("axis value")
    plt.ylabel("variance of residuals")
    plt.legend()
    plt.title("Binned variance profiles")
    plt.savefig(os.path.join(args.outdir, "variance_profiles.png"), dpi=160, bbox_inches="tight")
    plt.close()

    # Простой Breusch–Pagan-подобный тест (n·R²)
    bpz = bp_like(z, resid)
    bpζ = bp_like(df["zeta"].to_numpy(), resid)

    # Сохранения
    df.to_csv(os.path.join(args.outdir, "data_with_zeta_resid.csv"), index=False)
    pd.DataFrame({
        "center_z": c_z, "count_z": n_z, "var_z": v_z,
        "center_zeta": c_ze, "count_zeta": n_ze, "var_zeta": v_ze
    }).to_csv(os.path.join(args.outdir, "binned_variances.csv"), index=False)

    with open(os.path.join(args.outdir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "bp_like": {"z": {"stat": bpz[0], "R2": bpz[1], "pval": bpz[2]},
                        "zeta": {"stat": bpζ[0], "R2": bpζ[1], "pval": bpζ[2]}},
            "config": vars(args)
        }, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
