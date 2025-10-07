
"""
Unimetry D-rotation Sky Simulator (Quaternion-based)
----------------------------------------------------
- Defines D-rotation in {1, u} acting on imaginary quaternion 3-space (Im H).
- Transforms sky directions (RA, Dec) for a given speed beta along axis u (RA/Dec).
- Can load a Gaia-like CSV (columns: ra, dec, optional phot_g_mean_mag).
- Can generate a synthetic all-sky sample if no file is provided.
- Plots Mollweide maps before/after and saves transformed CSV.

Usage (example):
    from unimetry_d_rotation_sky import *
    df, src = load_catalog_or_synthetic("/mnt/data/gaia_sample.csv", N_synth=40000)
    out = transform_catalog(df, u_ra_deg=0.0, u_dec_deg=0.0, beta=0.8)
    out.to_csv("/mnt/data/sky_transformed.csv", index=False)
    plot_mollweide(df["ra"].values, df["dec"].values, size_by_mag=df.get("phot_g_mean_mag"), title="Original")
    plot_mollweide(out["ra_prime"].values, out["dec_prime"].values, size_by_mag=out.get("phot_g_mean_mag"), title="After D-rotation")

Notes:
- Internet is disabled in this environment; provide your own CSV to /mnt/data/gaia_sample.csv.
- D-rotation angle alpha is related to beta by tan(alpha) = beta; so cos(alpha) = 1/gamma, sin(alpha) = beta/gamma.
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Math helpers ----------

def deg2rad(x):
    return np.deg2rad(x)

def rad2deg(x):
    return np.rad2deg(x)

def sph_to_cart(ra_deg, dec_deg):
    """Convert RA,Dec in degrees to 3D unit vectors (x,y,z). RA in [0,360), Dec in [-90,90]."""
    ra = deg2rad(ra_deg)
    dec = deg2rad(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.stack([x, y, z], axis=-1)

def cart_to_sph(vecs):
    """Convert 3D unit vectors (N,3) to RA,Dec in degrees. RA in [0,360)."""
    x, y, z = vecs[:,0], vecs[:,1], vecs[:,2]
    ra = np.arctan2(y, x) % (2*np.pi)
    dec = np.arcsin(np.clip(z, -1.0, 1.0))
    return rad2deg(ra), rad2deg(dec)

def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.maximum(n, 1e-15)
    return v / n

def axis_from_radec(ra_deg, dec_deg):
    """Axis u (unit 3-vector) from RA/Dec (deg)."""
    return sph_to_cart(np.array([ra_deg]), np.array([dec_deg]))[0]

def alpha_from_beta(beta):
    """Map relative speed beta in [0,1) to D-rotation angle alpha via tan(alpha) = beta."""
    return math.atan(beta)

def d_rotate_dirs(n_vecs, u, beta):
    """
    Apply D-rotation along axis u with parameter beta (|beta|<1) to an array of unit directions n_vecs (N,3).
    Returns new unit directions (N,3).
    Formula in Im(H): n' = n_parallel + cos(alpha) * n_perp, then renormalize to unit.
    """
    alpha = alpha_from_beta(beta)
    c = math.cos(alpha)
    u = unit(u.reshape(1,3))[0]
    sigma = np.dot(n_vecs, u)             # (N,)
    n_par = (sigma[:,None]) * u[None,:]   # (N,3)
    n_perp = n_vecs - n_par               # (N,3)
    n_im = n_par + c * n_perp             # imaginary part after D-rotation
    n_out = unit(n_im)
    return n_out

# ---------- Catalog loading ----------

def load_catalog_or_synthetic(path="/mnt/data/gaia_sample.csv", N_synth=30000, seed=42):
    """
    Load a Gaia-like CSV with columns 'ra','dec' (degrees) and optional 'phot_g_mean_mag'.
    If not found, create a synthetic all-sky set of N_synth points with pseudo-magnitudes.
    """
    if os.path.exists(path):
        df = pd.read_csv(path)
        if not {"ra","dec"}.issubset(df.columns):
            raise ValueError("CSV must contain at least 'ra' and 'dec' columns in degrees.")
        if "phot_g_mean_mag" not in df.columns:
            df["phot_g_mean_mag"] = 12.0  # flat placeholder
        source = f"Loaded {len(df)} rows from {path}"
    else:
        rng = np.random.default_rng(seed)
        ra = rng.uniform(0, 360, N_synth)
        u = rng.uniform(-1, 1, N_synth)
        dec = np.degrees(np.arcsin(u))
        gmag = rng.uniform(5, 15, N_synth)  # crude placeholder
        df = pd.DataFrame({"ra": ra, "dec": dec, "phot_g_mean_mag": gmag})
        source = f"Synthetic catalog generated: N={N_synth}"
    return df, source

# ---------- Plotting ----------

def plot_mollweide(ra_deg, dec_deg, size_by_mag=None, title=""):
    """
    Plot a sky scatter in Mollweide projection.
    - ra_deg: degrees [0,360). We'll shift to [-180,180] and convert to radians.
    - dec_deg: degrees [-90,90]
    - size_by_mag: if provided (mag), convert to marker size. Otherwise fixed small size.
    """
    ra_wrapped = ((np.asarray(ra_deg) + 180.0) % 360.0) - 180.0
    x = np.deg2rad(ra_wrapped) * -1.0  # invert to match conventional sky maps
    y = np.deg2rad(np.asarray(dec_deg))

    plt.figure(figsize=(10, 5))
    ax = plt.subplot(111, projection="mollweide")
    if size_by_mag is not None:
        m = np.asarray(size_by_mag)
        m = np.clip(m, 2, 18)
        s = (10.0 ** (0.3 * (12.0 - m)))
        s = np.clip(s, 0.2, 50.0)
    else:
        s = 0.5
    ax.scatter(x, y, s=s)
    ax.grid(True)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# ---------- End-to-end transform ----------

def transform_catalog(df, u_ra_deg=0.0, u_dec_deg=0.0, beta=0.5):
    """
    Apply D-rotation transform to the catalog sky directions.
    Returns transformed DataFrame with new ra', dec' columns.
    """
    u = axis_from_radec(u_ra_deg, u_dec_deg)
    n = sph_to_cart(df["ra"].values, df["dec"].values)
    n2 = d_rotate_dirs(n, u, beta)
    ra2, dec2 = cart_to_sph(n2)
    out = df.copy()
    out["ra_prime"] = ra2
    out["dec_prime"] = dec2
    out.attrs["u_ra_deg"] = u_ra_deg
    out.attrs["u_dec_deg"] = u_dec_deg
    out.attrs["beta"] = beta
    return out

# ---------- Script demo (optional) ----------

if __name__ == "__main__":
    df, source_msg = load_catalog_or_synthetic("/mnt/data/gaia_sample.csv", N_synth=40000)
    print(source_msg)
    u_ra_deg_demo = 0.0
    u_dec_deg_demo = 0.0
    beta_demo = 0.8
    df2 = transform_catalog(df, u_ra_deg_demo, u_dec_deg_demo, beta_demo)
    out_path = "/mnt/data/sky_transformed.csv"
    df2.to_csv(out_path, index=False)
    print(f"Transformed catalog saved: {out_path}")
    plot_mollweide(df["ra"].values, df["dec"].values, size_by_mag=df.get("phot_g_mean_mag", None), 
                   title="Original sky (Mollweide)")
    plot_mollweide(df2["ra_prime"].values, df2["dec_prime"].values, size_by_mag=df2.get("phot_g_mean_mag", None), 
                   title=f"After D-rotation (beta={beta_demo}, u=({u_ra_deg_demo}°, {u_dec_deg_demo}°))")
