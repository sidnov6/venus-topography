"""Fit the Magellan incidence angle from the data, instead of assuming it.

    python -m data.calibrate_geometry --patches 120 --out data_raw/incidence_fit.json

`data.geometry` ships a documented *placeholder* profile: incidence peaking near 45
degrees around 10 N and falling toward the poles. The physics loss reads those angles
directly, so a systematic few-degree error becomes a systematic slope error over the whole
planet — which is the kind of mistake that produces a confident, plausible, wrong DEM.

The architecture note says to recover the real angles from the F-BIDR labels. There is a
second route that uses only data already downloaded, and it is a better check because it
measures what the mosaic actually contains rather than what its metadata claims:

    RV_observed  =  10 log10( M(theta - alpha) / M(theta) )  +  b

Over the ~20% of Venus with a stereo DEM, `alpha` is known. So for a patch, the only
unknowns are the incidence angle `theta` and a constant intrinsic brightness `b`, and
`theta` can be recovered by a one-dimensional search with `b` eliminated in closed form
at each step. Repeat over patches, bin by latitude, and refit the profile.

Both the SAR and the DEM are compared at the stereo DEM's own 600 m posting: at 75 m the
radiometric residual is dominated by speckle, and the stereo DEM has no real information
there anyway.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import geometry
from .sources import PRODUCTS, StereoDEM, Window, sample

MUHLEMAN_A, MUHLEMAN_C = 0.0118, 0.111


def muhleman_db(theta: np.ndarray) -> np.ndarray:
    t = np.clip(theta, 1e-3, math.pi / 2 - 1e-3)
    sigma = MUHLEMAN_A * np.cos(t) / np.maximum(np.sin(t) + MUHLEMAN_C * np.cos(t), 1e-9) ** 3
    return 10.0 * np.log10(np.maximum(sigma, 1e-12))


@dataclass
class PatchFit:
    lon: float
    lat: float
    theta_deg: float
    brightness_db: float
    residual_db: float
    n_pixels: int
    slope_std_deg: float


def fit_patch(rv: np.ndarray, alpha: np.ndarray, valid: np.ndarray,
              theta_grid_deg: np.ndarray) -> tuple[float, float, float]:
    """Recover `(theta, b, residual)` for one patch.

    For each candidate `theta`, the model is linear in `b`, so the optimal `b` is just the
    mean residual and can be removed analytically. That leaves a clean 1-D search and no
    optimiser to tune.
    """
    best = (float("nan"), float("nan"), float("inf"))
    rv_v, a_v = rv[valid], alpha[valid]
    if rv_v.size < 200:
        return best

    for theta_deg in theta_grid_deg:
        theta = math.radians(theta_deg)
        local = theta - a_v
        # Facets past grazing or into layover carry no usable information.
        ok = (local > math.radians(3.0)) & (local < math.radians(87.0))
        if ok.sum() < 200:
            continue
        pred = muhleman_db(local[ok]) - muhleman_db(np.full(ok.sum(), theta))
        resid = rv_v[ok] - pred
        b = float(np.median(resid))
        rms = float(np.sqrt(np.mean((resid - b) ** 2)))
        if rms < best[2]:
            best = (float(theta_deg), b, rms)
    return best


def slope_toward_radar(z: np.ndarray, pixel_size_m: float, look: np.ndarray) -> np.ndarray:
    """`alpha` in radians on a north-up grid, using the repo's down-range convention."""
    dz_dr, dz_dc = np.gradient(z, pixel_size_m)
    dz_de, dz_dn = dz_dc, -dz_dr
    return np.arctan(dz_de * look[0] + dz_dn * look[1])


def fit_from_store(store_path: Path, pixel_size_m: float, min_slope_std_deg: float,
                   theta_grid: np.ndarray) -> list[PatchFit]:
    """Fit from an already-ingested tile store, with no network at all.

    The store holds the same SAR and the same stereo DEM the S3 path would fetch, already
    on a common grid — so this is the same measurement, minutes faster, and it is the one
    to use once `data.ingest` has run.
    """
    from data.real import _Store

    st = _Store(store_path)
    dn, sdem, sok = st["dn_left"], st["stereo_dem"], st["stereo_valid"]
    lat, lon = st["lat"], st["lon"]
    look = geometry.look_vector("left")

    fits: list[PatchFit] = []
    for i in range(len(lat)):
        ok = np.asarray(sok[i]) > 0.5
        d = np.asarray(dn[i])
        valid = ok & (d > 0)
        if valid.mean() < 0.5:
            continue
        z = np.asarray(sdem[i], dtype=np.float64)
        alpha = slope_toward_radar(z, pixel_size_m, look)
        ss = float(np.degrees(alpha[valid].std()))
        if ss < min_slope_std_deg:
            continue
        rv = (d - 1.0) / 5.0 - 20.0
        theta_deg, b, resid = fit_patch(rv, alpha, valid, theta_grid)
        if math.isfinite(theta_deg):
            fits.append(PatchFit(float(lon[i]), float(lat[i]), theta_deg, b, resid,
                                 int(valid.sum()), ss))
    return fits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-store", type=Path,
                    help="an ingested tile store; skips the network entirely")
    ap.add_argument("--raw", type=Path, default=Path("data_raw"))
    ap.add_argument("--patches", type=int, default=120)
    ap.add_argument("--span-km", type=float, default=60.0)
    ap.add_argument("--pixel-size", type=float, default=600.0)
    ap.add_argument("--min-slope-std-deg", type=float, default=0.8,
                    help="a flat patch constrains nothing; the fit needs relief")
    ap.add_argument("--out", type=Path, default=Path("data_raw/incidence_fit.json"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import os
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
    # Without these a stalled connection blocks forever: the process sits at 0% CPU with
    # no error and no progress, which is indistinguishable from slow.
    os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
    os.environ.setdefault("GDAL_HTTP_CONNECTTIMEOUT", "20")
    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
    os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
    os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")

    theta_grid_all = np.arange(8.0, 56.0, 0.5)
    if a.from_store:
        fits = fit_from_store(a.from_store, a.pixel_size, a.min_slope_std_deg, theta_grid_all)
        print(f"fitted {len(fits)} tiles from {a.from_store} (no network)")
        _report(fits, a.out)
        return

    img = a.raw / "mosaic_allstereo.img"
    stereo = StereoDEM.from_label(img)
    west, south, east, north = stereo.bounds_deg
    south = max(south, stereo.available_south_deg())
    frac = img.stat().st_size / stereo.expected_bytes
    print(f"stereo DEM {frac:.0%} present; usable lon {west:.1f}..{east:.1f}, "
          f"lat {south:.1f}..{north:.1f}")

    rng = np.random.default_rng(a.seed)
    look = geometry.look_vector("left")
    theta_grid = theta_grid_all

    fits: list[PatchFit] = []
    tried = 0
    while len(fits) < a.patches and tried < a.patches * 25:
        tried += 1
        lon = float(rng.uniform(west + 1, east - 1)) % 360.0
        lat = float(rng.uniform(max(south + 1, -75), min(north - 1, 75)))
        win = Window.centred(lon, lat, a.span_km, a.pixel_size)
        try:
            z, zok = stereo.read(win)
            if zok.mean() < 0.9:
                continue
            dn, dok = sample(PRODUCTS["sar_left_75m"].uri(a.raw), win, resampling="average",
                             nodata_out=0.0)
        except Exception:
            continue
        valid = zok & dok & (dn > 0)
        if valid.mean() < 0.8:
            continue

        alpha = slope_toward_radar(np.nan_to_num(z), a.pixel_size, look)
        slope_std = float(np.degrees(alpha[valid].std()))
        if slope_std < a.min_slope_std_deg:
            continue

        rv = (dn - 1.0) / 5.0 - 20.0
        theta_deg, b, resid = fit_patch(rv, alpha, valid, theta_grid)
        if not math.isfinite(theta_deg):
            continue
        fits.append(PatchFit(lon, lat, theta_deg, b, resid, int(valid.sum()), slope_std))
        if len(fits) % 10 == 0:
            print(f"  {len(fits)} patches fitted ({tried} tried)", flush=True)

    _report(fits, a.out)


def _report(fits: list[PatchFit], out_path: Path) -> None:
    if len(fits) < 12:
        raise SystemExit(f"only {len(fits)} usable patches; not enough to fit a profile")

    lat = np.array([f.lat for f in fits])
    th = np.array([f.theta_deg for f in fits])
    w = np.array([1.0 / max(f.residual_db, 0.2) ** 2 for f in fits])

    model = geometry.fit_incidence_from_labels(lat, th, "cycle1_left_fitted")
    placeholder = geometry.CYCLE1

    # --- is the fit actually determined by the data? -------------------------------
    #
    # The model's sensitivity to slope falls monotonically with theta: a +/-3 degree slope
    # swing moves the predicted RV by 3.6 dB at 15 degrees and 1.4 dB at 53. So when the
    # observation does not track slope, the least-squares fit does not fail loudly — it
    # slides to the largest theta on the grid, because that predicts the least variation.
    # A flat profile at the top of the search range is the signature.
    grid_top = 55.5
    pinned = float(np.mean(th >= grid_top - 0.51))
    bad_peak = abs(model.lat_peak_deg) > 90.0
    degenerate = pinned > 0.30 or bad_peak

    print(f"\n{len(fits)} patches, |lat| up to {np.abs(lat).max():.0f} deg")
    print(f"  median residual {np.median([f.residual_db for f in fits]):.2f} dB")
    print(f"\n{'lat band':>12s} {'n':>4s} {'fitted theta':>13s} {'placeholder':>12s}")
    for lo in range(-70, 80, 20):
        m = (lat >= lo) & (lat < lo + 20)
        if m.sum() >= 3:
            print(f"{lo:+4d}..{lo + 20:+4d} {m.sum():4d} {np.average(th[m], weights=w[m]):12.1f}° "
                  f"{float(placeholder.theta_deg(lat[m].mean())):11.1f}°")

    out: dict = {
        "n_patches": len(fits),
        "degenerate": False,
        "fitted": {"lat_peak_deg": model.lat_peak_deg, "theta_peak_deg": model.theta_peak_deg,
                   "curvature_deg_per_deg2": model.curvature_deg_per_deg2,
                   "theta_min_deg": model.theta_min_deg},
        "placeholder": {"lat_peak_deg": placeholder.lat_peak_deg,
                        "theta_peak_deg": placeholder.theta_peak_deg,
                        "curvature_deg_per_deg2": placeholder.curvature_deg_per_deg2,
                        "theta_min_deg": placeholder.theta_min_deg},
        "patches": [vars(f) for f in fits],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    if degenerate:
        print()
        print("  REFUSING to emit a profile: the fit is degenerate.")
        print(f"    {pinned:.0%} of tiles are railed against the top of the search grid"
              + (f", and the quadratic peaks at latitude {model.lat_peak_deg:.0f} deg,"
                 " which is not a latitude." if bad_peak else "."))
        print("    That is a fit sliding to the least-sensitive theta, not measuring one.")
        print("    The cause is upstream: the stereo DEM's 71 m vertical noise becomes")
        print("    ~9.5 deg of slope noise at its 600 m posting, against real slopes of a")
        print("    few degrees, so brightness and stereo-derived slope correlate at only")
        print("    r=+0.09. Recovering the angles needs the F-BIDR labels, as the")
        print("    architecture note originally specified.")
        out["degenerate"] = True
        out["fitted"] = None
    else:
        print(f"\nfitted profile: peak {model.theta_peak_deg:.1f}° at {model.lat_peak_deg:+.1f}°, "
              f"curvature {model.curvature_deg_per_deg2:.5f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
