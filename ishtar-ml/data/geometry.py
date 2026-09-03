"""Per-pixel viewing geometry: incidence angle and look direction.

Magellan's incidence angle varied continuously along each orbit because the spacecraft
was in a highly elliptical orbit and the radar look angle was scheduled against range.
The mosaics do not ship a pixel-level angle raster, so we fit a smooth latitude model
per cycle and rasterise it.

**The coefficients below are a documented placeholder**, matching the Cycle 1 profile
described in the architecture note (~45 deg near periapsis around 10 deg N, falling
below 20 deg toward the poles). Before any run whose numbers you intend to publish,
replace `fit_incidence_from_labels` output with a fit to the actual F-BIDR / mosaic
labels: the physics loss reads these angles directly and a systematic few-degree error
becomes a systematic slope error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VENUS_RADIUS_M = 6_051_800.0


@dataclass(frozen=True)
class IncidenceModel:
    """Smooth incidence-vs-latitude profile for one Magellan cycle.

    `theta(lat) = theta_peak - k * (lat - lat_peak)^2`, clamped to `[theta_min, theta_peak]`.
    A quadratic is enough: the profile is broad and single-peaked, and the fit residual
    is far below the uncertainty in the surface model anyway.
    """

    lat_peak_deg: float
    theta_peak_deg: float
    curvature_deg_per_deg2: float
    theta_min_deg: float
    cycle: str

    def theta_deg(self, lat_deg: np.ndarray | float) -> np.ndarray:
        lat = np.asarray(lat_deg, dtype=np.float64)
        t = self.theta_peak_deg - self.curvature_deg_per_deg2 * (lat - self.lat_peak_deg) ** 2
        return np.clip(t, self.theta_min_deg, self.theta_peak_deg)

    def theta_rad(self, lat_deg: np.ndarray | float) -> np.ndarray:
        return np.deg2rad(self.theta_deg(lat_deg))


# PLACEHOLDER fits — see module docstring.
CYCLE1 = IncidenceModel(lat_peak_deg=10.0, theta_peak_deg=45.0, curvature_deg_per_deg2=0.0040,
                        theta_min_deg=17.0, cycle="cycle1_left")
CYCLE2 = IncidenceModel(lat_peak_deg=10.0, theta_peak_deg=45.0, curvature_deg_per_deg2=0.0040,
                        theta_min_deg=17.0, cycle="cycle2_right")
CYCLE3 = IncidenceModel(lat_peak_deg=10.0, theta_peak_deg=25.0, curvature_deg_per_deg2=0.0025,
                        theta_min_deg=12.0, cycle="cycle3_stereo")

INCIDENCE_MODELS = {"left": CYCLE1, "right": CYCLE2, "stereo": CYCLE3}


def look_vector(look: str, orbit_azimuth_deg: float = 0.0) -> np.ndarray:
    """Horizontal ground-range direction `(east, north)`, pointing **away from the radar**.

    This is the convention `model.physics` uses: terrain rising along `look_vector` is
    tilted toward the radar, so `alpha = atan(grad(z) . look_vec)` is positive there.

    Magellan's orbit was near-polar, so the ground track runs roughly north-south and the
    beam points roughly east-west. `orbit_azimuth_deg` is the ground-track azimuth
    clockwise from north; pass the real per-tile value when you have it, since it departs
    from 0 near the poles.
    """
    az = np.deg2rad(orbit_azimuth_deg)
    track = np.array([np.sin(az), np.cos(az)])  # (east, north) along the ground track
    # Left of track is the track rotated 90 deg counter-clockwise in the (E, N) plane.
    left_of_track = np.array([-track[1], track[0]])
    if look == "left":
        v = left_of_track          # a left-looking beam propagates to the left of track
    elif look in ("right", "stereo"):
        v = -left_of_track
    else:
        raise ValueError(f"unknown look {look!r}; expected 'left', 'right' or 'stereo'")
    return (v / np.linalg.norm(v)).astype(np.float32)


def incidence_raster(lats_deg: np.ndarray, look: str) -> np.ndarray:
    """Rasterise the incidence angle (radians) for a tile's latitude grid."""
    return INCIDENCE_MODELS[look].theta_rad(lats_deg).astype(np.float32)


def latitude_grid(lat_top_deg: float, lat_bottom_deg: float, height: int, width: int) -> np.ndarray:
    """Per-pixel latitude for a north-up cylindrical tile."""
    lats = np.linspace(lat_top_deg, lat_bottom_deg, height, dtype=np.float32)
    return np.repeat(lats[:, None], width, axis=1)


def pixel_size_m(resolution_deg_per_px: float) -> float:
    """Metres per pixel at the equator on the Venus sphere for a cylindrical grid."""
    return np.deg2rad(resolution_deg_per_px) * VENUS_RADIUS_M


def cos_lat_weight(lats_deg: np.ndarray, max_lat_deg: float = 80.0) -> np.ndarray:
    """Loss weight for a cylindrical grid: cos(lat), zeroed past `max_lat_deg`.

    Cylindrical tiles over-represent high latitudes and distort badly near the poles.
    Training stays equatorward of 80 deg; the caps are re-tiled in polar stereographic
    for inference only.

    **Not applied yet, deliberately.** It belongs in `dataset.build_batch`, weighting the
    per-tile loss contributions, and it only makes sense once tiles come from a real
    cylindrical mosaic. `data.synthetic` assigns each tile a latitude but does not distort
    it, so applying the weight there would down-weight high-latitude tiles that are not
    actually oversampled — a correction for a problem the synthetic set does not have.
    """
    w = np.cos(np.deg2rad(lats_deg)).astype(np.float32)
    return np.where(np.abs(lats_deg) > max_lat_deg, 0.0, w).astype(np.float32)


def fit_incidence_from_labels(lat_deg: np.ndarray, theta_deg: np.ndarray, cycle: str) -> IncidenceModel:
    """Least-squares refit of the quadratic profile to angles parsed from product labels.

    Call this once per cycle after ingesting the F-BIDR / mosaic metadata, and persist
    the result next to the tiles so training and inference cannot disagree.
    """
    lat = np.asarray(lat_deg, dtype=np.float64)
    th = np.asarray(theta_deg, dtype=np.float64)
    # theta = a + b*lat + c*lat^2  ->  peak at -b/2c
    A = np.stack([np.ones_like(lat), lat, lat**2], axis=1)
    coef, *_ = np.linalg.lstsq(A, th, rcond=None)
    a, b, c = coef
    if c >= 0:  # degenerate: no peak, fall back to a flat profile at the mean
        return IncidenceModel(0.0, float(th.mean()), 0.0, float(th.min()), cycle)
    lat_peak = -b / (2 * c)
    theta_peak = a + b * lat_peak + c * lat_peak**2
    return IncidenceModel(
        lat_peak_deg=float(lat_peak),
        theta_peak_deg=float(theta_peak),
        curvature_deg_per_deg2=float(-c),
        theta_min_deg=float(max(th.min(), 5.0)),
        cycle=cycle,
    )
