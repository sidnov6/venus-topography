/**
 * Elevation profile: the statistics and the sparkline.
 *
 * Framework-free, so `__tests__/profile.test.ts` can check the slope arithmetic under
 * plain Node. The geodesic interpolation and terrain sampling need Cesium and live in
 * `profile.cesium.ts`.
 */

export interface ProfilePoint {
  /** Distance along the transect from the start, metres. */
  distanceM: number;
  /** Elevation above the reference sphere, metres. Null where terrain has no data. */
  heightM: number | null;
  lon: number;
  lat: number;
}

export interface ProfileStats {
  minM: number;
  maxM: number;
  reliefM: number;
  lengthKm: number;
  meanSlopeDeg: number;
  maxSlopeDeg: number;
}

export function profileStats(points: readonly ProfilePoint[]): ProfileStats | null {
  const valid = points.filter((p): p is ProfilePoint & { heightM: number } => p.heightM !== null);
  if (valid.length < 2) return null;

  const heights = valid.map((p) => p.heightM);
  const slopes: number[] = [];
  for (let i = 1; i < valid.length; i++) {
    const dx = valid[i]!.distanceM - valid[i - 1]!.distanceM;
    if (dx > 0) slopes.push(Math.abs(Math.atan((valid[i]!.heightM - valid[i - 1]!.heightM) / dx)));
  }
  const toDeg = (r: number) => (r * 180) / Math.PI;
  const min = Math.min(...heights);
  const max = Math.max(...heights);
  return {
    minM: min,
    maxM: max,
    reliefM: max - min,
    lengthKm: (valid[valid.length - 1]!.distanceM - valid[0]!.distanceM) / 1000,
    meanSlopeDeg: toDeg(slopes.reduce((a, b) => a + b, 0) / Math.max(slopes.length, 1)),
    maxSlopeDeg: toDeg(Math.max(0, ...slopes)),
  };
}

/** Minimal inline SVG chart — no plotting dependency for one sparkline. */
export function profileSvgPath(
  points: readonly ProfilePoint[],
  width: number,
  height: number,
): string {
  const valid = points.filter((p): p is ProfilePoint & { heightM: number } => p.heightM !== null);
  if (valid.length < 2) return "";
  const xs = valid.map((p) => p.distanceM);
  const ys = valid.map((p) => p.heightM);
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs);
  const y0 = Math.min(...ys);
  const y1 = Math.max(...ys);
  const sx = (v: number) => ((v - x0) / Math.max(x1 - x0, 1)) * width;
  const sy = (v: number) => height - ((v - y0) / Math.max(y1 - y0, 1)) * height;
  return valid.map((p, i) => `${i === 0 ? "M" : "L"}${sx(p.distanceM).toFixed(1)},${sy(p.heightM).toFixed(1)}`).join(" ");
}
