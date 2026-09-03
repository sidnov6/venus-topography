/**
 * IAU nomenclature: parsing, search, and the one place longitude is converted.
 *
 * Framework-free on purpose. The gazetteer ships as GeoJSON from
 * planetarynames.wr.usgs.gov with longitudes in 0-360 east; Cesium wants -180..180, and
 * getting that wrong puts every feature on the wrong side of the planet — which looks
 * like a data problem rather than a conversion one. Keeping the conversion here, with no
 * Cesium import, is what lets `__tests__/gazetteer.test.ts` check it under plain Node.
 * The camera bindings live in `gazetteer.cesium.ts`.
 */
import { VENUS_RADIUS_M } from "../venus-constants.ts";

export interface Feature {
  name: string;
  /** Feature type: Mons, Corona, Tessera, Chasma, Crater, Regio, Planitia... */
  kind: string;
  /** Degrees east, 0..360 as published. */
  lon: number;
  /** Planetocentric latitude. */
  lat: number;
  /** Approximate diameter in km; drives how far out the camera stops. */
  diameterKm: number;
}

export function toCesiumLongitude(lonEast360: number): number {
  const wrapped = ((lonEast360 % 360) + 360) % 360;
  return wrapped > 180 ? wrapped - 360 : wrapped;
}

/** Parse the IAU GeoJSON export into the shape the UI uses. */
export function parseGazetteer(geojson: unknown): Feature[] {
  const fc = geojson as {
    features?: Array<{
      properties?: Record<string, unknown>;
      geometry?: { coordinates?: [number, number] };
    }>;
  };
  const out: Feature[] = [];
  for (const f of fc.features ?? []) {
    const p = f.properties ?? {};
    const coords = f.geometry?.coordinates;
    const name = typeof p.name === "string" ? p.name : typeof p.Name === "string" ? p.Name : null;
    if (!name || !coords) continue;
    out.push({
      name,
      kind: String(p.feature_type ?? p.type ?? ""),
      lon: Number(coords[0]),
      lat: Number(coords[1]),
      diameterKm: Number(p.diameter ?? p.diameter_km ?? 100) || 100,
    });
  }
  return out.sort((a, b) => a.name.localeCompare(b.name));
}

export async function loadGazetteer(url = "/gazetteer/venus.geojson"): Promise<Feature[]> {
  // The file is not committed (see public/gazetteer/README.md). A dev server answers a
  // missing path with index.html rather than a 404, so checking `res.ok` is not enough —
  // without the content-type check this throws a JSON parse error on every startup.
  try {
    const res = await fetch(url);
    if (!res.ok) return [];
    if (!(res.headers.get("content-type") ?? "").includes("json")) return [];
    return parseGazetteer(await res.json());
  } catch {
    return [];
  }
}

/** Case- and diacritic-insensitive prefix-then-substring ranking. */
export function search(features: readonly Feature[], query: string, limit = 12): Feature[] {
  const q = query.trim().toLowerCase();
  if (!q) return [];
  const norm = (s: string) => s.toLowerCase().normalize("NFD").replace(/\p{Diacritic}/gu, "");
  const scored: Array<[number, Feature]> = [];
  for (const f of features) {
    const n = norm(f.name);
    const i = n.indexOf(norm(q));
    if (i < 0) continue;
    scored.push([i === 0 ? 0 : 1 + i, f]);
  }
  scored.sort((a, b) => a[0] - b[0] || a[1].name.localeCompare(b[1].name));
  return scored.slice(0, limit).map(([, f]) => f);
}

/** Degrees framing a feature of the given diameter, with margin: `[west, south, east, north]`. */
export function frameDegrees(f: Feature, margin = 1.8): [number, number, number, number] {
  const halfDeg = Math.max(
    0.15,
    ((f.diameterKm * 1000 * margin) / 2 / VENUS_RADIUS_M) * (180 / Math.PI),
  );
  const lon = toCesiumLongitude(f.lon);
  return [
    lon - halfDeg,
    Math.max(-89.5, f.lat - halfDeg),
    lon + halfDeg,
    Math.min(89.5, f.lat + halfDeg),
  ];
}
