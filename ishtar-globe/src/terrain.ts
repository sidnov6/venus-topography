/**
 * Terrain on a non-WGS84 body.
 *
 * `CesiumTerrainProvider` decodes quantized-mesh heights correctly once it is given
 * the Venus ellipsoid, but most tilers (Cesium Terrain Builder and its forks) compute
 * each tile's bounding sphere and horizon occlusion point assuming WGS84. Those are
 * culling metadata: wrong values do not corrupt the mesh, they make tiles vanish at
 * the limb or refuse to refine. Verify at the limb before building anything on top.
 *
 * Route A (correct) is a tiler that computes bounds on the Venus sphere.
 * Route B (fallback) keeps a WGS84 tiler and pre-scales elevations by
 * `EARTH_OVER_VENUS` so relative relief survives on an Earth-sized globe. It must be
 * labelled in the UI, because absolute elevations are then wrong by 5%.
 */
import { CesiumTerrainProvider, EllipsoidTerrainProvider, type TerrainProvider } from "cesium";

import { VENUS, VENUS_RADIUS_M } from "./venus.ts";

export const EARTH_RADIUS_M = 6_371_000;
export const EARTH_OVER_VENUS = EARTH_RADIUS_M / VENUS_RADIUS_M; // ~1.0527

export type TerrainId = "smooth" | "gtdr" | "ishtar";

export interface TerrainDef {
  id: TerrainId;
  label: string;
  url?: string;
  blurb: string;
}

export const TERRAINS: readonly TerrainDef[] = [
  { id: "smooth", label: "Sphere (no terrain)", blurb: "The 6051.8 km reference sphere." },
  {
    id: "gtdr",
    label: "Magellan altimetry (GTDR)",
    url: "/tiles/terrain_gtdr",
    blurb: "Bicubic 4641 m altimetry — the baseline the model has to beat.",
  },
  {
    id: "ishtar",
    label: "ISHTAR (learned)",
    url: "/tiles/terrain_ishtar",
    blurb: "Model-derived candidate topography. Not a measurement — check the uncertainty layer.",
  },
] as const;

export async function loadTerrain(def: TerrainDef): Promise<TerrainProvider> {
  if (!def.url) return new EllipsoidTerrainProvider({ ellipsoid: VENUS });
  return CesiumTerrainProvider.fromUrl(def.url, {
    ellipsoid: VENUS,
    requestVertexNormals: true,
  });
}

/**
 * Acceptance check for step 1 of the globe work plan: a point on the surface must be
 * one Venus radius from the centre. If this fails, `Ellipsoid.default` was set after
 * something already captured the WGS84 default.
 */
export function assertVenusGeometry(): void {
  const magnitude = VENUS.maximumRadius;
  if (Math.abs(magnitude - VENUS_RADIUS_M) > 1) {
    throw new Error(
      `Ellipsoid is not Venus (radius ${magnitude} m). installVenusEllipsoid() must run ` +
        "before any Viewer, tiling scheme or Cartesian3.fromDegrees call.",
    );
  }
}
