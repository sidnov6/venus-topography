/**
 * Sampling an elevation transect from whichever terrain provider is active — switch
 * between GTDR and ISHTAR and the same transect shows what the model added.
 *
 * Sample positions come from great-circle interpolation on the Venus sphere via
 * `EllipsoidGeodesic`, not from linear interpolation of longitude and latitude: the
 * latter bends visibly at high latitude and reports the wrong distance.
 */
import {
  Cartographic,
  EllipsoidGeodesic,
  sampleTerrainMostDetailed,
  type TerrainProvider,
} from "cesium";

import { VENUS } from "../venus.ts";
import type { ProfilePoint } from "./profile.ts";

export function interpolateGeodesic(
  start: Cartographic,
  end: Cartographic,
  samples: number,
): { positions: Cartographic[]; totalDistanceM: number } {
  const geodesic = new EllipsoidGeodesic(start, end, VENUS);
  const total = geodesic.surfaceDistance;
  const positions: Cartographic[] = [];
  for (let i = 0; i < samples; i++) {
    positions.push(geodesic.interpolateUsingFraction(i / (samples - 1)));
  }
  return { positions, totalDistanceM: total };
}

export async function sampleProfile(
  terrain: TerrainProvider,
  start: Cartographic,
  end: Cartographic,
  samples = 256,
): Promise<ProfilePoint[]> {
  const { positions, totalDistanceM } = interpolateGeodesic(start, end, samples);
  const sampled = await sampleTerrainMostDetailed(terrain, positions);
  return sampled.map((c, i) => ({
    distanceM: (i / (samples - 1)) * totalDistanceM,
    heightM: Number.isFinite(c.height) ? c.height : null,
    lon: (c.longitude * 180) / Math.PI,
    lat: (c.latitude * 180) / Math.PI,
  }));
}

