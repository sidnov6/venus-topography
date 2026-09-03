/**
 * Venus constants and the one piece of setup order that matters.
 *
 * `Ellipsoid.default` must be assigned before anything constructs a `Viewer`, a
 * tiling scheme, or a `Cartesian3.fromDegrees`. CesiumJS reads the default at
 * construction time, so a Viewer built first is permanently on WGS84 and every
 * subsequent `ellipsoid: VENUS` argument silently disagrees with it.
 */
import {
  Cartesian3,
  Ellipsoid,
  GeographicTilingScheme,
  Rectangle,
} from "cesium";

export { VENUS_RADIUS_M } from "./venus-constants.ts";

import { VENUS_RADIUS_M } from "./venus-constants.ts";

export const VENUS = Ellipsoid.fromCartesian3(
  new Cartesian3(VENUS_RADIUS_M, VENUS_RADIUS_M, VENUS_RADIUS_M),
);

let installed = false;

/** Call once, at module load, before creating the Viewer. */
export function installVenusEllipsoid(): void {
  if (installed) return;
  Ellipsoid.default = VENUS;
  installed = true;
}

/**
 * Geodetic (2x1) tiling scheme on the Venus sphere.
 *
 * Not Web Mercator: the tiles are produced by `gdal2tiles.py --profile=geodetic`,
 * and Cesium's default `WebMercatorTilingScheme` would sample them at the wrong
 * latitudes with no visible error other than features landing in the wrong place.
 */
export function venusTilingScheme(): GeographicTilingScheme {
  return new GeographicTilingScheme({ ellipsoid: VENUS });
}

/** Metres per pixel at the equator for a geodetic tile pyramid level. */
export function metresPerPixelAtLevel(level: number, tileSize = 256): number {
  const degreesPerTile = 180 / 2 ** level;
  return ((degreesPerTile / tileSize) * Math.PI * VENUS_RADIUS_M) / 180;
}

/** Rectangle from degrees on the Venus ellipsoid. Longitudes may be 0..360. */
export function venusRectangle(
  west: number,
  south: number,
  east: number,
  north: number,
): Rectangle {
  const wrap = (lon: number) => (lon > 180 ? lon - 360 : lon);
  return Rectangle.fromDegrees(wrap(west), south, wrap(east), north);
}

/** Places of interest, from the IAU gazetteer. Demo sites, not test regions. */
export const SITES: ReadonlyArray<{
  name: string;
  lon: number;
  lat: number;
  blurb: string;
}> = [
  { name: "Maxwell Montes", lon: 3.3, lat: 65.2, blurb: "Highest relief on Venus, ~11 km above mean radius." },
  { name: "Maat Mons", lon: 194.6, lat: 0.5, blurb: "Large shield volcano, ~8 km high." },
  { name: "Ovda Regio", lon: 85.6, lat: -2.8, blurb: "Tessera: the most deformed terrain type." },
  { name: "Artemis Corona", lon: 135.0, lat: -35.0, blurb: "Largest corona, ~2100 km across." },
  { name: "Mead Crater", lon: 57.2, lat: 12.5, blurb: "Largest impact crater; the imagery alignment check." },
  { name: "Alpha Regio", lon: 5.0, lat: -25.0, blurb: "Tessera and pancake domes." },
];
