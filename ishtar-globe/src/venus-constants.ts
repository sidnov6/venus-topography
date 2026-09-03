/**
 * Planetary constants, with no Cesium import.
 *
 * `venus.ts` needs Cesium to build the ellipsoid and the tiling scheme, which makes it
 * unloadable outside a browser bundle. The numbers themselves are needed by pure logic
 * that is worth unit-testing, so they live here and `venus.ts` imports them.
 */

/** Venus is effectively a sphere; there is no meaningful flattening. */
export const VENUS_RADIUS_M = 6_051_800;

/** Magellan FMAP native posting, metres per pixel at the equator. */
export const FMAP_POSTING_M = 75;

/** Magellan GTDR altimetry grid, metres per post. */
export const GTDR_POSTING_M = 4641;
