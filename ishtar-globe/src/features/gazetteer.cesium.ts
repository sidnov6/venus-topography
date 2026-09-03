/** Camera bindings for the gazetteer. Everything Cesium-shaped lives here so the
 * parsing and search in `gazetteer.ts` stay testable under plain Node. */
import { Cartesian3, Math as CesiumMath, Rectangle, type Viewer } from "cesium";

import { VENUS } from "../venus.ts";
import { frameDegrees, toCesiumLongitude, type Feature } from "./gazetteer.ts";

export function frameFor(f: Feature, margin = 1.8): Rectangle {
  const [west, south, east, north] = frameDegrees(f, margin);
  return Rectangle.fromDegrees(west, south, east, north);
}

export function flyToFeature(viewer: Viewer, f: Feature, duration = 2.5): void {
  viewer.camera.flyTo({ destination: frameFor(f), duration });
}

export function flyToPoint(viewer: Viewer, lon: number, lat: number, heightM = 900_000): void {
  viewer.camera.flyTo({
    destination: Cartesian3.fromDegrees(toCesiumLongitude(lon), lat, heightM, VENUS),
    orientation: { heading: 0, pitch: CesiumMath.toRadians(-55), roll: 0 },
    duration: 2.5,
  });
}
