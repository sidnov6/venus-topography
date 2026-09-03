/** Reading the camera out of a Viewer, for `urlState.encode`. */
import { Cartographic, Math as CesiumMath, type Viewer } from "cesium";

import type { ViewState } from "./urlState.ts";

export function readCamera(viewer: Viewer): Omit<ViewState, "layers" | "terrain" | "exaggeration"> {
  const c = Cartographic.fromCartesian(viewer.camera.positionWC, viewer.scene.globe.ellipsoid);
  return {
    lon: Number(CesiumMath.toDegrees(c.longitude).toFixed(4)),
    lat: Number(CesiumMath.toDegrees(c.latitude).toFixed(4)),
    heightM: Math.round(c.height),
    headingDeg: Number(CesiumMath.toDegrees(viewer.camera.heading).toFixed(1)),
    pitchDeg: Number(CesiumMath.toDegrees(viewer.camera.pitch).toFixed(1)),
  };
}

