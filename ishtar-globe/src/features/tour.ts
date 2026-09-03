/**
 * Cinematic tour: a chain of camera flights through the sites worth looking at.
 *
 * Each stop states what the model is claiming there, because the interesting part of a
 * learned DEM is where it disagrees with the altimetry and how confident it is about
 * the disagreement.
 */
import { Cartesian3, Math as CesiumMath, type Viewer } from "cesium";

import { VENUS } from "../venus.ts";

export interface TourStop {
  name: string;
  lon: number;
  lat: number;
  heightM: number;
  pitchDeg: number;
  headingDeg: number;
  durationS: number;
  holdS: number;
  caption: string;
}

export const TOUR: readonly TourStop[] = [
  {
    name: "Maxwell Montes",
    lon: 3.3, lat: 65.2, heightM: 420_000, pitchDeg: -32, headingDeg: 20,
    durationS: 4, holdS: 5,
    caption: "11 km of relief. Slopes here exceed the incidence angle, so the physics loss is masked out and the stereo and altimetry terms carry the shape alone.",
  },
  {
    name: "Ovda Regio",
    lon: 85.6, lat: -2.8, heightM: 620_000, pitchDeg: -40, headingDeg: 0,
    durationS: 5, holdS: 5,
    caption: "Tessera: the most deformed terrain on the planet, and a held-out validation region. Nothing here was trained on.",
  },
  {
    name: "Artemis Corona",
    lon: 135.0, lat: -35.0, heightM: 1_400_000, pitchDeg: -50, headingDeg: 0,
    durationS: 5, holdS: 5,
    caption: "2100 km across. The rim is a long-wavelength feature the altimetry already resolves; the model's contribution is the texture inside it.",
  },
  {
    name: "Mead crater",
    lon: 57.2, lat: 12.5, heightM: 300_000, pitchDeg: -45, headingDeg: 0,
    durationS: 4, holdS: 4,
    caption: "The imagery alignment check: if the SAR layer and the terrain disagree here, the tiling scheme is wrong.",
  },
  {
    name: "Maat Mons",
    lon: 194.6, lat: 0.5, heightM: 500_000, pitchDeg: -30, headingDeg: 300,
    durationS: 5, holdS: 5,
    caption: "8 km shield volcano. A demo site, deliberately kept out of the test set.",
  },
  {
    name: "Alpha Regio domes",
    lon: 11.8, lat: -25.5, heightM: 180_000, pitchDeg: -25, headingDeg: 90,
    durationS: 5, holdS: 6,
    caption: "Pancake domes, ~25 km across and under a kilometre high. Below the altimeter footprint entirely — this is what the model exists to recover.",
  },
] as const;

export interface TourHandle {
  stop: () => void;
}

export function runTour(
  viewer: Viewer,
  onStop: (stop: TourStop, index: number) => void,
  stops: readonly TourStop[] = TOUR,
): TourHandle {
  let cancelled = false;
  let timer: ReturnType<typeof setTimeout> | undefined;

  const go = (i: number) => {
    if (cancelled || i >= stops.length) return;
    const s = stops[i]!;
    onStop(s, i);
    viewer.camera.flyTo({
      destination: Cartesian3.fromDegrees(
        s.lon > 180 ? s.lon - 360 : s.lon, s.lat, s.heightM, VENUS,
      ),
      orientation: {
        heading: CesiumMath.toRadians(s.headingDeg),
        pitch: CesiumMath.toRadians(s.pitchDeg),
        roll: 0,
      },
      duration: s.durationS,
      complete: () => {
        if (!cancelled) timer = setTimeout(() => go(i + 1), s.holdS * 1000);
      },
    });
  };

  go(0);
  return {
    stop: () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
      viewer.camera.cancelFlight();
    },
  };
}
