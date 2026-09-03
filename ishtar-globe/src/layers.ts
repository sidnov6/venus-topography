/**
 * Imagery layers. Every provider is given the Venus geodetic tiling scheme
 * explicitly; none of them may fall back to a Cesium ion or Bing asset.
 */
import {
  ImageryLayer,
  UrlTemplateImageryProvider,
  type Viewer,
} from "cesium";

import { venusTilingScheme } from "./venus.ts";

export type LayerId =
  | "sar_left"
  | "sar_right"
  | "colour_relief"
  | "hillshade"
  | "uncertainty"
  | "stereo_coverage"
  | "emissivity"
  | "graticule";

export interface LayerDef {
  id: LayerId;
  label: string;
  url: string;
  maximumLevel: number;
  defaultAlpha: number;
  defaultOn: boolean;
  blurb: string;
}

/** Level 9 is ~580 m/px at the equator; level 12 is ~73 m/px, ROIs only. */
export const LAYERS: readonly LayerDef[] = [
  {
    id: "sar_left",
    label: "SAR (left look)",
    url: "/tiles/sar_left/{z}/{x}/{reverseY}.png",
    maximumLevel: 12,
    defaultAlpha: 1,
    defaultOn: true,
    blurb: "Magellan FMAP left-look mosaic, 75 m. The model's primary input.",
  },
  {
    id: "sar_right",
    label: "SAR (right look)",
    url: "/tiles/sar_right/{z}/{x}/{reverseY}.png",
    maximumLevel: 12,
    defaultAlpha: 0,
    defaultOn: false,
    blurb: "~17% coverage. Where it exists, the model had a second look.",
  },
  {
    id: "colour_relief",
    label: "Colour relief",
    url: "/tiles/colour_relief/{z}/{x}/{reverseY}.png",
    maximumLevel: 9,
    defaultAlpha: 0.7,
    defaultOn: false,
    blurb: "Predicted elevation, hypsometric ramp.",
  },
  {
    id: "hillshade",
    label: "Hillshade",
    url: "/tiles/hillshade/{z}/{x}/{reverseY}.png",
    maximumLevel: 9,
    defaultAlpha: 0.5,
    defaultOn: false,
    blurb: "Shaded relief of the predicted DEM.",
  },
  {
    id: "uncertainty",
    label: "Uncertainty (1σ)",
    url: "/tiles/uncertainty/{z}/{x}/{reverseY}.png",
    maximumLevel: 9,
    defaultAlpha: 0.6,
    defaultOn: false,
    blurb: "Per-pixel 1σ from the model's variance head. Grey means do not trust the relief.",
  },
  {
    id: "stereo_coverage",
    label: "Stereo coverage",
    url: "/tiles/stereo_coverage/{z}/{x}/{reverseY}.png",
    maximumLevel: 9,
    defaultAlpha: 0.45,
    defaultOn: false,
    blurb: "Where a stereo DEM constrained training — roughly 20% of the planet.",
  },
  {
    id: "graticule",
    label: "Graticule + landmarks",
    url: "/tiles/graticule/{z}/{x}/{reverseY}.png",
    maximumLevel: 9,
    defaultAlpha: 0.9,
    defaultOn: false,
    blurb:
      "10-degree grid with markers at the named sites. The alignment check: the marker " +
      "is drawn from degrees, the camera flies to degrees, the terrain is tiled from " +
      "degrees. If they disagree, they separate here.",
  },
  {
    id: "emissivity",
    label: "Emissivity (GEDR)",
    url: "/tiles/emissivity/{z}/{x}/{reverseY}.png",
    maximumLevel: 9,
    defaultAlpha: 0.5,
    defaultOn: false,
    blurb: "Magellan emissivity; the high-emissivity highlands behave differently in radar.",
  },
] as const;

/**
 * Optional `tiles/manifest.json`, written by `export/demo_tiles.py`.
 *
 * The pyramid that actually exists on disk is usually shallower than the product plan —
 * the demo set stops at level 5, and a regional 75 m product only reaches level 12 inside
 * its ROIs. Without this, Cesium keeps requesting levels that are not there and the
 * console fills with 404s that look like a bug.
 */
export interface TileManifest {
  synthetic?: boolean;
  maxLevel?: Partial<Record<LayerId, number>>;
  terrainMaxLevel?: Record<string, number>;
  note?: string;
}

export async function loadManifest(url = "/tiles/manifest.json"): Promise<TileManifest> {
  try {
    const res = await fetch(url);
    return res.ok ? ((await res.json()) as TileManifest) : {};
  } catch {
    return {};
  }
}

export function makeProvider(def: LayerDef, manifest: TileManifest = {}): UrlTemplateImageryProvider {
  return new UrlTemplateImageryProvider({
    url: def.url,
    tilingScheme: venusTilingScheme(),
    maximumLevel: manifest.maxLevel?.[def.id] ?? def.maximumLevel,
    // The pyramid is sparse outside the ROIs; a missing tile is normal, not an error.
    hasAlphaChannel: true,
  });
}

export function installLayers(
  viewer: Viewer,
  manifest: TileManifest = {},
): Map<LayerId, ImageryLayer> {
  const layers = new Map<LayerId, ImageryLayer>();
  viewer.imageryLayers.removeAll();
  for (const def of LAYERS) {
    const layer = viewer.imageryLayers.addImageryProvider(makeProvider(def, manifest));
    layer.alpha = def.defaultAlpha;
    layer.show = def.defaultOn;
    layers.set(def.id, layer);
  }
  return layers;
}
