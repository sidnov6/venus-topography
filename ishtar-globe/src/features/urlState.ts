/**
 * Camera and layer state in the URL hash, so a view can be shared as a link.
 *
 * Framework-free. Longitude, latitude and height are rounded to a sane precision — full
 * float64 in a URL is noise, and the resulting links are unreadable. `readCamera` needs
 * Cesium and lives in `urlState.cesium.ts`.
 */

export interface ViewState {
  lon: number;
  lat: number;
  heightM: number;
  headingDeg: number;
  pitchDeg: number;
  layers: string[];
  terrain: string;
  exaggeration: number;
}

export function encode(state: ViewState): string {
  const p = new URLSearchParams({
    lon: String(state.lon),
    lat: String(state.lat),
    h: String(state.heightM),
    hd: String(state.headingDeg),
    p: String(state.pitchDeg),
    t: state.terrain,
    x: String(state.exaggeration),
  });
  if (state.layers.length) p.set("l", state.layers.join(","));
  return `#${p.toString()}`;
}

export function decode(hash: string): Partial<ViewState> {
  const p = new URLSearchParams(hash.replace(/^#/, ""));
  const num = (k: string) => {
    const v = p.get(k);
    // `Number("")` is 0, not NaN, so an empty parameter would decode as a real value —
    // `h=` would put the camera on the surface rather than leaving the height alone.
    if (v === null || v.trim() === "") return undefined;
    const n = Number(v);
    return Number.isFinite(n) ? n : undefined;
  };
  const layers = p.get("l");
  return {
    lon: num("lon"),
    lat: num("lat"),
    heightM: num("h"),
    headingDeg: num("hd"),
    pitchDeg: num("p"),
    exaggeration: num("x"),
    terrain: p.get("t") ?? undefined,
    layers: layers ? layers.split(",").filter(Boolean) : undefined,
  };
}

export function writeHash(state: ViewState): void {
  const next = encode(state);
  if (next !== window.location.hash) {
    window.history.replaceState(null, "", next);
  }
}
