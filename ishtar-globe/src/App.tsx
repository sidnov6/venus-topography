import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Cartesian3,
  Cartographic,
  Color,
  DirectionalLight,
  Ellipsoid,
  Math as CesiumMath,
  ScreenSpaceEventHandler,
  ScreenSpaceEventType,
  Viewer,
  type ImageryLayer,
} from "cesium";
import "cesium/Build/Cesium/Widgets/widgets.css";

import { LAYERS, installLayers, loadManifest, type LayerId, type TileManifest } from "./layers.ts";
import { TERRAINS, assertVenusGeometry, loadTerrain, type TerrainId } from "./terrain.ts";
import { SITES, VENUS, VENUS_RADIUS_M } from "./venus.ts";
import { loadGazetteer, search, type Feature } from "./features/gazetteer.ts";
import { flyToFeature } from "./features/gazetteer.cesium.ts";
import { profileStats, profileSvgPath, type ProfilePoint } from "./features/profile.ts";
import { sampleProfile } from "./features/profile.cesium.ts";
import { attachSplitterDrag, disableSwipe, enableSwipe } from "./features/swipe.ts";
import { TOUR, runTour, type TourHandle, type TourStop } from "./features/tour.ts";
import { decode, writeHash, type ViewState } from "./features/urlState.ts";

type Mode = "explore" | "profile";

export function App() {
  const container = useRef<HTMLDivElement>(null);
  const splitter = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<Viewer | null>(null);
  const layersRef = useRef<Map<LayerId, ImageryLayer>>(new Map());
  const tourRef = useRef<TourHandle | null>(null);
  const clicksRef = useRef<Cartographic[]>([]);

  const initial = useMemo(() => decode(window.location.hash), []);
  const [terrainId, setTerrainId] = useState<TerrainId>(
    (initial.terrain as TerrainId) ?? "smooth",
  );
  const [exaggeration, setExaggeration] = useState(initial.exaggeration ?? 1);
  const [visible, setVisible] = useState<Record<string, boolean>>(() =>
    initial.layers
      ? Object.fromEntries(LAYERS.map((l) => [l.id, initial.layers!.includes(l.id)]))
      : Object.fromEntries(LAYERS.map((l) => [l.id, l.defaultOn])),
  );
  const [swipe, setSwipe] = useState(false);
  const [mode, setMode] = useState<Mode>("explore");
  const [profile, setProfile] = useState<ProfilePoint[] | null>(null);
  const [gazetteer, setGazetteer] = useState<Feature[]>([]);
  const [query, setQuery] = useState("");
  const [tourStop, setTourStop] = useState<TourStop | null>(null);
  const [manifest, setManifest] = useState<TileManifest | null>(null);

  // The tile manifest gates viewer construction: imagery providers take their
  // maximumLevel at construction time and it cannot be changed afterwards.
  useEffect(() => {
    void loadManifest().then(setManifest);
  }, []);

  // ---- viewer ----
  useEffect(() => {
    if (!container.current || viewerRef.current || manifest === null) return;
    assertVenusGeometry();

    const viewer = new Viewer(container.current, {
      // Every Earth default is off. Cesium otherwise reaches for Bing imagery and
      // Cesium World Terrain, both of which are Earth assets on a Venus globe.
      baseLayer: false,
      baseLayerPicker: false,
      geocoder: false,
      homeButton: false,
      sceneModePicker: false,
      navigationHelpButton: false,
      fullscreenButton: false,
      timeline: false,
      animation: false,
      infoBox: false,
      selectionIndicator: false,
      // Lets anything read the rendered frame — `canvas.toDataURL()` for saving an image
      // of the view, and `scripts/smoke.mjs` for the imagery-alignment check. WebGL
      // discards the drawing buffer after each frame otherwise, and a read comes back
      // empty with no error.
      contextOptions: { webgl: { preserveDrawingBuffer: true } },
    });
    viewerRef.current = viewer;
    layersRef.current = installLayers(viewer, manifest);

    const { scene } = viewer;
    scene.globe.baseColor = Color.fromCssColorString("#3a3026");
    scene.globe.enableLighting = true;
    scene.globe.showGroundAtmosphere = false;

    // Venus rotates once per 243 Earth days, so a physically placed sun is both static
    // and, for most camera positions, on the wrong side — the globe renders black. The
    // light is instead pinned to the camera with a fixed offset, which keeps relief
    // shaded from every viewpoint and gives the terrain switch something to show.
    const light = new DirectionalLight({ direction: Cartesian3.UNIT_X, intensity: 2.2 });
    scene.light = light;
    const aimLight = () => {
      const d = scene.camera.directionWC;
      const r = scene.camera.rightWC;
      const u = scene.camera.upWC;
      const dir = Cartesian3.clone(d, new Cartesian3());
      Cartesian3.add(dir, Cartesian3.multiplyByScalar(r, -0.55, new Cartesian3()), dir);
      Cartesian3.add(dir, Cartesian3.multiplyByScalar(u, -0.42, new Cartesian3()), dir);
      light.direction = Cartesian3.normalize(dir, new Cartesian3());
    };
    aimLight();
    const removeLightHook = scene.preRender.addEventListener(aimLight);
    if (scene.skyAtmosphere) {
      scene.skyAtmosphere.show = true;
      // A pale sulphur cast, not Earth's blue limb. Venus rotates once per 243 days, so
      // a real sun position would be static and dull; the light direction is fixed.
      scene.skyAtmosphere.hueShift = -0.08;
      scene.skyAtmosphere.saturationShift = -0.3;
      scene.skyAtmosphere.brightnessShift = 0.35;
    }
    scene.backgroundColor = Color.fromCssColorString("#05060a");
    viewer.cesiumWidget.creditContainer.setAttribute("style", "display:none");
    viewer.screenSpaceEventHandler.removeInputAction(ScreenSpaceEventType.LEFT_DOUBLE_CLICK);

    // Test hook, dev builds only: the smoke test needs to place the camera precisely
    // overhead, which no UI control does.
    if (import.meta.env.DEV) {
      (window as unknown as Record<string, unknown>).__ishtar = {
        viewer, VENUS, Cartesian3, CesiumMath,
      };
    }

    viewer.camera.setView({
      destination: Cartesian3.fromDegrees(
        initial.lon ?? 60,
        initial.lat ?? 0,
        initial.heightM ?? VENUS_RADIUS_M * 2.6,
        VENUS,
      ),
      orientation: {
        heading: CesiumMath.toRadians(initial.headingDeg ?? 0),
        pitch: CesiumMath.toRadians(initial.pitchDeg ?? -90),
        roll: 0,
      },
    });

    void loadGazetteer().then(setGazetteer);

    return () => {
      removeLightHook();
      viewer.destroy();
      viewerRef.current = null;
    };
    // `initial` is read once, deliberately: the hash seeds the view, it does not drive it.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [manifest]);

  // ---- terrain ----
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    const def = TERRAINS.find((t) => t.id === terrainId)!;
    let cancelled = false;
    void loadTerrain(def).then((provider) => {
      if (!cancelled && viewerRef.current) viewerRef.current.terrainProvider = provider;
    });
    return () => {
      cancelled = true;
    };
  }, [terrainId]);

  useEffect(() => {
    const viewer = viewerRef.current;
    if (viewer) viewer.scene.verticalExaggeration = exaggeration;
  }, [exaggeration]);

  useEffect(() => {
    for (const [id, layer] of layersRef.current) layer.show = visible[id] ?? false;
  }, [visible]);

  // ---- swipe: GTDR hillshade against the learned hillshade ----
  useEffect(() => {
    const viewer = viewerRef.current;
    const left = layersRef.current.get("hillshade");
    const right = layersRef.current.get("colour_relief");
    if (!viewer || !left || !right) return;
    if (swipe) {
      enableSwipe(viewer, { left, right });
      const handle = splitter.current;
      return handle ? attachSplitterDrag(viewer, handle) : undefined;
    }
    disableSwipe({ left, right });
    return undefined;
  }, [swipe]);

  // ---- profile: two clicks, then sample the active terrain ----
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || mode !== "profile") return;
    clicksRef.current = [];
    setProfile(null);

    const handler = new ScreenSpaceEventHandler(viewer.canvas);
    handler.setInputAction((click: { position: { x: number; y: number } }) => {
      const cartesian = viewer.scene.pickPosition(click.position as never);
      if (!cartesian) return;
      clicksRef.current.push(Cartographic.fromCartesian(cartesian, VENUS));
      if (clicksRef.current.length === 2) {
        const [a, b] = clicksRef.current;
        void sampleProfile(viewer.terrainProvider, a!, b!).then(setProfile);
        clicksRef.current = [];
      }
    }, ScreenSpaceEventType.LEFT_CLICK);

    return () => handler.destroy();
  }, [mode]);

  // ---- shareable URL ----
  const syncHash = useCallback(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    const c = Cartographic.fromCartesian(viewer.camera.positionWC, VENUS);
    const state: ViewState = {
      lon: Number(CesiumMath.toDegrees(c.longitude).toFixed(4)),
      lat: Number(CesiumMath.toDegrees(c.latitude).toFixed(4)),
      heightM: Math.round(c.height),
      headingDeg: Number(CesiumMath.toDegrees(viewer.camera.heading).toFixed(1)),
      pitchDeg: Number(CesiumMath.toDegrees(viewer.camera.pitch).toFixed(1)),
      layers: Object.entries(visible).filter(([, on]) => on).map(([id]) => id),
      terrain: terrainId,
      exaggeration,
    };
    writeHash(state);
  }, [visible, terrainId, exaggeration]);

  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    const remove = viewer.camera.moveEnd.addEventListener(syncHash);
    syncHash();
    return () => remove();
  }, [syncHash]);

  const results = useMemo(() => search(gazetteer, query), [gazetteer, query]);
  const stats = profile ? profileStats(profile) : null;

  const toggleTour = () => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    if (tourRef.current) {
      tourRef.current.stop();
      tourRef.current = null;
      setTourStop(null);
      return;
    }
    tourRef.current = runTour(viewer, (s) => setTourStop(s), TOUR);
  };

  return (
    <div style={{ position: "relative", height: "100vh", width: "100vw", overflow: "hidden" }}>
      <div ref={container} style={{ height: "100%", width: "100%" }} />
      {swipe && <div ref={splitter} style={splitterStyle} />}

      <aside style={panel}>
        <h1 style={{ margin: "0 0 2px", fontSize: 15, letterSpacing: 1.5 }}>ISHTAR</h1>
        <p style={caption}>
          Learned topography of Venus from Magellan SAR. A model-derived candidate, not a
          measurement — check the uncertainty layer before believing any relief.
        </p>

        <h2 style={heading}>Terrain</h2>
        {TERRAINS.map((t) => (
          <label key={t.id} style={row} title={t.blurb}>
            <input type="radio" name="terrain" checked={terrainId === t.id}
                   onChange={() => setTerrainId(t.id)} /> {t.label}
          </label>
        ))}
        <label style={{ ...row, marginTop: 6 }}>
          Exaggeration {exaggeration.toFixed(1)}x
          <input type="range" min={1} max={12} step={0.5} value={exaggeration}
                 onChange={(e) => setExaggeration(Number(e.target.value))}
                 style={{ width: "100%" }} />
        </label>

        <h2 style={heading}>Layers</h2>
        {LAYERS.map((l) => (
          <label key={l.id} style={row} title={l.blurb}>
            <input type="checkbox" checked={visible[l.id] ?? false}
                   onChange={(e) => setVisible((v) => ({ ...v, [l.id]: e.target.checked }))} />{" "}
            {l.label}
          </label>
        ))}
        <label style={row} title="Hillshade against colour relief, split down the middle.">
          <input type="checkbox" checked={swipe} onChange={(e) => setSwipe(e.target.checked)} />{" "}
          Compare (swipe)
        </label>

        <h2 style={heading}>Search</h2>
        <input value={query} onChange={(e) => setQuery(e.target.value)}
               placeholder={gazetteer.length ? "IAU feature name" : "gazetteer not loaded"}
               style={input} />
        {results.map((f) => (
          <button key={`${f.name}-${f.lon}`} style={button}
                  onClick={() => viewerRef.current && flyToFeature(viewerRef.current, f)}>
            {f.name} <span style={{ opacity: 0.5 }}>{f.kind}</span>
          </button>
        ))}
        {!gazetteer.length &&
          SITES.map((s) => (
            <button key={s.name} style={button} title={s.blurb}
                    onClick={() =>
                      viewerRef.current?.camera.flyTo({
                        destination: Cartesian3.fromDegrees(
                          s.lon > 180 ? s.lon - 360 : s.lon, s.lat, 900_000, VENUS,
                        ),
                        orientation: { heading: 0, pitch: CesiumMath.toRadians(-55), roll: 0 },
                        duration: 2.5,
                      })
                    }>
              {s.name}
            </button>
          ))}

        <h2 style={heading}>Tools</h2>
        <button style={button} onClick={() => setMode(mode === "profile" ? "explore" : "profile")}>
          {mode === "profile" ? "Profile: click two points…" : "Elevation profile"}
        </button>
        <button style={button} onClick={toggleTour}>
          {tourRef.current ? "Stop tour" : "Take the tour"}
        </button>

        <p style={{ ...caption, marginTop: 10 }}>
          Sphere radius {(VENUS_RADIUS_M / 1000).toFixed(1)} km ·{" "}
          {Ellipsoid.default === VENUS ? "Venus ellipsoid active" : "WGS84 — setup order bug"}
          {manifest?.synthetic && (
            <>
              <br />
              <strong style={{ color: "#e8b06a" }}>Synthetic demo tiles</strong> — generated
              terrain, not Magellan data.
            </>
          )}
        </p>
      </aside>

      {stats && profile && (
        <div style={profilePanel}>
          <svg width={360} height={90} style={{ display: "block" }}>
            <path d={profileSvgPath(profile, 360, 86)} fill="none" stroke="#e8c98a" strokeWidth={1.5} />
          </svg>
          <p style={caption}>
            {stats.lengthKm.toFixed(0)} km · relief {stats.reliefM.toFixed(0)} m ·
            mean slope {stats.meanSlopeDeg.toFixed(1)}° · max {stats.maxSlopeDeg.toFixed(1)}°
          </p>
        </div>
      )}

      {tourStop && (
        <div style={tourPanel}>
          <strong>{tourStop.name}</strong>
          <p style={{ ...caption, marginTop: 4 }}>{tourStop.caption}</p>
        </div>
      )}
    </div>
  );
}

const panel: React.CSSProperties = {
  position: "absolute", top: 16, left: 16, width: 250, maxHeight: "calc(100vh - 32px)",
  overflowY: "auto", padding: "14px 16px", background: "rgba(12,12,16,0.84)",
  backdropFilter: "blur(8px)", border: "1px solid rgba(230,200,140,0.18)",
  borderRadius: 8, color: "#e8e2d6",
  font: "13px/1.45 ui-sans-serif, system-ui, sans-serif",
};
const heading: React.CSSProperties = {
  margin: "14px 0 6px", fontSize: 11, letterSpacing: 1.2,
  textTransform: "uppercase", color: "#c9b48a",
};
const row: React.CSSProperties = { display: "block", margin: "3px 0", cursor: "pointer" };
const caption: React.CSSProperties = { margin: 0, fontSize: 11, color: "#9a948a" };
const button: React.CSSProperties = {
  display: "block", width: "100%", margin: "3px 0", padding: "5px 8px", textAlign: "left",
  background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)",
  borderRadius: 5, color: "#e8e2d6", font: "inherit", cursor: "pointer",
};
const input: React.CSSProperties = {
  width: "100%", padding: "5px 8px", background: "rgba(255,255,255,0.06)",
  border: "1px solid rgba(255,255,255,0.14)", borderRadius: 5, color: "#e8e2d6",
  font: "inherit", boxSizing: "border-box",
};
const splitterStyle: React.CSSProperties = {
  position: "absolute", top: 0, left: "50%", width: 4, height: "100%",
  marginLeft: -2, background: "rgba(230,200,140,0.5)", cursor: "col-resize", zIndex: 5,
};
const profilePanel: React.CSSProperties = {
  position: "absolute", bottom: 16, left: "50%", transform: "translateX(-50%)",
  padding: "10px 14px", background: "rgba(12,12,16,0.84)", backdropFilter: "blur(8px)",
  border: "1px solid rgba(230,200,140,0.18)", borderRadius: 8, color: "#e8e2d6",
  font: "13px/1.45 ui-sans-serif, system-ui, sans-serif",
};
const tourPanel: React.CSSProperties = {
  position: "absolute", bottom: 16, right: 16, width: 320, padding: "12px 14px",
  background: "rgba(12,12,16,0.84)", backdropFilter: "blur(8px)",
  border: "1px solid rgba(230,200,140,0.18)", borderRadius: 8, color: "#e8e2d6",
  font: "13px/1.45 ui-sans-serif, system-ui, sans-serif",
};
