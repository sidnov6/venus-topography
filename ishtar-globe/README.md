# ishtar-globe

The Venus globe: a CesiumJS app that serves ISHTAR's learned topography, the Magellan
SAR mosaics, and the model's uncertainty on the 6051.8 km sphere.

```bash
npm install
npm run dev          # http://localhost:5173
```

With no tiles present the globe renders an empty Venus sphere. To get something to look
at, build the synthetic demo set (nothing here needs the 300 GB of Magellan products):

```bash
cd ../ishtar-ml
python -m export.demo_tiles --out ../ishtar-globe/public/tiles \
    --max-level 5 --roi maxwell mead --roi-level 8
```

That writes every layer the panel offers plus both terrain pyramids, global to
`--max-level` and deeper inside the named regions — the same sparse arrangement the real
product uses, because level 12 globally would be 33 million tiles. About 110 MB at level 4
plus the regions; four times that per extra global level. It is generated, not measured,
and the panel says so on screen.

## The three rules

Everything specific to this app follows from Venus not being Earth. They are restated in
the repo root `CLAUDE.md` because an agent trained on Earth-centric Cesium examples will
regress on all three.

1. **`Ellipsoid.default` is set before any Cesium object exists.** `src/main.tsx` calls
   `installVenusEllipsoid()` at module scope and then imports `App` *dynamically* — a
   static import would be hoisted above the assignment, and CesiumJS reads the default at
   construction time, so the Viewer would be permanently WGS84.
2. **The tiling scheme is geodetic, not Web Mercator.** Tiles come from
   `gdal2tiles.py --profile=geodetic` and `export/quantized_mesh.py`, both 2×1 at level 0.
3. **No Earth assets.** No Bing, no Cesium World Terrain, no `createWorldTerrainAsync`,
   no ion token path. `baseLayer: false` and every provider is constructed explicitly.

## Two things that are easy to get wrong

**Terrain tiles need `Content-Encoding: gzip`.** Quantized-mesh files are stored gzipped
and CesiumJS does not decompress them itself. Without the header the browser hands Cesium
compressed bytes and every tile fails to parse, reported as a generic terrain error with
no hint at the cause. `vite.config.ts` sets it for the dev server; a production host needs
the same.

**Lighting cannot use a real sun.** Venus rotates once per 243 Earth days, so a physically
placed sun is static and, from most viewpoints, on the far side — the globe renders black.
`App.tsx` pins a `DirectionalLight` to the camera with a fixed offset instead, which keeps
relief shaded from every angle.

## Layout

```
src/venus-constants.ts  the planetary numbers, with no Cesium import
src/venus.ts            ellipsoid, geodetic tiling scheme, sites, level->metres
src/layers.ts           imagery layers and the tile manifest that clamps their levels
                        (SAR left/right, colour relief, hillshade, uncertainty fog,
                         stereo coverage, emissivity, graticule + landmarks)
src/terrain.ts          non-WGS84 terrain, Route A and the Route B fallback
src/features/
  gazetteer.ts          IAU nomenclature search; 0-360 -> -180..180 in one place
  profile.ts            transect statistics and the sparkline
  urlState.ts           camera and layers in the hash, for shareable links
  *.cesium.ts           the camera and terrain bindings for the three modules above
  swipe.ts              hillshade against colour relief via SplitDirection
  tour.ts               camera path through six sites, with what the model claims there
  __tests__/            21 tests, run by `node --test` with no dependencies
scripts/smoke.mjs       headless check: Venus ellipsoid, terrain parses, nothing 404s
```

## Tests

```bash
npm run typecheck
npm test          # node --test, no framework
npm run build
```

Node 22+ strips TypeScript natively, so `npm test` runs the source directly with no build
step and no test runner in `package.json`. That only works because the framework-free
logic is kept apart from the Cesium bindings: `gazetteer.ts`, `profile.ts` and
`urlState.ts` import nothing from Cesium, and their camera and terrain halves live in
`*.cesium.ts` siblings. The split is what makes longitude conversion, slope arithmetic
and URL round-tripping testable at all.

It has already earned itself: `decode("#h=")` used to return height 0 rather than
"unset", because `Number("")` is 0, which would have put a shared link's camera on the
surface.

## Smoke test

```bash
npm run dev
npx playwright install chromium     # once; playwright is not a dependency
npm run smoke -- globe.png
```

It fails if the ellipsoid is not Venus, if no terrain tile is requested, if anything
404s, or if the imagery is misaligned — the four failures a screenshot cannot show you.

The alignment check is the one from the globe work plan. The `graticule` layer draws a
marker at Mead crater from its published coordinates (57.2 E, 12.5 N); the test places the
camera exactly overhead at those coordinates and reads the framebuffer. If the tile
pyramid and the camera disagree about what those degrees mean — a Web Mercator tiling
scheme is the usual way — the marker is nowhere near the centre. It currently lands
**1.5 px** from centre.

Two things make that check possible and are there for it: the Viewer is constructed with
`preserveDrawingBuffer` (WebGL otherwise discards the frame and the read comes back empty
with no error), and dev builds expose `window.__ishtar` so the test can place the camera
precisely, which no UI control does.
