/**
 * Longitude conversion and search. Node 22+ strips types natively, so these run with
 * `node --test` and no test framework — the globe's build already needs Node, and a
 * runner would be a dependency for four files of pure functions.
 */
import assert from "node:assert/strict";
import { test } from "node:test";

import { frameDegrees, parseGazetteer, search, toCesiumLongitude, type Feature } from "../gazetteer.ts";

const FEATURES: Feature[] = [
  { name: "Maxwell Montes", kind: "Mons", lon: 3.3, lat: 65.2, diameterKm: 853 },
  { name: "Maat Mons", kind: "Mons", lon: 194.6, lat: 0.5, diameterKm: 395 },
  { name: "Mead", kind: "Crater", lon: 57.2, lat: 12.5, diameterKm: 269 },
  { name: "Sacajawea Patera", kind: "Patera", lon: 335.0, lat: 64.5, diameterKm: 233 },
];

/** Degrees compare to a micro-degree — about 10 cm on Venus — so the float subtraction
 *  in the wrap does not turn an exact assertion into a flaky one. */
const near = (got: number, want: number, msg?: string) =>
  assert.ok(Math.abs(got - want) < 1e-6, msg ?? `expected ${want}, got ${got}`);

test("IAU longitudes are 0-360 east; Cesium wants -180..180", () => {
  near(toCesiumLongitude(0), 0);
  near(toCesiumLongitude(57.2), 57.2);
  near(toCesiumLongitude(180), 180);
  near(toCesiumLongitude(194.6), -165.4);
  near(toCesiumLongitude(335), -25);
  // Wrapping is idempotent, so a value that was already converted survives a second pass.
  near(toCesiumLongitude(toCesiumLongitude(335)), -25);
});

test("longitude conversion handles values outside 0-360", () => {
  near(toCesiumLongitude(-25), -25);
  near(toCesiumLongitude(720 + 90), 90);
});

test("search ranks prefix matches above interior ones", () => {
  const hits = search(FEATURES, "ma");
  assert.equal(hits[0]?.name, "Maat Mons");
  assert.ok(hits.some((f) => f.name === "Maxwell Montes"));
});

test("search ignores case and diacritics", () => {
  const withDiacritic: Feature[] = [
    { name: "Şäwtämä", kind: "Corona", lon: 10, lat: 10, diameterKm: 100 },
  ];
  assert.equal(search(withDiacritic, "sawtama").length, 1);
  assert.equal(search(FEATURES, "MEAD")[0]?.name, "Mead");
});

test("an empty query returns nothing rather than everything", () => {
  assert.deepEqual(search(FEATURES, "   "), []);
});

test("frameDegrees sizes the view to the feature, on the Venus sphere", () => {
  const [bw, , be] = frameDegrees(FEATURES[0]!);
  const [sw, , se] = frameDegrees(FEATURES[2]!);
  assert.ok(be - bw > se - sw, "Maxwell is larger than Mead");
  // 853 km on a 6051.8 km sphere is ~8 degrees; with margin, well under a quadrant.
  assert.ok(be - bw < 45);
  assert.ok(se - sw > 0.3);
});

test("frameDegrees clamps latitude away from the poles", () => {
  const [, south, , north] = frameDegrees({ name: "x", kind: "", lon: 0, lat: 89.9, diameterKm: 4000 });
  assert.ok(north <= 89.5, "a frame that runs past the pole breaks Rectangle");
  assert.ok(south < north);
});

test("frameDegrees converts the feature longitude exactly once", () => {
  const [west, , east] = frameDegrees(FEATURES[1]!); // Maat Mons at 194.6 E
  assert.ok(west < -160 && east > -170, `expected a frame around -165.4, got ${west}..${east}`);
});

test("parseGazetteer reads the IAU GeoJSON shape and sorts by name", () => {
  const parsed = parseGazetteer({
    features: [
      { properties: { name: "Zorya", feature_type: "Corona", diameter: 60 },
        geometry: { coordinates: [200, -10] } },
      { properties: { Name: "Aphrodite Terra", diameter_km: 10000 },
        geometry: { coordinates: [100, 0] } },
      { properties: { name: "no geometry" } },
    ],
  });
  assert.equal(parsed.length, 2, "an entry without coordinates is dropped, not defaulted");
  assert.equal(parsed[0]?.name, "Aphrodite Terra");
  assert.equal(parsed[1]?.diameterKm, 60);
});
