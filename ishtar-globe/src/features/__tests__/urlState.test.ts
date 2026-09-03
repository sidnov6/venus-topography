/**
 * Shareable links. A URL that decodes to a different view than it encoded is worse than
 * no URL at all, so the round trip is the whole test.
 */
import assert from "node:assert/strict";
import { test } from "node:test";

import { decode, encode, type ViewState } from "../urlState.ts";

const VIEW: ViewState = {
  lon: 57.2,
  lat: 12.5,
  heightM: 400_000,
  headingDeg: 0,
  pitchDeg: -90,
  layers: ["sar_left", "graticule"],
  terrain: "ishtar",
  exaggeration: 8,
};

test("a view round-trips through the hash", () => {
  assert.deepEqual(decode(encode(VIEW)), VIEW);
});

test("a view with no layers round-trips too", () => {
  const bare = { ...VIEW, layers: [] };
  const back = decode(encode(bare));
  assert.deepEqual(back.layers, undefined, "an absent list decodes as absent, not as ['']");
  assert.equal(back.terrain, "ishtar");
});

test("an empty or absent hash yields no opinions", () => {
  const back = decode("");
  assert.equal(back.lon, undefined);
  assert.equal(back.terrain, undefined);
  assert.equal(back.layers, undefined);
});

test("junk in the hash is ignored rather than becoming NaN", () => {
  const back = decode("#lon=abc&lat=12.5&h=&x=3");
  assert.equal(back.lon, undefined, "a NaN longitude would send the camera to the origin");
  assert.equal(back.lat, 12.5);
  assert.equal(back.heightM, undefined);
  assert.equal(back.exaggeration, 3);
});

test("negative longitudes and pitches survive the round trip", () => {
  const west: ViewState = { ...VIEW, lon: -165.4, pitchDeg: -55.3, headingDeg: 271.5 };
  const back = decode(encode(west));
  assert.equal(back.lon, -165.4);
  assert.equal(back.pitchDeg, -55.3);
  assert.equal(back.headingDeg, 271.5);
});

test("the hash is a readable query string, not an opaque blob", () => {
  const hash = encode(VIEW);
  assert.ok(hash.startsWith("#"));
  assert.ok(hash.includes("lon=57.2"));
  assert.ok(hash.includes("t=ishtar"));
});
