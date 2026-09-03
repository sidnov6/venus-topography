/**
 * Profile statistics. The slope numbers here are what a reader will quote off the screen,
 * so the arithmetic is worth pinning: metres over metres, converted once, with nodata
 * gaps skipped rather than treated as zero.
 */
import assert from "node:assert/strict";
import { test } from "node:test";

import { profileStats, profileSvgPath, type ProfilePoint } from "../profile.ts";

function ramp(n: number, lengthM: number, riseM: number): ProfilePoint[] {
  return Array.from({ length: n }, (_, i) => ({
    distanceM: (i / (n - 1)) * lengthM,
    heightM: (i / (n - 1)) * riseM,
    lon: 0,
    lat: 0,
  }));
}

test("a constant ramp reports its own slope", () => {
  const stats = profileStats(ramp(101, 10_000, 1000))!;
  assert.ok(stats);
  assert.ok(Math.abs(stats.meanSlopeDeg - (Math.atan(0.1) * 180) / Math.PI) < 1e-6);
  assert.ok(Math.abs(stats.maxSlopeDeg - stats.meanSlopeDeg) < 1e-6);
  assert.equal(stats.reliefM, 1000);
  assert.equal(stats.lengthKm, 10);
});

test("relief is max minus min, not first minus last", () => {
  const v: ProfilePoint[] = [
    { distanceM: 0, heightM: 0, lon: 0, lat: 0 },
    { distanceM: 1000, heightM: 800, lon: 0, lat: 0 },
    { distanceM: 2000, heightM: 0, lon: 0, lat: 0 },
  ];
  assert.equal(profileStats(v)!.reliefM, 800);
});

test("nodata points are skipped, not read as sea level", () => {
  const withGap: ProfilePoint[] = [
    { distanceM: 0, heightM: 500, lon: 0, lat: 0 },
    { distanceM: 1000, heightM: null, lon: 0, lat: 0 },
    { distanceM: 2000, heightM: 520, lon: 0, lat: 0 },
  ];
  const stats = profileStats(withGap)!;
  assert.equal(stats.minM, 500);
  assert.equal(stats.maxM, 520);
  // Venus has no sea level; a null read as 0 would invent a 500 m cliff.
  assert.ok(stats.maxSlopeDeg < 1);
});

test("a transect with fewer than two valid points reports nothing", () => {
  assert.equal(profileStats([]), null);
  assert.equal(profileStats([{ distanceM: 0, heightM: null, lon: 0, lat: 0 }]), null);
});

test("the sparkline path spans the box and starts with a move", () => {
  const path = profileSvgPath(ramp(21, 5000, 400), 360, 90);
  assert.ok(path.startsWith("M"));
  const xs = [...path.matchAll(/[ML]([\d.]+),/g)].map((m) => Number(m[1]));
  assert.ok(Math.min(...xs) === 0 && Math.max(...xs) === 360);
});

test("a flat transect still draws a path rather than dividing by zero", () => {
  const flat = Array.from({ length: 10 }, (_, i) => ({
    distanceM: i * 100, heightM: 300, lon: 0, lat: 0,
  }));
  const path = profileSvgPath(flat, 200, 50);
  assert.ok(path.includes("L"));
  assert.ok(!path.includes("NaN"));
});
