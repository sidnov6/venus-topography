/**
 * Headless smoke test for the globe.
 *
 *   npm run dev            # in one shell
 *   npx playwright install chromium   # once
 *   node scripts/smoke.mjs out.png    # in another
 *
 * Checks the things that are invisible in a screenshot and silent in the console:
 * the Venus ellipsoid is actually the default, terrain tiles load and parse, and no
 * request 404s. Cesium reports a terrain tile it cannot parse as a generic error with
 * no indication that the problem is a missing `Content-Encoding: gzip`, so the failed-
 * request count is the only reliable signal.
 *
 * Playwright is deliberately not a dependency of this package — it is a ~300 MB install
 * for a test that is not part of the build.
 */
import { chromium } from "playwright";

const URL = process.env.ISHTAR_URL ?? "http://localhost:5173/";
const OUT = process.argv[2] ?? "globe.png";

const browser = await chromium.launch({
  args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
});
const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });

const errors = [];
const failed = [];
page.on("pageerror", (e) => errors.push(String(e).slice(0, 200)));
page.on("response", (r) => {
  if (r.status() >= 400) failed.push(`${r.status()} ${r.url().slice(-70)}`);
});

await page.goto(URL, { waitUntil: "networkidle", timeout: 60_000 });
await page.waitForTimeout(4000);

const venus = await page.evaluate(() =>
  document.querySelector("aside")?.innerText?.includes("Venus ellipsoid active") ?? false,
);

await page.getByText("ISHTAR (learned)").click();
await page.getByText("Colour relief", { exact: true }).click();
await page.locator('input[type="range"]').fill("8");
await page.getByRole("button", { name: /Maxwell Montes/ }).click();
await page.waitForTimeout(9000);

const terrainRequests = await page.evaluate(() =>
  performance.getEntriesByType("resource").filter((r) => r.name.includes(".terrain")).length,
);

/**
 * Imagery alignment, the check from the globe work plan. The graticule layer draws a
 * marker at Mead crater from its published degrees; the camera flies to the same
 * degrees. If the imagery tiling scheme disagrees with the camera — a Web Mercator
 * scheme is the usual way — the marker lands somewhere else entirely, and nothing else
 * in this script would notice.
 */
await page.getByText("Graticule + landmarks").click();
await page.waitForTimeout(2000);

// Straight down over Mead crater. The UI's fly-to uses a 55-degree tilt, under which the
// target sits low in the frame by design, so the camera is placed directly through the
// dev-only test hook instead.
await page.evaluate(() => {
  const { viewer, VENUS, Cartesian3 } = window.__ishtar;
  viewer.camera.setView({
    destination: Cartesian3.fromDegrees(57.2, 12.5, 400_000, VENUS),
    orientation: { heading: 0, pitch: -Math.PI / 2, roll: 0 },
  });
});
await page.waitForTimeout(9000);

const alignment = await page.evaluate(() => {
  const canvas = document.querySelector("canvas");
  const gl = canvas.getContext("webgl2") ?? canvas.getContext("webgl");
  const w = canvas.width, h = canvas.height;
  const box = 300;                       // search window around screen centre
  const px = new Uint8Array(box * box * 4);
  gl.readPixels(Math.round((w - box) / 2), Math.round((h - box) / 2), box, box,
                gl.RGBA, gl.UNSIGNED_BYTE, px);
  // The marker is drawn in (255, 90, 60): strongly red-dominant, unlike the terrain ramp.
  let hits = 0, sumX = 0, sumY = 0;
  for (let i = 0; i < box * box; i++) {
    const r = px[i * 4], g = px[i * 4 + 1], b = px[i * 4 + 2];
    if (r > 150 && r - g > 60 && r - b > 60) {
      hits++; sumX += i % box; sumY += Math.floor(i / box);
    }
  }
  return hits ? { hits, dx: sumX / hits - box / 2, dy: sumY / hits - box / 2 } : { hits: 0 };
});

await page.screenshot({ path: OUT });
await browser.close();

const problems = [];
if (!venus) problems.push("the panel does not report the Venus ellipsoid as active");
if (terrainRequests === 0) problems.push("no terrain tiles were requested");
if (!alignment.hits) {
  problems.push("the Mead crater marker is not near screen centre: imagery and camera disagree");
} else if (Math.hypot(alignment.dx, alignment.dy) > 45) {
  problems.push(
    `the Mead marker is ${Math.hypot(alignment.dx, alignment.dy).toFixed(0)} px off centre`,
  );
}
if (failed.length) problems.push(`${failed.length} failed requests: ${failed.slice(0, 3).join(", ")}`);
if (errors.length) problems.push(`${errors.length} page errors: ${errors.slice(0, 2).join(", ")}`);

console.log(
  JSON.stringify({
    venusEllipsoid: venus,
    terrainRequests,
    markerOffsetPx: alignment.hits
      ? Number(Math.hypot(alignment.dx, alignment.dy).toFixed(1))
      : null,
    failed: failed.length,
    errors: errors.length,
    screenshot: OUT,
  }, null, 2),
);
if (problems.length) {
  console.error("\nFAIL:\n  " + problems.join("\n  "));
  process.exit(1);
}
console.log("\nOK");
