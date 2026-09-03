/**
 * Entry point.
 *
 * `installVenusEllipsoid()` runs at module scope and `App` is loaded dynamically
 * *after* it. That ordering is the point: CesiumJS reads `Ellipsoid.default` when an
 * object is constructed, so anything built before the assignment is permanently on
 * WGS84 and every later `ellipsoid: VENUS` argument silently disagrees with it. A
 * static import of `App` would be hoisted above this call.
 */
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { installVenusEllipsoid } from "./venus.ts";

installVenusEllipsoid();

void import("./App").then(({ App }) => {
  createRoot(document.getElementById("root")!).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
});
