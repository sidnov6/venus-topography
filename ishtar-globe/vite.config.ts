import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";
import cesium from "vite-plugin-cesium";

/**
 * Quantized-mesh tiles are stored gzipped, as the format expects, and CesiumJS does not
 * gunzip them itself — it relies on the transport. Without `Content-Encoding: gzip` the
 * browser hands Cesium the compressed bytes and every tile fails to parse, with nothing
 * in the console but a generic terrain error.
 *
 * Any production host serving these files needs the same header. So does the imagery
 * pyramid's `.terrain` sibling if you ever gzip the PNGs.
 */
function terrainEncoding(): Plugin {
  return {
    name: "ishtar-terrain-encoding",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (req.url?.includes(".terrain")) {
          res.setHeader("Content-Encoding", "gzip");
          res.setHeader("Content-Type", "application/octet-stream");
        }
        next();
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), cesium(), terrainEncoding()],
  server: { port: 5173 },
  // Tiles are static assets, not modules; keep Vite from trying to transform them.
  assetsInclude: ["**/*.terrain"],
});
