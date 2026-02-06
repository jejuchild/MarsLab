import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],

  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          // Separate heavy vendor libraries into their own cached chunks
          if (id.includes("node_modules/cesium")) return "vendor-cesium";
          if (id.includes("node_modules/three") || id.includes("node_modules/@react-three")) return "vendor-three";
          if (id.includes("node_modules/recharts") || id.includes("node_modules/d3-")) return "vendor-recharts";
        },
      },
    },
  },

  server: {
    watch: {
      // Ignore large directories to avoid ENOSPC error
      ignored: ["**/public/tiles/**", "**/node_modules/**"],
    },
    fs: {
      allow: [
        path.resolve(__dirname),
        path.resolve(__dirname, "../Data"),
        path.resolve(__dirname, "node_modules"),
      ],
    },
    proxy: {
      // Proxy all API requests to backend
      "/api": "http://localhost:8000",
      "/hirise": "http://localhost:8000",
      "/crism": "http://localhost:8000",
      "/hirise_index.geojson": "http://localhost:8000",
      "/crism_index.geojson": "http://localhost:8000",
      "/hirise_lbl": "http://localhost:8000",
      "/crism_lbl": "http://localhost:8000",
      "/hirise_viewer": "http://localhost:8000",
      "/world_meta": "http://localhost:8000",
      "/world_tiles": "http://localhost:8000",
      "/sharad_index.geojson": "http://localhost:8000",
      "/sharad": "http://localhost:8000",
      "/terrain": "http://localhost:8000",
    },
  },
});
