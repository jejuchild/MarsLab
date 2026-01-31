import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],

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
      "/api": "http://localhost:5001",
      "/hirise": "http://localhost:5001",
      "/crism": "http://localhost:5001",
      "/hirise_index.geojson": "http://localhost:5001",
      "/crism_index.geojson": "http://localhost:5001",
      "/hirise_lbl": "http://localhost:5001",
      "/crism_lbl": "http://localhost:5001",
      "/hirise_viewer": "http://localhost:5001",
      "/world_meta": "http://localhost:5001",
      "/world_tiles": "http://localhost:5001",
      "/sharad_index.geojson": "http://localhost:5001",
      "/sharad": "http://localhost:5001",
    },
  },
});
