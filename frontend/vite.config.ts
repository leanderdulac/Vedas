import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
      "/ask": "http://127.0.0.1:8000",
      "/search": "http://127.0.0.1:8000",
      // Endpoints da fila de jobs (página Operações) ficam fora de /api.
      "/jobs": "http://127.0.0.1:8000",
      "/ingest": "http://127.0.0.1:8000",
      "/tokenize": "http://127.0.0.1:8000",
      "/train": "http://127.0.0.1:8000",
      "/build-index": "http://127.0.0.1:8000",
      "/db": "http://127.0.0.1:8000",
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
