import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Alvo da API no dev (scripts/run_dev.sh sobe em 8000).
// Sobrescreva com VEDIC_API_TARGET=http://127.0.0.1:8611 se a API estiver em outra porta.
const apiTarget = process.env.VEDIC_API_TARGET || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": apiTarget,
      "/health": apiTarget,
      "/ask": apiTarget,
      "/search": apiTarget,
      // Endpoints da fila de jobs (página Operações) ficam fora de /api.
      "/jobs": apiTarget,
      "/ingest": apiTarget,
      "/tokenize": apiTarget,
      "/train": apiTarget,
      "/build-index": apiTarget,
      "/db": apiTarget,
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
