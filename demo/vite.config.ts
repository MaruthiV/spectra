import { defineConfig } from "vite";

export default defineConfig({
  server: {
    port: 5173,
    strictPort: true,
    // Required headers for SharedArrayBuffer / cross-origin isolation
    // (some WebGPU paths benefit from these; keeps us in the safe lane).
    headers: {
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Opener-Policy": "same-origin",
    },
  },
  // Allow .wasm to be fetched as a binary asset.
  assetsInclude: ["**/*.wasm"],
  // Disable single-file optimizations that interfere with web-llm's worker.
  optimizeDeps: {
    exclude: ["@mlc-ai/web-llm"],
  },
});
