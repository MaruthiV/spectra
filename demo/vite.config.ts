import { defineConfig } from "vite";
import path from "node:path";

const SHIM = path.resolve(__dirname, "src/shims/node-empty.ts");

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
  // Alias web-llm's Node-only deps (used only in worker code paths) to an
  // empty shim. The browser never reaches the code that uses them at runtime;
  // we just need module resolution to succeed in both dev and build.
  resolve: {
    alias: {
      ws: SHIM,
      perf_hooks: SHIM,
      module: SHIM,
    },
  },
  build: {
    rollupOptions: {
      // Belt-and-suspenders for production builds in case alias misses.
      external: ["ws", "perf_hooks", "module"],
    },
  },
});
