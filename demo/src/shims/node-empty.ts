/**
 * Empty shim for Node-only modules referenced by `@mlc-ai/web-llm`'s
 * worker code paths. The browser bundle never reaches those paths at
 * runtime; we just need module resolution to succeed.
 *
 * Aliased in `vite.config.ts` for: ws, perf_hooks, module.
 */
const empty = {};
export default empty;
export const WebSocket = undefined;
export const Server = undefined;
export const performance = globalThis.performance;
export const createRequire = () => () => undefined;
