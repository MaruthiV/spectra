/**
 * Spectra demo — Phase 3 smoke test.
 *
 * Two paths:
 *  1. "stock" — load Qwen2.5-0.5B-Instruct-q4f16_1-MLC from the official
 *     mlc-ai prebuilt list. This proves the env + WebGPU + web-llm work.
 *  2. "spectra" — load the same weights but with our own compiled .wasm
 *     (placed under demo/public/). Proves our compile output is loadable
 *     by web-llm and gives us a tok/s reading we can quote in the README.
 */

import * as webllm from "@mlc-ai/web-llm";

const STOCK_MODEL_ID = "Qwen2.5-0.5B-Instruct-q4f16_1-MLC";
const SPECTRA_MODEL_ID = "Spectra-Qwen2.5-0.5B-Instruct-q4f16_1";

const HF_WEIGHTS_BASE_URL =
  "https://huggingface.co/mlc-ai/Qwen2.5-0.5B-Instruct-q4f16_1-MLC/resolve/main/";

const PROMPTS = [
  "Compose a haiku about WebGPU running an LLM in the browser.",
  "Explain rejection-sampling speculative decoding in one paragraph.",
  "Write five Python list comprehensions, one per line, that each filter primes from a list.",
];

function $(id: string): HTMLElement {
  const el = document.getElementById(id);
  if (!el) throw new Error(`No element #${id}`);
  return el;
}

function log(msg: string): void {
  const el = $("log");
  el.innerText += `${new Date().toISOString().slice(11, 19)}  ${msg}\n`;
  el.scrollTop = el.scrollHeight;
  console.log("[spectra-demo]", msg);
}

function initProgressCallback(report: webllm.InitProgressReport): void {
  $("init-label").innerText = report.text;
}

async function runStock(): Promise<void> {
  setBusy(true);
  log(`Loading STOCK model ${STOCK_MODEL_ID} from prebuilt config…`);
  const t0 = performance.now();
  const engine = await webllm.CreateMLCEngine(
    STOCK_MODEL_ID,
    {
      initProgressCallback,
      logLevel: "INFO",
    },
    {
      context_window_size: 4096,
    },
  );
  log(`Loaded in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
  await runPromptsOn(engine);
  setBusy(false);
}

async function runSpectra(): Promise<void> {
  setBusy(true);
  log(`Loading SPECTRA-compiled lib ${SPECTRA_MODEL_ID}…`);
  const appConfig: webllm.AppConfig = {
    model_list: [
      {
        model: HF_WEIGHTS_BASE_URL,
        model_id: SPECTRA_MODEL_ID,
        // Served from /spectra-qwen2_5_0_5b_webgpu.wasm (in demo/public/).
        // Built locally via mlc_llm compile --enable-subgroups
        //   --overrides "max_batch_size=4;prefill_chunk_size=1024;context_window_size=4096"
        model_lib: `${window.location.origin}/spectra-qwen2_5_0_5b_webgpu.wasm`,
        overrides: {
          context_window_size: 4096,
        },
      },
    ],
  };
  const t0 = performance.now();
  const engine = await webllm.CreateMLCEngine(
    SPECTRA_MODEL_ID,
    {
      appConfig,
      initProgressCallback,
      logLevel: "INFO",
    },
  );
  log(`Loaded in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
  await runPromptsOn(engine);
  setBusy(false);
}

async function runPromptsOn(engine: webllm.MLCEngineInterface): Promise<void> {
  log(`Running ${PROMPTS.length} prompts (greedy, max_tokens=64)…`);
  const allStats: { prefill: number; decode: number; tokens: number }[] = [];
  for (let i = 0; i < PROMPTS.length; i++) {
    const prompt = PROMPTS[i];
    log(`[${i + 1}/${PROMPTS.length}] ${prompt.slice(0, 60)}…`);
    const reply = await engine.chat.completions.create({
      messages: [{ role: "user", content: prompt }],
      max_tokens: 64,
      temperature: 0,
    });
    const text = reply.choices[0]?.message?.content ?? "(no content)";
    const usage = reply.usage as Record<string, number> | undefined;
    const prefillTps = (usage?.extra?.["prefill_tokens_per_s"] as number | undefined)
      ?? (reply as any).usage?.prefill_tokens_per_s
      ?? 0;
    const decodeTps = (usage?.extra?.["decode_tokens_per_s"] as number | undefined)
      ?? (reply as any).usage?.decode_tokens_per_s
      ?? 0;
    const completionTokens = usage?.completion_tokens ?? 0;
    allStats.push({
      prefill: prefillTps,
      decode: decodeTps,
      tokens: completionTokens,
    });
    log(
      `  → ${completionTokens} tokens, prefill ${prefillTps.toFixed(1)} tok/s, decode ${decodeTps.toFixed(1)} tok/s`,
    );
    $("completion").innerText = text;
  }
  const meanDecode =
    allStats.reduce((s, a) => s + a.decode, 0) / Math.max(allStats.length, 1);
  const meanPrefill =
    allStats.reduce((s, a) => s + a.prefill, 0) / Math.max(allStats.length, 1);
  log(
    `\n=== summary ===\n` +
      `prompts: ${allStats.length}\n` +
      `mean prefill: ${meanPrefill.toFixed(1)} tok/s\n` +
      `mean decode:  ${meanDecode.toFixed(1)} tok/s   ← this is the North Star\n`,
  );
  $("stats").innerHTML =
    `<span class="stat">decode ${meanDecode.toFixed(1)} tok/s</span>` +
    `<span class="stat">prefill ${meanPrefill.toFixed(1)} tok/s</span>` +
    `<span class="stat">${allStats.length} prompts</span>`;
}

async function dumpFuncs(): Promise<void> {
  log("PackedFunc dump is unavailable from the public web-llm API. The list");
  log("of functions exposed by the loaded .wasm is in the model's metadata");
  log("inside the binary. Use `strings spectra-qwen2_5_0_5b_webgpu.wasm | grep`");
  log("for an offline check; per Phase 0b we already saw batch_verify there.");
}

function setBusy(busy: boolean): void {
  for (const id of ["run-stock", "run-spectra", "dump-funcs"] as const) {
    ($(id) as HTMLButtonElement).disabled = busy;
  }
}

window.addEventListener("DOMContentLoaded", async () => {
  log("Initializing Spectra demo…");
  if (!("gpu" in navigator)) {
    log("ERROR: navigator.gpu is missing — WebGPU not available in this browser.");
    setBusy(true);
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    log("ERROR: no WebGPU adapter (does this browser have hardware acceleration enabled?)");
    setBusy(true);
    return;
  }
  // @ts-expect-error: subgroupMinSize is a recent API extension
  const subgroupMin = adapter.info?.subgroupMinSize ?? "n/a";
  log(`WebGPU adapter ready. subgroupMinSize=${subgroupMin}`);
  log(`Features available: ${[...adapter.features].join(", ")}`);

  $("run-stock").addEventListener("click", () => runStock().catch((e) => log(`ERROR: ${e}`)));
  $("run-spectra").addEventListener("click", () => runSpectra().catch((e) => log(`ERROR: ${e}`)));
  $("dump-funcs").addEventListener("click", () => dumpFuncs().catch((e) => log(`ERROR: ${e}`)));
});
