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
import { SpecController, type SpecConfig } from "./SpecController.js";

const STOCK_MODEL_ID = "Qwen2.5-0.5B-Instruct-q4f16_1-MLC";
const SPECTRA_MODEL_ID = "Spectra-Qwen2.5-0.5B-Instruct-q4f16_1";

const SPECTRA_TARGET_ID = "Spectra-Qwen2.5-1.5B-Instruct-q4f16_1";
const SPECTRA_DRAFT_ID = "Spectra-Qwen2.5-0.5B-Instruct-q4f16_1";

const HF_WEIGHTS_BASE_URL =
  "https://huggingface.co/mlc-ai/Qwen2.5-0.5B-Instruct-q4f16_1-MLC/resolve/main/";
const HF_TARGET_WEIGHTS_URL =
  "https://huggingface.co/mlc-ai/Qwen2.5-1.5B-Instruct-q4f16_1-MLC/resolve/main/";

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

/**
 * Phase 4 main event: load 1.5B target + 0.5B draft, run the Spectra
 * SpecController, report acceptance rate + tok/s.
 */
async function runSpec(): Promise<void> {
  setBusy(true);
  log(`Setting up spec engine: target=${SPECTRA_TARGET_ID}, draft=${SPECTRA_DRAFT_ID}`);

  // Both models loaded into one engine via successive reloads.
  const appConfig: webllm.AppConfig = {
    model_list: [
      {
        model: HF_TARGET_WEIGHTS_URL,
        model_id: SPECTRA_TARGET_ID,
        model_lib: `${window.location.origin}/spectra-qwen2_5_1_5b_webgpu.wasm`,
        overrides: { context_window_size: 4096 },
      },
      {
        model: HF_WEIGHTS_BASE_URL,
        model_id: SPECTRA_DRAFT_ID,
        model_lib: `${window.location.origin}/spectra-qwen2_5_0_5b_webgpu.wasm`,
        overrides: { context_window_size: 4096 },
      },
    ],
  };

  const engine = new webllm.MLCEngine({ appConfig, initProgressCallback });

  log(`Loading target (1.5B)…`);
  const tT0 = performance.now();
  await engine.reload(SPECTRA_TARGET_ID);
  log(`  target loaded in ${((performance.now() - tT0) / 1000).toFixed(1)} s`);

  log(`Loading draft (0.5B)…`);
  const tD0 = performance.now();
  await engine.reload(SPECTRA_DRAFT_ID);
  log(`  draft loaded in ${((performance.now() - tD0) / 1000).toFixed(1)} s`);
  log(`Loaded models: ${engine.spectraListLoadedModels().join(", ")}`);

  const target = engine.spectraGetChatPipeline(SPECTRA_TARGET_ID);
  const draft = engine.spectraGetChatPipeline(SPECTRA_DRAFT_ID);
  if (!target || !draft) {
    log("ERROR: failed to retrieve pipelines from engine");
    setBusy(false);
    return;
  }

  // Check batch_verify availability on the target.
  const fbatchVerify = target.spectraGetPackedFunc("batch_verify");
  log(`target.batch_verify PackedFunc resolved: ${fbatchVerify ? "yes ✓" : "NO ✗"}`);
  log(`target vocab=${target.spectraGetVocabSize()}, draft vocab=${draft.spectraGetVocabSize()}`);

  const cfg: SpecConfig = {
    draftLength: 2,
    maxTokens: 64,
    onStep: (e) => {
      const pat = e.committed
        .map((_, i) => (i < e.accepted ? "✓" : "·"))
        .join("");
      log(
        `  round: drafts=[${e.drafts.join(",")}] accepted=${e.accepted}/${e.drafts.length} ${pat} committed=${e.committed.length} ${e.latencyMs.toFixed(0)}ms`,
      );
    },
  };

  const allStats: { decode: number; tokens: number; accept: number; rounds: number }[] = [];
  for (let i = 0; i < PROMPTS.length; i++) {
    const prompt = PROMPTS[i];
    log(`[${i + 1}/${PROMPTS.length}] ${prompt.slice(0, 60)}…`);

    // Reset both pipelines' KV between prompts.
    target.resetChat();
    draft.resetChat();

    // Tokenize with target's tokenizer (same vocab as draft — Phase 0a A5).
    const tokenizer = target.spectraGetTokenizer();
    const promptBytes = new TextEncoder().encode(prompt);
    const promptTokens = Array.from(tokenizer.encode(prompt));
    log(`  prompt → ${promptTokens.length} tokens`);

    const ctl = new SpecController(target, draft, cfg);
    try {
      const result = await ctl.generate(promptTokens);
      const text = new TextDecoder().decode(
        tokenizer.decode(Int32Array.from(result.tokens)),
      );
      log(
        `  → ${result.tokens.length} tokens, ${result.rounds} rounds, accept=${result.cumulativeAcceptance.toFixed(3)}, decode=${result.tokensPerSecond.toFixed(1)} tok/s (${result.decodeMs.toFixed(0)}ms)`,
      );
      $("completion").innerText = text;
      allStats.push({
        decode: result.tokensPerSecond,
        tokens: result.tokens.length,
        accept: result.cumulativeAcceptance,
        rounds: result.rounds,
      });
    } catch (e) {
      log(`  ERROR during generate: ${e}`);
      console.error(e);
      void promptBytes;
      setBusy(false);
      return;
    }
  }

  const meanDecode = allStats.reduce((s, a) => s + a.decode, 0) / Math.max(allStats.length, 1);
  const meanAccept = allStats.reduce((s, a) => s + a.accept, 0) / Math.max(allStats.length, 1);
  log(
    `\n=== SPECTRA SPEC SUMMARY ===\n` +
      `prompts: ${allStats.length}\n` +
      `mean acceptance: ${meanAccept.toFixed(3)}\n` +
      `mean decode (spec): ${meanDecode.toFixed(1)} tok/s\n` +
      `\nCompare vs target greedy decode on M3 Pro Chrome (~35 tok/s for 1.5B).\n` +
      `Speedup = ${(meanDecode / 35).toFixed(2)}× (vs ~35 tok/s reference).\n`,
  );
  $("stats").innerHTML =
    `<span class="stat">spec ${meanDecode.toFixed(1)} tok/s</span>` +
    `<span class="stat">accept ${meanAccept.toFixed(3)}</span>` +
    `<span class="stat">${(meanDecode / 35).toFixed(2)}× vs 35 tok/s</span>`;
  setBusy(false);
}

function setBusy(busy: boolean): void {
  for (const id of ["run-stock", "run-spectra", "run-spec"] as const) {
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
  $("run-spec").addEventListener("click", () => runSpec().catch((e) => log(`ERROR: ${e}`)));
});
