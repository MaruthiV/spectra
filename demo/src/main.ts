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
import { EagleSpecController, type EagleSpecConfig } from "./EagleSpecController.js";

const STOCK_MODEL_ID = "Qwen2.5-0.5B-Instruct-q4f16_1-MLC";
const SPECTRA_MODEL_ID = "Spectra-Qwen2.5-0.5B-Instruct-q4f16_1";

const SPECTRA_TARGET_ID = "Spectra-Qwen2.5-1.5B-Instruct-q4f16_1";
const SPECTRA_DRAFT_ID = "Spectra-Qwen2.5-0.5B-Instruct-q4f16_1";

const HF_WEIGHTS_BASE_URL =
  "https://huggingface.co/mlc-ai/Qwen2.5-0.5B-Instruct-q4f16_1-MLC/resolve/main/";
const HF_TARGET_WEIGHTS_URL =
  "https://huggingface.co/mlc-ai/Qwen2.5-1.5B-Instruct-q4f16_1-MLC/resolve/main/";

// EAGLE-3 path: Qwen3-1.7B target + AngelSlim pre-trained head.
const EAGLE3_TARGET_ID = "Spectra-Qwen3-1.7B";
const EAGLE3_HEAD_ID = "Spectra-Eagle3-Qwen3-1.7B";
const EAGLE3_TARGET_WEIGHTS_URL =
  "https://huggingface.co/mlc-ai/Qwen3-1.7B-q4f16_1-MLC/resolve/main/";
// Head weights live on HF — see https://huggingface.co/VemVemRu/Spectra-Eagle3-Qwen3-1.7B
const EAGLE3_HEAD_WEIGHTS_URL =
  "https://huggingface.co/VemVemRu/Spectra-Eagle3-Qwen3-1.7B/resolve/main/";

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

  log(`Loading target (1.5B) + draft (0.5B) together via engine.reload([…])…`);
  const tT0 = performance.now();
  await engine.reload([SPECTRA_TARGET_ID, SPECTRA_DRAFT_ID]);
  log(`  both loaded in ${((performance.now() - tT0) / 1000).toFixed(1)} s`);
  log(`Loaded models: ${engine.spectraListLoadedModels().join(", ")}`);

  const target = engine.spectraGetChatPipeline(SPECTRA_TARGET_ID);
  const draft = engine.spectraGetChatPipeline(SPECTRA_DRAFT_ID);
  if (!target || !draft) {
    log("ERROR: failed to retrieve pipelines from engine");
    setBusy(false);
    return;
  }

  // Diagnostic: probe VM function table on the target.
  const probe = target.spectraProbeVMFunctions([
    "prefill",
    "batch_prefill",
    "decode",
    "batch_decode",
    "embed",
    "sample_with_top_p",
    "argsort_probs",
    "batch_verify",
    "batch_verifier",
    "create_tir_paged_kv_cache",
    "_metadata",
  ]);
  const abi = target.spectraGetResolvedABI();
  log(`target ABI resolved: prefill=${abi.prefill}, decode=${abi.decode}`);
  log(`target VM function table:`);
  for (const [k, v] of Object.entries(probe)) {
    log(`  ${v ? "✓" : "✗"} ${k}`);
  }
  log(`target vocab=${target.spectraGetVocabSize()}, draft vocab=${draft.spectraGetVocabSize()}`);

  const gammaInput = $("gamma") as HTMLInputElement;
  const gamma = parseInt(gammaInput.value, 10) || 2;
  log(`Using γ=${gamma} (drafts per spec round)`);
  const cfg: SpecConfig = {
    draftLength: gamma,
    maxTokens: 64,
    onStep: (e) => {
      const pat = e.committed
        .map((_, i) => (i < e.accepted ? "✓" : "·"))
        .join("");
      log(
        `  r${e.round}: d=[${e.drafts.join(",")}] a=${e.accepted}/${e.drafts.length} ${pat} c=${e.committed.length} | tot=${e.timing.totalMs.toFixed(0)} draft=${e.timing.draftLoopMs.toFixed(0)} verify=${e.timing.targetVerifyMs.toFixed(0)} kv=${e.timing.kvOpsMs.toFixed(0)} arg=${e.timing.argmaxMs.toFixed(0)} drLC=${e.timing.draftLastCommitMs.toFixed(0)}`,
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
      log(`  starting generate...`);
      const result = await ctl.generate(promptTokens);
      // web-tokenizers' decode() returns a string directly.
      const decoded = tokenizer.decode(Int32Array.from(result.tokens)) as unknown;
      const text = typeof decoded === "string"
        ? decoded
        : new TextDecoder().decode(decoded as BufferSource);
      log(
        `  → ${result.tokens.length} tokens, ${result.rounds} rounds, accept=${result.cumulativeAcceptance.toFixed(3)}, decode=${result.tokensPerSecond.toFixed(1)} tok/s (${result.decodeMs.toFixed(0)}ms)`,
      );
      const mt = result.meanTiming;
      log(
        `     mean per-round: tot=${mt.totalMs.toFixed(1)} draft=${mt.draftLoopMs.toFixed(1)} verify=${mt.targetVerifyMs.toFixed(1)} kv=${mt.kvOpsMs.toFixed(1)} arg=${mt.argmaxMs.toFixed(1)} drLC=${mt.draftLastCommitMs.toFixed(1)} ms`,
      );
      $("completion").innerText = text;
      allStats.push({
        decode: result.tokensPerSecond,
        tokens: result.tokens.length,
        accept: result.cumulativeAcceptance,
        rounds: result.rounds,
      });
    } catch (e) {
      const err = e as Error;
      log(`  ERROR during generate: ${err}`);
      if (err.stack) {
        const stackLines = err.stack.split("\n").slice(0, 6);
        for (const line of stackLines) log(`    at ${line.trim()}`);
      }
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
  for (const id of ["run-stock", "run-spectra", "run-spec", "run-race", "run-eagle3"] as const) {
    const el = document.getElementById(id) as HTMLButtonElement | null;
    if (el) el.disabled = busy;
  }
}

// ---- Race UI (D5) -----------------------------------------------------------
// Shared engine (target+draft) lazy-loaded on first race click and cached.
let cachedSpecEngine: webllm.MLCEngine | null = null;
let cachedTarget: webllm.LLMChatPipeline | null = null;
let cachedDraft: webllm.LLMChatPipeline | null = null;

async function ensureSpecEngine(): Promise<{
  engine: webllm.MLCEngine;
  target: webllm.LLMChatPipeline;
  draft: webllm.LLMChatPipeline;
}> {
  if (cachedSpecEngine && cachedTarget && cachedDraft) {
    return { engine: cachedSpecEngine, target: cachedTarget, draft: cachedDraft };
  }
  log("[race] loading target+draft engine (one-time, ~30s)…");
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
  await engine.reload([SPECTRA_TARGET_ID, SPECTRA_DRAFT_ID]);
  const target = engine.spectraGetChatPipeline(SPECTRA_TARGET_ID);
  const draft = engine.spectraGetChatPipeline(SPECTRA_DRAFT_ID);
  if (!target || !draft) throw new Error("[race] failed to retrieve pipelines");
  cachedSpecEngine = engine;
  cachedTarget = target;
  cachedDraft = draft;
  log("[race] engine ready.");
  return { engine, target, draft };
}

function argmaxLogits(logits: Float32Array): number {
  let bestIdx = 0;
  let bestVal = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > bestVal) {
      bestVal = logits[i];
      bestIdx = i;
    }
  }
  return bestIdx;
}

/**
 * Plain greedy decode against the target pipeline. One target forward per token —
 * this is the "baseline" we're trying to beat.
 */
async function runTargetGreedy(
  target: webllm.LLMChatPipeline,
  promptTokens: number[],
  maxTokens: number,
  onToken: (text: string) => void,
  eosId: number,
): Promise<{ tokens: number[]; decodeMs: number; tokPerSec: number }> {
  target.resetChat();
  const tokenizer = (target as any).spectraGetTokenizer();
  // Prefill the full prompt; get logits at last position.
  let lastLogits = await target.spectraPrefillMultiPosition(
    promptTokens,
    [promptTokens.length - 1],
  );
  const generated: number[] = [];
  const tStart = performance.now();
  for (let i = 0; i < maxTokens; i++) {
    const tokId = argmaxLogits(lastLogits);
    if (tokId === eosId) break;
    generated.push(tokId);
    const decoded = tokenizer.decode(Int32Array.from([tokId])) as unknown;
    const text = typeof decoded === "string"
      ? decoded
      : new TextDecoder().decode(decoded as BufferSource);
    onToken(text);
    // Feed the new token at "current end" position.
    lastLogits = await target.spectraPrefillMultiPosition([tokId], [0]);
  }
  const decodeMs = performance.now() - tStart;
  return {
    tokens: generated,
    decodeMs,
    tokPerSec: generated.length / (decodeMs / 1000),
  };
}

async function runRace(): Promise<void> {
  setBusy(true);
  const promptInput = $("race-prompt") as HTMLInputElement;
  const maxTokInput = $("race-max-tokens") as HTMLInputElement;
  const gammaInput = $("race-gamma") as HTMLInputElement;
  const prompt = promptInput.value.trim();
  if (!prompt) {
    setBusy(false);
    return;
  }
  const maxTokens = Math.max(16, Math.min(256, parseInt(maxTokInput.value, 10) || 80));
  const gamma = Math.max(1, Math.min(3, parseInt(gammaInput.value, 10) || 2));

  const baselinePane = $("race-pane-baseline");
  const spectraPane = $("race-pane-spectra");
  const baselineText = $("race-text-baseline");
  const spectraText = $("race-text-spectra");
  const baselineStats = $("race-stats-baseline");
  const spectraStats = $("race-stats-spectra");
  const winnerBanner = $("race-winner-banner");

  // Reset visuals.
  baselinePane.classList.remove("winner");
  spectraPane.classList.remove("winner");
  baselineText.innerText = "";
  spectraText.innerHTML = "";
  baselineStats.innerText = "running…";
  spectraStats.innerText = "queued";
  winnerBanner.innerText = "";
  winnerBanner.className = "";

  try {
    const { target, draft } = await ensureSpecEngine();
    const tokenizer = target.spectraGetTokenizer();
    const eosId = 151645;             // Qwen2.5 <|im_end|>
    const promptTokens = Array.from(tokenizer.encode(prompt));

    // ---------- Baseline run ----------
    log(`[race] baseline starting (target greedy, max=${maxTokens})…`);
    const baselineResult = await runTargetGreedy(target, promptTokens, maxTokens, (text) => {
      baselineText.innerText += text;
      baselineText.scrollTop = baselineText.scrollHeight;
    }, eosId);
    baselineStats.innerText = `${baselineResult.tokPerSec.toFixed(1)} tok/s · ${(baselineResult.decodeMs / 1000).toFixed(2)}s · ${baselineResult.tokens.length} tok`;
    log(`[race] baseline done: ${baselineResult.tokPerSec.toFixed(1)} tok/s in ${(baselineResult.decodeMs / 1000).toFixed(2)}s`);

    // ---------- Spectra run ----------
    spectraStats.innerText = "running…";
    log(`[race] spectra starting (γ=${gamma}, max=${maxTokens})…`);
    target.resetChat();
    draft.resetChat();
    const cfg: SpecConfig = {
      draftLength: gamma,
      maxTokens,
      eosTokenId: eosId,
      onStep: (e) => {
        // Render each committed token, color-coded: green = accepted draft, blue = bonus.
        for (let i = 0; i < e.committed.length; i++) {
          const tid = e.committed[i];
          const decoded = tokenizer.decode(Int32Array.from([tid])) as unknown;
          const text = typeof decoded === "string"
            ? decoded
            : new TextDecoder().decode(decoded as BufferSource);
          const span = document.createElement("span");
          span.className = i < e.accepted ? "tok-accept" : "tok-bonus";
          span.innerText = text;
          spectraText.appendChild(span);
        }
        spectraText.scrollTop = spectraText.scrollHeight;
      },
    };
    const ctl = new SpecController(target, draft, cfg);
    const tStart = performance.now();
    const specResult = await ctl.generate(promptTokens);
    const specMs = performance.now() - tStart;
    spectraStats.innerText = `${specResult.tokensPerSecond.toFixed(1)} tok/s · ${(specMs / 1000).toFixed(2)}s · ${specResult.tokens.length} tok · α=${specResult.cumulativeAcceptance.toFixed(2)}`;
    log(`[race] spectra done: ${specResult.tokensPerSecond.toFixed(1)} tok/s, α=${specResult.cumulativeAcceptance.toFixed(2)}`);

    // ---------- Winner ----------
    const speedup = specResult.tokensPerSecond / Math.max(baselineResult.tokPerSec, 1e-3);
    if (specResult.tokensPerSecond > baselineResult.tokPerSec) {
      spectraPane.classList.add("winner");
      winnerBanner.innerText = `⚡ Spectra wins: ${speedup.toFixed(2)}× faster than baseline`;
      winnerBanner.className = "race-winner-banner";
    } else {
      baselinePane.classList.add("winner");
      winnerBanner.innerText = `Baseline wins (${(1 / speedup).toFixed(2)}× faster than Spectra) — α too low, would need a better draft head.`;
      winnerBanner.className = "race-winner-banner";
    }
  } catch (e) {
    log(`[race] ERROR: ${e}`);
    winnerBanner.innerText = `Error: ${e}`;
  } finally {
    setBusy(false);
  }
}

// ---- EAGLE-3 race (Qwen3-1.7B target + AngelSlim pre-trained head) ---------
let cachedEagleEngine: webllm.MLCEngine | null = null;
let cachedEagleTarget: webllm.LLMChatPipeline | null = null;
let cachedEagleHead: webllm.LLMChatPipeline | null = null;
let cachedD2T: Int32Array | null = null;

async function ensureEagleEngine(): Promise<{
  target: webllm.LLMChatPipeline;
  head: webllm.LLMChatPipeline;
  d2t: Int32Array;
  targetVocabSize: number;
}> {
  if (cachedEagleEngine && cachedEagleTarget && cachedEagleHead && cachedD2T) {
    return {
      target: cachedEagleTarget,
      head: cachedEagleHead,
      d2t: cachedD2T,
      targetVocabSize: 151936,
    };
  }
  log("[eagle3-race] loading Qwen3-1.7B + AngelSlim EAGLE-3 head (one-time, ~30s)…");
  log(`[eagle3-race] target weights URL: ${EAGLE3_TARGET_WEIGHTS_URL}`);
  log(`[eagle3-race] head weights URL:   ${EAGLE3_HEAD_WEIGHTS_URL}`);
  log(`[eagle3-race] target wasm:        ${window.location.origin}/spectra-qwen3-1_7b_webgpu.wasm`);
  log(`[eagle3-race] head wasm:          ${window.location.origin}/spectra-eagle3-qwen3-1_7b_webgpu.wasm`);
  const appConfig: webllm.AppConfig = {
    model_list: [
      {
        model: EAGLE3_TARGET_WEIGHTS_URL,
        model_id: EAGLE3_TARGET_ID,
        model_lib: `${window.location.origin}/spectra-qwen3-1_7b_webgpu.wasm`,
        overrides: { context_window_size: 4096 },
      },
      {
        model: EAGLE3_HEAD_WEIGHTS_URL,
        model_id: EAGLE3_HEAD_ID,
        model_lib: `${window.location.origin}/spectra-eagle3-qwen3-1_7b_webgpu.wasm`,
        overrides: { context_window_size: 4096 },
      },
    ],
  };
  const engine = new webllm.MLCEngine({ appConfig, initProgressCallback });
  log("[eagle3-race] engine constructed, calling reload([target, head])…");
  try {
    await engine.reload([EAGLE3_TARGET_ID, EAGLE3_HEAD_ID]);
  } catch (e) {
    log(`[eagle3-race] engine.reload FAILED with both models: ${e}`);
    log("[eagle3-race] trying TARGET only to isolate which fails…");
    try {
      await engine.reload([EAGLE3_TARGET_ID]);
      log("[eagle3-race] target alone loaded OK — head is the problem");
    } catch (e2) {
      log(`[eagle3-race] target alone ALSO failed: ${e2}`);
    }
    throw e;
  }
  log("[eagle3-race] reload OK; getting pipelines…");
  const target = engine.spectraGetChatPipeline(EAGLE3_TARGET_ID);
  const head = engine.spectraGetChatPipeline(EAGLE3_HEAD_ID);
  if (!target || !head) throw new Error("[eagle3-race] failed to retrieve pipelines");
  // Fetch d2t vocab map from the head's static asset.
  log("[eagle3-race] fetching d2t vocab map…");
  const vocabResp = await fetch(`${EAGLE3_HEAD_WEIGHTS_URL}eagle3_vocab_map.json`);
  const vocabJson = await vocabResp.json();
  const d2t = Int32Array.from(vocabJson.d2t as number[]);
  cachedEagleEngine = engine;
  cachedEagleTarget = target;
  cachedEagleHead = head;
  cachedD2T = d2t;
  log(`[eagle3-race] engine ready. d2t length=${d2t.length}, target_vocab=${vocabJson.target_vocab_size}`);
  return { target, head, d2t, targetVocabSize: vocabJson.target_vocab_size };
}

async function runEagle3Race(): Promise<void> {
  setBusy(true);
  const promptInput = $("eagle3-prompt") as HTMLInputElement;
  const maxTokInput = $("eagle3-max-tokens") as HTMLInputElement;
  const gammaInput = $("eagle3-gamma") as HTMLInputElement;
  const prompt = promptInput.value.trim();
  if (!prompt) { setBusy(false); return; }
  const maxTokens = Math.max(16, Math.min(256, parseInt(maxTokInput.value, 10) || 80));
  const gamma = Math.max(1, Math.min(4, parseInt(gammaInput.value, 10) || 2));

  const baselinePane = $("eagle3-pane-baseline");
  const spectraPane = $("eagle3-pane-spectra");
  const baselineText = $("eagle3-text-baseline");
  const spectraText = $("eagle3-text-spectra");
  const baselineStats = $("eagle3-stats-baseline");
  const spectraStats = $("eagle3-stats-spectra");
  const winnerBanner = $("eagle3-winner-banner");

  baselinePane.classList.remove("winner");
  spectraPane.classList.remove("winner");
  baselineText.innerText = "";
  spectraText.innerText = "";
  baselineStats.innerText = "running…";
  spectraStats.innerText = "queued";
  winnerBanner.innerText = "";
  winnerBanner.className = "";

  try {
    const { target, head, d2t, targetVocabSize } = await ensureEagleEngine();
    const tokenizer = target.spectraGetTokenizer();
    const eosId = 151645;  // Qwen3 <|im_end|>
    const promptTokens = Array.from(tokenizer.encode(prompt));

    // Baseline: Qwen3-1.7B greedy decode (one target forward per token).
    log(`[eagle3-race] baseline starting (Qwen3-1.7B greedy, max=${maxTokens})…`);
    const baselineResult = await runTargetGreedy(target, promptTokens, maxTokens, (text) => {
      baselineText.innerText += text;
      baselineText.scrollTop = baselineText.scrollHeight;
    }, eosId);
    baselineStats.innerText = `${baselineResult.tokPerSec.toFixed(1)} tok/s · ${(baselineResult.decodeMs / 1000).toFixed(2)}s · ${baselineResult.tokens.length} tok`;
    log(`[eagle3-race] baseline done: ${baselineResult.tokPerSec.toFixed(1)} tok/s`);

    // Spectra-EAGLE3: target + AngelSlim head.
    spectraStats.innerText = "running…";
    log(`[eagle3-race] EAGLE-3 spec starting (γ=${gamma}, max=${maxTokens})…`);
    const cfg: EagleSpecConfig = {
      draftLength: gamma,
      maxTokens,
      eosTokenId: eosId,
      onStep: (e) => {
        for (const tid of e.committed) {
          const decoded = tokenizer.decode(Int32Array.from([tid])) as unknown;
          const text = typeof decoded === "string" ? decoded : new TextDecoder().decode(decoded as BufferSource);
          spectraText.innerText += text;
        }
        spectraText.scrollTop = spectraText.scrollHeight;
        log(
          `  r${e.round} d=[${e.drafts.join(",")}] a=${e.accepted}/${e.drafts.length} c=${e.committed.length} ` +
          `head=${e.timing.headDraftMs.toFixed(0)} verify=${e.timing.targetVerifyMs.toFixed(0)} kv=${e.timing.kvOpsMs.toFixed(0)} tot=${e.timing.totalMs.toFixed(0)}ms`,
        );
      },
    };
    const ctl = new EagleSpecController(target, head, d2t, targetVocabSize, cfg);
    const specResult = await ctl.generate(promptTokens);
    spectraStats.innerText = `${specResult.tokensPerSecond.toFixed(1)} tok/s · ${(specResult.decodeMs / 1000).toFixed(2)}s · ${specResult.tokens.length} tok · α=${specResult.cumulativeAcceptance.toFixed(2)}`;
    log(`[eagle3-race] EAGLE-3 done: ${specResult.tokensPerSecond.toFixed(1)} tok/s, α=${specResult.cumulativeAcceptance.toFixed(2)}`);

    const speedup = specResult.tokensPerSecond / Math.max(baselineResult.tokPerSec, 1e-3);
    if (specResult.tokensPerSecond > baselineResult.tokPerSec) {
      spectraPane.classList.add("winner");
      winnerBanner.innerText = `⚡ EAGLE-3 wins: ${speedup.toFixed(2)}× faster than baseline (α=${specResult.cumulativeAcceptance.toFixed(2)})`;
      winnerBanner.className = "race-winner-banner";
    } else {
      baselinePane.classList.add("winner");
      winnerBanner.innerText = `Baseline wins (${(1 / speedup).toFixed(2)}× faster) — α=${specResult.cumulativeAcceptance.toFixed(2)} too low`;
      winnerBanner.className = "race-winner-banner";
    }
  } catch (e) {
    log(`[eagle3-race] ERROR: ${e}`);
    winnerBanner.innerText = `Error: ${e}`;
    console.error(e);
  } finally {
    setBusy(false);
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
  $("run-race").addEventListener("click", () => runRace().catch((e) => log(`ERROR: ${e}`)));
  $("run-eagle3").addEventListener("click", () => runEagle3Race().catch((e) => log(`ERROR: ${e}`)));

  const gammaInput = $("gamma") as HTMLInputElement;
  const gammaVal = $("gamma-val");
  gammaVal.innerText = gammaInput.value;
  gammaInput.addEventListener("input", () => {
    gammaVal.innerText = gammaInput.value;
  });
});
