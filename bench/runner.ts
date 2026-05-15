/**
 * Spectra benchmark runner.
 *
 * Drives the Vite dev server's demo via Playwright + Chromium with WebGPU
 * enabled, sweeps γ ∈ {2, 3, 4} on the same 3-prompt set, and writes raw
 * results to `bench/results/<machine>_<timestamp>.json`.
 *
 * Prereq:
 *   pnpm install            # in this directory
 *   pnpm bench:install      # fetches Playwright's bundled Chromium
 *   pnpm dev                # in spectra/demo/ (separate terminal)
 *
 * Run:
 *   pnpm bench              # in this directory
 *   # → writes to bench/results/<host>_<utc>.json
 */

import { chromium, Browser, Page, ConsoleMessage } from "playwright";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

const DEMO_URL = process.env.SPECTRA_DEMO_URL ?? "http://localhost:5173/";
const GAMMAS = (process.env.SPECTRA_GAMMAS ?? "2,3,4")
  .split(",")
  .map((s) => parseInt(s.trim(), 10))
  .filter((n) => Number.isInteger(n) && n >= 1 && n <= 8);
const OUTPUT_DIR = path.join(import.meta.dirname, "results");
const TIMEOUT_MS = 5 * 60 * 1000;       // 5 min per γ run
const HEADLESS = process.env.SPECTRA_HEADLESS !== "false";

interface PerPromptStat {
  prompt: string;
  tokens: number;
  rounds: number;
  acceptance: number;
  tokensPerSecond: number;
  decodeMs: number;
  meanTotalMs?: number;
  meanDraftLoopMs?: number;
  meanTargetVerifyMs?: number;
  meanKVOpsMs?: number;
  meanArgmaxMs?: number;
  meanDraftLastCommitMs?: number;
}

interface GammaRun {
  gamma: number;
  perPrompt: PerPromptStat[];
  meanAcceptance: number;
  meanTokensPerSecond: number;
  speedupVsBaseline: number;  // vs 35 tok/s
  errors: string[];
}

interface BenchResult {
  schema: 1;
  utcStartedAt: string;
  utcEndedAt: string;
  hostname: string;
  platform: string;
  arch: string;
  cpus: number;
  totalMemMB: number;
  demoUrl: string;
  baselineTokPerSec: number;
  runs: GammaRun[];
}

const BASELINE_TOK_PER_SEC = 35;  // M3 Pro Chrome target greedy decode (Phase 0a A3 follow-up)

async function setupPage(browser: Browser): Promise<Page> {
  const context = await browser.newContext({
    viewport: { width: 1280, height: 900 },
  });
  const page = await context.newPage();
  page.setDefaultTimeout(TIMEOUT_MS);
  page.on("pageerror", (err) => console.error("[page-error]", err.message));
  return page;
}

interface ParsedSummary {
  promptStats: PerPromptStat[];
  meanAcceptance: number;
  meanTokensPerSecond: number;
}

/**
 * Parse the demo's on-page #log innerText into per-prompt stats + summary.
 * Demo log format we rely on:
 *   `  → 64 tokens, 23 rounds, accept=0.891, decode=56.8 tok/s (1126ms)`
 *   `     mean per-round: tot=49.3 draft=8.2 verify=18.5 kv=2.4 arg=4.0 drLC=11.2 ms`
 *   `\n=== SPECTRA SPEC SUMMARY ===\nprompts: 3\nmean acceptance: 0.598\nmean decode (spec): 43.0 tok/s`
 */
function parseLog(rawLog: string, prompts: string[]): ParsedSummary {
  const stats: PerPromptStat[] = [];
  const lines = rawLog.split("\n");
  let curPrompt: string | undefined;
  let curStat: Partial<PerPromptStat> | undefined;
  for (const ln of lines) {
    const promptMatch = ln.match(/\[\d+\/\d+\]\s+(.+?)…/);
    if (promptMatch) {
      curPrompt = promptMatch[1].trim();
      curStat = { prompt: curPrompt };
      continue;
    }
    const arrowMatch = ln.match(
      /→\s+(\d+) tokens,\s+(\d+) rounds,\s+accept=([\d.]+),\s+decode=([\d.]+) tok\/s\s+\((\d+)ms\)/,
    );
    if (arrowMatch && curStat) {
      curStat.tokens = parseInt(arrowMatch[1], 10);
      curStat.rounds = parseInt(arrowMatch[2], 10);
      curStat.acceptance = parseFloat(arrowMatch[3]);
      curStat.tokensPerSecond = parseFloat(arrowMatch[4]);
      curStat.decodeMs = parseFloat(arrowMatch[5]);
      continue;
    }
    const meanMatch = ln.match(
      /mean per-round:\s+tot=([\d.]+)\s+draft=([\d.]+)\s+verify=([\d.]+)\s+kv=([\d.]+)\s+arg=([\d.]+)\s+drLC=([\d.]+)/,
    );
    if (meanMatch && curStat) {
      curStat.meanTotalMs = parseFloat(meanMatch[1]);
      curStat.meanDraftLoopMs = parseFloat(meanMatch[2]);
      curStat.meanTargetVerifyMs = parseFloat(meanMatch[3]);
      curStat.meanKVOpsMs = parseFloat(meanMatch[4]);
      curStat.meanArgmaxMs = parseFloat(meanMatch[5]);
      curStat.meanDraftLastCommitMs = parseFloat(meanMatch[6]);
      stats.push(curStat as PerPromptStat);
      curStat = undefined;
    }
  }
  // Pull the summary line.
  const summaryMatch = rawLog.match(
    /mean acceptance:\s+([\d.]+)\s*[\s\S]*?mean decode \(spec\):\s+([\d.]+)\s+tok\/s/,
  );
  const meanAcceptance = summaryMatch ? parseFloat(summaryMatch[1]) : NaN;
  const meanTokensPerSecond = summaryMatch ? parseFloat(summaryMatch[2]) : NaN;
  void prompts;
  return { promptStats: stats, meanAcceptance, meanTokensPerSecond };
}

async function runOnce(page: Page, gamma: number): Promise<GammaRun> {
  const errors: string[] = [];
  page.on("console", (msg: ConsoleMessage) => {
    if (msg.type() === "error") errors.push(`[browser-console-error] ${msg.text()}`);
  });

  console.log(`\n=== γ=${gamma} ===`);
  await page.goto(DEMO_URL, { waitUntil: "load" });

  // Wait for the demo's WebGPU adapter check to complete.
  await page.waitForFunction(() => {
    const el = document.getElementById("log");
    return el && /WebGPU adapter ready/.test(el.innerText);
  }, undefined, { timeout: 30_000 });

  // Set γ slider and click spec button.
  await page.evaluate((g) => {
    const input = document.getElementById("gamma") as HTMLInputElement;
    input.value = String(g);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  }, gamma);
  await page.click("#run-spec");

  // Wait for summary line in the log.
  await page.waitForFunction(() => {
    const el = document.getElementById("log");
    return el && /=== SPECTRA SPEC SUMMARY ===/.test(el.innerText);
  }, undefined, { timeout: TIMEOUT_MS });

  const rawLog = await page.evaluate(() => {
    const el = document.getElementById("log");
    return el ? el.innerText : "";
  });

  const parsed = parseLog(rawLog, []);
  const run: GammaRun = {
    gamma,
    perPrompt: parsed.promptStats,
    meanAcceptance: parsed.meanAcceptance,
    meanTokensPerSecond: parsed.meanTokensPerSecond,
    speedupVsBaseline: parsed.meanTokensPerSecond / BASELINE_TOK_PER_SEC,
    errors,
  };
  console.log(
    `  → mean accept=${run.meanAcceptance.toFixed(3)}  decode=${run.meanTokensPerSecond.toFixed(1)} tok/s  speedup=${run.speedupVsBaseline.toFixed(2)}×`,
  );
  for (const s of run.perPrompt) {
    console.log(
      `    ${s.tokens}t α=${s.acceptance.toFixed(2)} ${s.tokensPerSecond.toFixed(1)} t/s  | mean: tot=${s.meanTotalMs?.toFixed(1)} verify=${s.meanTargetVerifyMs?.toFixed(1)} drLC=${s.meanDraftLastCommitMs?.toFixed(1)}`,
    );
  }
  return run;
}

async function main(): Promise<void> {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  console.log(`Spectra benchmark runner`);
  console.log(`  demo url:   ${DEMO_URL}`);
  console.log(`  γ sweep:    ${GAMMAS.join(", ")}`);
  console.log(`  headless:   ${HEADLESS}`);
  console.log(`  output dir: ${OUTPUT_DIR}`);

  const utcStartedAt = new Date().toISOString();
  const browser = await chromium.launch({
    headless: HEADLESS,
    args: [
      "--enable-unsafe-webgpu",
      "--enable-features=Vulkan",
      // For Linux GPU containers (Modal etc.) — these are no-ops on macOS.
      "--use-vulkan=swiftshader",
      "--no-sandbox",
    ],
  });

  const runs: GammaRun[] = [];
  try {
    for (const g of GAMMAS) {
      const page = await setupPage(browser);
      try {
        const run = await runOnce(page, g);
        runs.push(run);
      } finally {
        await page.context().close();
      }
    }
  } finally {
    await browser.close();
  }

  const result: BenchResult = {
    schema: 1,
    utcStartedAt,
    utcEndedAt: new Date().toISOString(),
    hostname: os.hostname(),
    platform: os.platform(),
    arch: os.arch(),
    cpus: os.cpus().length,
    totalMemMB: Math.round(os.totalmem() / (1024 * 1024)),
    demoUrl: DEMO_URL,
    baselineTokPerSec: BASELINE_TOK_PER_SEC,
    runs,
  };
  const slug = `${os.hostname().replace(/\s+/g, "_")}_${utcStartedAt.replace(/[:.]/g, "-")}.json`;
  const outPath = path.join(OUTPUT_DIR, slug);
  fs.writeFileSync(outPath, JSON.stringify(result, null, 2));
  console.log(`\nWrote ${outPath}`);

  // Markdown table summary on stdout.
  console.log("\n=== SUMMARY ===");
  console.log(`| γ | mean accept | mean tok/s | speedup vs ${BASELINE_TOK_PER_SEC} t/s |`);
  console.log(`|---|---|---|---|`);
  for (const r of runs) {
    console.log(
      `| ${r.gamma} | ${r.meanAcceptance.toFixed(3)} | ${r.meanTokensPerSecond.toFixed(1)} | ${r.speedupVsBaseline.toFixed(2)}× |`,
    );
  }
}

main().catch((e) => {
  console.error("FATAL:", e);
  process.exit(1);
});
