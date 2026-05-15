# `spectra/bench`

Playwright-driven benchmark runner for Spectra. Sweeps γ ∈ {2, 3, 4} (configurable),
records per-prompt acceptance + tok/s + per-section timing breakdown, writes
machine-tagged JSON to `results/`.

## Local run (M3 Pro Chrome)

In one terminal — start the demo dev server:

```bash
cd spectra/demo
pnpm install         # only needed once
pnpm dev             # serves http://localhost:5173/
```

In another terminal — run the benchmark:

```bash
cd spectra/bench
pnpm install         # only needed once
pnpm bench:install   # downloads Playwright's bundled Chromium (~120 MB; only once)
pnpm bench
```

Output:

- A markdown table on stdout with γ × {acceptance, tok/s, speedup}
- A `results/<host>_<UTC>.json` file with full per-prompt + per-round detail

## Configuration via env vars

| Var | Default | Notes |
|---|---|---|
| `SPECTRA_DEMO_URL` | `http://localhost:5173/` | Where the demo is served |
| `SPECTRA_GAMMAS` | `2,3,4` | Comma-separated γ values to sweep |
| `SPECTRA_HEADLESS` | `true` | Set to `false` to watch the runs in a visible Chrome window |

## Result schema (`results/*.json`)

```jsonc
{
  "schema": 1,
  "utcStartedAt": "...", "utcEndedAt": "...",
  "hostname": "Maruthis-MacBook-Pro",
  "platform": "darwin", "arch": "arm64", "cpus": 11, "totalMemMB": 18432,
  "demoUrl": "http://localhost:5173/",
  "baselineTokPerSec": 35,
  "runs": [
    {
      "gamma": 2,
      "perPrompt": [
        {
          "prompt": "Compose a haiku…",
          "tokens": 64, "rounds": 36, "acceptance": 0.403,
          "tokensPerSecond": 30.1, "decodeMs": 2124,
          "meanTotalMs": 49.3, "meanTargetVerifyMs": 18.5,
          "meanDraftLastCommitMs": 11.2, ...
        }, ...
      ],
      "meanAcceptance": 0.598,
      "meanTokensPerSecond": 43.0,
      "speedupVsBaseline": 1.23,
      "errors": []
    }, ...
  ]
}
```

## Reproducibility

- Each result file is hostname + UTC-timestamp tagged.
- `baselineTokPerSec=35` is the M3 Pro Chrome stock-WebLLM-greedy decode rate
  measured in Phase 0a A3 (follow-up). Re-measure if hardware changes.
- The first run of each γ pays the model-load cost (~30s for HF cache-cold weights).
  Subsequent runs reuse the WebLLM IndexedDB cache. To force a clean cold start,
  clear the browser's IndexedDB for `localhost:5173` between runs.

## Troubleshooting

- **`net::ERR_CONNECTION_REFUSED`**: the demo dev server isn't running. Start it
  per the instructions above.
- **WebGPU not available**: confirm Chromium has WebGPU (it does in Playwright's
  bundled build by default with `--enable-unsafe-webgpu`). For Linux/server
  runs, GPU passthrough + Vulkan setup is required (see `modal_bench.py`).
- **Sourcemap warnings from Vite**: harmless; web-llm's source map files
  reference paths outside the install dir.
