# Dockerfile — headless Chrome with WebGPU + Spectra demo built-in.
#
# Used by `modal_bench.py` to run cross-vendor benchmarks on Nvidia GPUs.
# The image:
#   1. Installs Chromium + Vulkan stack (Chrome's WebGPU on Linux uses Vulkan).
#   2. Bundles the demo's static `dist/` build so the runner can serve it locally.
#   3. Bundles the bench/runner.ts compiled output.
#
# The Modal-side script (`modal_bench.py`) attaches an Nvidia GPU + Vulkan ICD,
# then `node bench/runner.js` against `http://localhost:8000` (a tiny http server
# serving demo/dist).

FROM mcr.microsoft.com/playwright:v1.50.0-jammy

# Vulkan ICD for Nvidia (the Playwright base has Vulkan loader; we add the ICD).
RUN apt-get update && apt-get install -y --no-install-recommends \
        vulkan-tools \
        mesa-vulkan-drivers \
        libnvidia-gl-* \
        nvidia-vulkan-icd \
        ca-certificates \
        curl \
        wget \
        gnupg \
        xz-utils \
    && rm -rf /var/lib/apt/lists/* \
    || true   # the libnvidia-gl-* / nvidia-vulkan-icd lines are best-effort; Modal images may have them already

# Install Node 22 (Playwright's bundled Chromium needs a modern Node).
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /spectra

# Bundle the static demo + benchmark runner.
# (Built locally before image build via `pnpm build` in spectra/demo and spectra/bench.)
COPY demo/dist /spectra/demo
COPY bench/runner.ts /spectra/bench/runner.ts
COPY bench/package.json /spectra/bench/package.json
COPY bench/tsconfig.json /spectra/bench/tsconfig.json

WORKDIR /spectra/bench
RUN npm install --production && \
    npx playwright install chromium

# Default entrypoint: serve demo/dist on :8000, run benchmark, dump results to stdout.
CMD ["bash", "-c", "\
    npx http-server /spectra/demo -p 8000 --silent & \
    sleep 2 && \
    SPECTRA_DEMO_URL=http://localhost:8000/ \
    SPECTRA_HEADLESS=true \
    npx tsx runner.ts \
"]
