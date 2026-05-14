/**
 * Sampling — host-side helpers for greedy / top-p / top-k sampling.
 *
 * Most sampling at hot path runs on GPU via `sample_with_top_p` PackedFunc.
 * Helpers here are for CPU-side pre/post processing and unit tests.
 *
 * Status: skeleton.
 */

/** Argmax of a logit array (greedy decode, CPU). */
export function greedy(logits: Float32Array | Float64Array): number {
  let best = 0;
  let bestVal = logits[0];
  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > bestVal) {
      bestVal = logits[i];
      best = i;
    }
  }
  return best;
}

/** Numerically-stable softmax (CPU, fp32). */
export function softmax(logits: Float32Array, temperature = 1.0): Float32Array {
  const out = new Float32Array(logits.length);
  let maxLogit = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    const l = logits[i] / temperature;
    if (l > maxLogit) maxLogit = l;
  }
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const v = Math.exp(logits[i] / temperature - maxLogit);
    out[i] = v;
    sum += v;
  }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}
