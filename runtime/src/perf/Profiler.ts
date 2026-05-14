/**
 * Profiler — lightweight per-event timing.
 *
 * Status: skeleton (basic implementation works; integration with SpecController
 * comes later).
 */

export interface ProfilerEvent {
  readonly name: string;
  readonly durationMs: number;
  readonly t0: number;
  readonly t1: number;
}

export class Profiler {
  private readonly events: ProfilerEvent[] = [];
  private readonly clock: () => number =
    typeof performance !== "undefined" ? () => performance.now() : () => Date.now();

  /** Time an async operation. Returns its result; appends an event. */
  async time<T>(name: string, fn: () => Promise<T>): Promise<T> {
    const t0 = this.clock();
    try {
      return await fn();
    } finally {
      const t1 = this.clock();
      this.events.push({ name, durationMs: t1 - t0, t0, t1 });
    }
  }

  /** All collected events (ordered by t0). */
  getEvents(): readonly ProfilerEvent[] {
    return this.events;
  }

  /** Sum of durations grouped by event name. */
  summary(): Record<string, { count: number; totalMs: number; meanMs: number }> {
    const acc: Record<string, { count: number; totalMs: number; meanMs: number }> = {};
    for (const e of this.events) {
      const cur = acc[e.name] ?? { count: 0, totalMs: 0, meanMs: 0 };
      cur.count += 1;
      cur.totalMs += e.durationMs;
      cur.meanMs = cur.totalMs / cur.count;
      acc[e.name] = cur;
    }
    return acc;
  }

  reset(): void {
    this.events.length = 0;
  }
}
