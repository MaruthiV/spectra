/**
 * StepLogger — structured per-spec-step logs.
 *
 * Each spec round produces one record. These get fed to the demo UI's
 * acceptance heatmap and to the benchmark harness CSVs.
 *
 * Status: skeleton (interface defined; no persistence yet).
 */

export interface StepRecord {
  /** 0-based spec round index within the request. */
  readonly round: number;
  /** Drafts produced this round (γ). */
  readonly draftsProposed: number;
  /** Drafts accepted (0..γ). */
  readonly draftsAccepted: number;
  /** Per-position accept/reject (length γ). True = accepted. */
  readonly perPositionAccept: readonly boolean[];
  /** Wall-clock duration of this round (ms). */
  readonly latencyMs: number;
  /** First-child / next-sibling tree shape, for tree variant. Empty for chain. */
  readonly treeShape?: {
    readonly firstChild: readonly number[];
    readonly nextSibling: readonly number[];
  };
}

export class StepLogger {
  private readonly records: StepRecord[] = [];

  log(record: StepRecord): void {
    this.records.push(record);
  }

  /** Read-only view of all logged records. */
  getRecords(): readonly StepRecord[] {
    return this.records;
  }

  /** Clear records (between requests). */
  reset(): void {
    this.records.length = 0;
  }

  /** Rolling cumulative acceptance ratio (sum accepted / sum proposed). */
  cumulativeAcceptance(): number {
    let proposed = 0;
    let accepted = 0;
    for (const r of this.records) {
      proposed += r.draftsProposed;
      accepted += r.draftsAccepted;
    }
    return proposed > 0 ? accepted / proposed : 0;
  }
}
