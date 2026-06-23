# Shadow Branching - Benchmark & Ablation Harness

This directory contains reproducible micro-benchmarks and a single
controlled ablation supporting the claims in the paper.

The headline question we want to answer is:

> **Does the session–branch reload mechanism actually prevent the
> divergence between in-memory events and on-disk files that we claim it
> does - and what is the measurable cost of running with reload enabled?**

The harness here is intentionally *small* and *runnable from a laptop*. It
is not an attempt to reproduce SWE-bench; it is an ablation that isolates
the one mechanism the paper introduces.

---

## What's measured

### 1. `bench_checkpoint.py` - task-level overhead

Simulates a configurable agent run (25 tool calls by default, matching the
checked-in results) on a synthetic repository. For each configuration we
measure:

* total wall-clock time **without** shadow-branching (control),
* total wall-clock time **with** shadow-branching enabled,
* mean and 95th-percentile per-checkpoint latency,
* on-disk size of `.git/objects/` after the run,
* number of pack files before/after `git gc`.

Output: a CSV in `results/checkpoint_overhead.csv` plus a summary table
on stdout.

### 2. `bench_reload_ablation.py` - the consistency invariant ablation

This is the more important one. We construct a session that:

1. Performs `T₁` tool calls on the canonical shadow branch.
2. Issues `/branch` (rewinding to checkpoint `T₁/2`), then performs
   `T₂` tool calls on the new shadow.
3. Issues `/switch` back to the preserved fork.
4. Performs `T₃` more tool calls on the fork.

We compare two conditions:

* **`reload=on`** - the protocol as described in the paper.
* **`reload=off`** - the `ReloadSession` step is replaced with a no-op.

For each condition we record, after every tool call following a branch
operation:

* `event_path_set` - the set of file paths mentioned in the in-memory
  event stream (extracted from message contents and tool-call arguments
  via a regex over path-like tokens).
* `disk_path_set` - the set of files actually present on the
  currently checked-out branch (`git ls-files`).
* `divergence = |event_path_set \ disk_path_set|` - the number of paths
  the LLM "knows about" that no longer exist on disk.

The ablation hypothesis is: with `reload=off`, divergence is non-zero
and grows with the number of branch operations; with `reload=on`,
divergence is always zero.

Output: `results/reload_ablation.csv` with columns
`condition, branch_op_index, divergence, total_events, total_files`,
plus a Markdown summary suitable for inclusion in the paper.

---

## Running

```bash
cd docs/shadow_branching_paper/benchmark
pip install -e ../../../aar-extensions-registry/packages/aar-ext-shadow-branching
python bench_reload_ablation.py --turns 16 --branches 3 --output results/
python bench_checkpoint.py --files 50 200 --turns 25 --output results/
```

Each script prints a Markdown table on completion that can be pasted
directly into the paper's evaluation section.

---

## What this is *not*

* Not a full SWE-bench-style evaluation. The harness uses synthetic
  edits (random small file rewrites) so it isolates the protocol's
  overhead from LLM cost, prompt-tuning effects, and task difficulty.
* Not a replacement for the 56 functional tests in
  `aar-extensions-registry/packages/aar-ext-shadow-branching/tests/`.
  Those verify *correctness*; these scripts verify *cost* and the
  ablation's central claim.

A future extension is to wire a small cohort of SWE-bench-Lite tasks
through an actual agent loop with reload toggled, but that requires API
keys and a tens-of-minutes runtime per condition and is out of scope for
this initial harness.
