"""Checkpoint-overhead micro-benchmark.

Measures the wall-clock cost of running shadow-branching on a synthetic
agent workload (sequential file rewrites) for various repository sizes.

Usage:
    python bench_checkpoint.py --files 50 200 --turns 25 --output results/

Reports:
    - mean / p95 per-checkpoint latency (ms)
    - total wall-clock for the run (s)
    - .git/objects/ size delta and pack count
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

HERE = Path(__file__).resolve().parent
PKG_ROOT = (
    HERE / ".." / ".." / ".." / "aar-extensions-registry" / "packages" / "aar-ext-shadow-branching"
).resolve()
if PKG_ROOT.exists():
    sys.path.insert(0, str(PKG_ROOT))

import aar_ext_shadow_branching as ext  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
LOG = logging.getLogger("checkpoint-bench")


class FakeAPI:
    def __init__(self) -> None:
        self.handlers: dict[str, list] = {}
        self.commands: dict[str, tuple[str, Any]] = {}

    def on(self, event):
        def deco(fn):
            self.handlers.setdefault(event, []).append(fn)
            return fn

        return deco

    def command(self, name, *, description=""):
        def deco(fn):
            self.commands[name] = (description, fn)
            return fn

        return deco

    def tool(self, *a, **kw):
        return lambda fn: fn

    def append_system_prompt(self, _text):
        pass


class FakeSession:
    def __init__(self, sid: str) -> None:
        self.session_id = sid
        self.events: list = []
        self.step_count = 0
        self.metadata: dict = {}


class FakeCtx:
    def __init__(self, session: FakeSession) -> None:
        self.session = session
        self.logger = LOG


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False)


def _init_repo(path: Path, file_count: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _git("init", "-b", "main", cwd=path)
    _git("config", "user.name", "Bench", cwd=path)
    _git("config", "user.email", "bench@example.com", cwd=path)
    _git("config", "commit.gpgsign", "false", cwd=path)
    src = path / "src"
    src.mkdir()
    for i in range(file_count):
        (src / f"f_{i:05}.py").write_text(
            f"# generated file {i}\n\ndef f{i}():\n    return {i}\n",
            encoding="utf-8",
        )
    _git("add", "-A", cwd=path)
    _git("commit", "-m", "initial", cwd=path)


def dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total


def pack_count(repo: Path) -> int:
    pack_dir = repo / ".git" / "objects" / "pack"
    if not pack_dir.exists():
        return 0
    return sum(1 for p in pack_dir.iterdir() if p.suffix == ".pack")


def run_one(file_count: int, turns: int, with_shadow: bool) -> dict:
    saved_cwd = os.getcwd()
    try:
        tmp = tempfile.mkdtemp(prefix=f"ckpt-bench-{file_count}-")
        repo = Path(tmp) / "repo"
        _init_repo(repo, file_count)
        os.chdir(repo)
        objects_dir = repo / ".git" / "objects"
        size_before = dir_size(objects_dir)

        api = FakeAPI()
        if with_shadow:
            ext.register(api)
            ctx = FakeCtx(FakeSession("bench"))
            for h in api.handlers["session_start"]:
                h(None, ctx)
        else:
            ctx = FakeCtx(FakeSession("bench"))

        latencies: list[float] = []
        t_start = time.perf_counter()
        for i in range(turns):
            target = repo / "src" / f"f_{i % file_count:05}.py"
            target.write_text(
                f"# turn {i}\n\ndef f{i % file_count}():\n    return {i}\n",
                encoding="utf-8",
            )
            t0 = time.perf_counter()
            if with_shadow:
                for h in api.handlers["tool_result"]:
                    h(SimpleNamespace(tool_name="write_file"), ctx)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        wall = time.perf_counter() - t_start

        size_after = dir_size(objects_dir)
        packs_before_gc = pack_count(repo)
        if with_shadow:
            _git("gc", "--quiet", cwd=repo)
        packs_after_gc = pack_count(repo)
        size_after_gc = dir_size(objects_dir)

        latencies_nonzero = [latency for latency in latencies if latency > 0]
        result = {
            "files": file_count,
            "turns": turns,
            "shadow": "on" if with_shadow else "off",
            "wall_s": round(wall, 3),
            "checkpoint_mean_ms": (
                round(statistics.mean(latencies_nonzero), 3) if latencies_nonzero else 0.0
            ),
            "checkpoint_p95_ms": (
                round(statistics.quantiles(latencies_nonzero, n=20)[-1], 3)
                if len(latencies_nonzero) >= 20
                else (round(max(latencies_nonzero), 3) if latencies_nonzero else 0.0)
            ),
            "objects_kb_before": round(size_before / 1024, 1),
            "objects_kb_after": round(size_after / 1024, 1),
            "objects_kb_after_gc": round(size_after_gc / 1024, 1),
            "packs_before_gc": packs_before_gc,
            "packs_after_gc": packs_after_gc,
        }
        return result
    finally:
        os.chdir(saved_cwd)
        try:
            import shutil as _shutil

            _shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", nargs="+", type=int, default=[50, 200])
    ap.add_argument("--turns", type=int, default=25)
    ap.add_argument("--output", type=Path, default=Path("results"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for fc in args.files:
        for shadow in (False, True):
            print(f"running files={fc} turns={args.turns} shadow={shadow} ...")
            rows.append(run_one(fc, args.turns, shadow))

    out_csv = args.output / "checkpoint_overhead.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_csv}")

    # Markdown summary
    print("\n## Checkpoint-overhead micro-benchmark\n")
    print("| files | turns | shadow | wall (s) | mean ms | p95 ms | objects kb (post-gc) |")
    print("|---:|---:|:---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['files']} | {r['turns']} | {r['shadow']} | "
            f"{r['wall_s']} | {r['checkpoint_mean_ms']} | "
            f"{r['checkpoint_p95_ms']} | {r['objects_kb_after_gc']} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
