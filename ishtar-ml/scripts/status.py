#!/usr/bin/env python3
"""What is running, how far along, and how much longer.

    python scripts/status.py            # once
    python scripts/status.py --watch    # refresh every 20 s

Reads `progress.json`, which `train.py` writes next to each checkpoint every logging
interval, so it works across shells and survives losing the terminal a run started in.
Runs whose heartbeat has gone stale are reported as stalled rather than silently shown as
in-progress — a job that died at step 300 otherwise looks identical to one still working.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STALE_S = 180


def human(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 5400:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


def bar(frac: float, width: int = 24) -> str:
    filled = int(round(frac * width))
    return "█" * filled + "·" * (width - filled)


LOG_RE = None


def from_log(log: Path, ckpt_name: str) -> dict | None:
    """Fall back to parsing a training log.

    `progress.json` only exists for runs started after the heartbeat was added, and a
    status tool that cannot see the run you are currently waiting on is not much use.
    """
    import re

    try:
        lines = log.read_text(errors="ignore").splitlines()
    except Exception:
        return None
    steps = total = None
    rate = 0.0
    loss: dict[str, float] = {}
    data = "?"
    finished = False
    for line in lines:
        m = re.search(r"steps=(\d+)", line)
        if m:
            total = int(m.group(1))
        m = re.match(r"^step\s+(\d+).*?\((\d+\.\d+)s/step\)", line)
        if m:
            steps, rate = int(m.group(1)), float(m.group(2))
            loss = {k: float(v) for k, v in re.findall(r"(\w+)=([0-9.]+)", line)
                    if k not in ("lr", "step")}
        if line.startswith("train:"):
            data = "real tiles"
        # A log that has moved past training says so; without this the last `step` line
        # sits there looking like a run still 47 seconds from the end, forever.
        if line.startswith("final val") or line.startswith("=== EVAL"):
            finished = True
    if steps is None or not total:
        return None
    if finished:
        steps = total
    return {"phase": "?", "data": data, "step": steps, "steps": total,
            "seconds_per_step": rate, "elapsed_s": steps * rate,
            "eta_s": 0 if finished else rate * (total - steps), "loss": loss,
            "updated": log.stat().st_mtime}


def running_jobs() -> list[str]:
    try:
        out = subprocess.run(["ps", "-Ao", "command"], capture_output=True, text=True,
                             timeout=10).stdout
    except Exception:
        return []
    wanted = (("train.py", "training"), ("data.ingest", "ingest"),
              ("calibrate_geometry", "incidence calibration"),
              ("region_product", "region product"), ("eval.", "evaluation"),
              ("curl", "download"))
    seen = []
    for line in out.splitlines():
        for needle, label in wanted:
            if needle in line and "grep" not in line and label not in seen:
                seen.append(label)
    return seen


def report() -> None:
    now = time.time()
    print(f"  ISHTAR status — {time.strftime('%H:%M:%S')}\n")

    found: dict[str, dict] = {}
    for f in sorted(ROOT.glob("runs/*/progress.json")):
        try:
            found[f.parent.name] = json.loads(f.read_text())
        except Exception:
            pass
    for log in sorted(ROOT.glob("runs/*/train.log")):
        if log.parent.name not in found:
            p = from_log(log, log.parent.name)
            if p:
                found[log.parent.name] = p

    if not found:
        print("  no training runs have reported progress yet")
    live = []
    for name, p in sorted(found.items()):
        f = ROOT / "runs" / name / "progress.json"
        age = now - p.get("updated", 0)
        frac = p["step"] / max(p["steps"], 1)
        # Completion wins over staleness: a finished run has no reason to keep writing,
        # so judging it by heartbeat age would report every successful run as stalled.
        if frac >= 0.999:
            state = "done"
        elif age > STALE_S:
            state = "stalled"
        else:
            state = "running"
        if state == "running":
            live.append(p)
        eta = "—" if state != "running" else human(p["eta_s"])
        print(f"  {name:<18} {bar(frac)} {p['step']:>5}/{p['steps']:<5} "
              f"{frac:5.0%}  {state:<8} eta {eta}")
        if state == "running":
            terms = ", ".join(f"{k}={v:g}" for k, v in list(p["loss"].items())[:5])
            print(f"  {'':<18} {p['seconds_per_step']:.2f}s/step on {p['data']}   {terms}")
        elif state == "stalled":
            print(f"  {'':<18} last heartbeat {human(age)} ago — the process is gone")

    jobs = running_jobs()
    print(f"\n  background jobs: {', '.join(jobs) if jobs else 'none'}")
    if live:
        longest = max(p["eta_s"] for p in live)
        print(f"  everything finishes in about {human(longest)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--every", type=int, default=20)
    a = ap.parse_args()
    if not a.watch:
        report()
        return
    try:
        while True:
            print("\033[2J\033[H", end="")
            report()
            time.sleep(a.every)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
