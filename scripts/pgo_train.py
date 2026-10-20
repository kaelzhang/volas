#!/usr/bin/env python3
"""Fixed-operation training. Never calibrate weights using elapsed time."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PASSES = 16
STAGES = (
    "batch",
    "append",
    "first-cache",
    "tf-live",
    "streams",
    "boundaries",
    "range",
    "exceptional",
)
SEED = 20261020
# Frozen before adoption measurements; none of these enter the training profile.
HOLDOUT = {
    "seed": 20261021,
    "lengths": [257, 8191, 65537],
    "directives": ["ma:7", "ma:63", "atr:21", "ema:37", "ma:7>ema:37"],
    "windows": [47, 127],
    "fold": "5m->1h",
    "inputs": ["reversal", "flat", "sparse-missing", "arrow-sliced-null-chunked"],
}


def load_site(site, expected):
    sys.path[:0] = [str(site), str(ROOT)]
    native = importlib.import_module("volas_rs.volas_rs")
    from scripts.build_release import verify_library

    verify_library(Path(native.__file__), site, expected)
    return native


def train(stage=None):
    import numpy as np
    from volas import DataFrame
    from test import bench_workloads as W, bench_tf_live as T, bench_first_cache as F
    from scripts.count_indicators import count

    inventory = json.loads((ROOT / "test/pgo_inventory.json").read_text())
    assert set(inventory) == set(count()["forms"]), "training inventory drift"
    forms = list(inventory.values())
    assert len(forms) == 254, "public inventory changed: review training budget"
    stages = []
    checks = hashlib.sha256()

    def record(name, fn):
        wall, cpu = time.monotonic(), time.process_time()
        operations = fn()
        stages.append(
            {
                "stage": name,
                "operations": operations,
                "wall_s": time.monotonic() - wall,
                "cpu_s": time.process_time() - cpu,
            }
        )
        print(json.dumps(stages[-1]), flush=True)

    def check(values):
        a = np.asarray(values, dtype="float64")
        checks.update(np.isnan(a).tobytes())
        checks.update(np.nan_to_num(a).tobytes())

    def batch():
        for n in (2048, 20000):
            data = W.generated_ohlcv(n, SEED)
            for d in forms:
                for _ in range(2):
                    df = DataFrame(data)
                    result = df.exec(d, dtype="float64")
                    assert len(result) == n, d
                    check(result)
        return len(forms) * 4

    def append():
        data = W.generated_ohlcv(2048, SEED)
        for d in forms:
            run, setup = W.append(data, d)
            for _ in range(16):
                args, kw = setup()
                run(*args, **kw)
                expected = DataFrame(data).exec(d, dtype="float64")
                np.testing.assert_array_equal(
                    args[0][d].to_numpy(dtype="float64"), expected
                )
        return len(forms) * 16

    def first():
        ids = [f"{d}@n=100000" for d in F.FIRST_CACHE_INDICATORS]
        ids += [
            f"{d}@n=1000000"
            for d in (
                "ma:20",
                "ema:20",
                "atr:14",
                "boll.upper:20",
                "stddev:20",
                "linearreg:14",
            )
        ]
        for d in ids:
            run, setup = F.build(d)
            args, kw = setup()
            run(*args, **kw)
        return len(ids)

    def live():
        data = W.generated_ohlcv(2048, SEED)
        for phase in T.TF_LIVE_PHASES:
            for d in T.TF_LIVE_INDICATORS:
                run, setup = T.build(data, W.COLUMNS, d, phase)
                for _ in range(32):
                    args, kw = setup()
                    run(*args, **kw)
        return len(T.TF_LIVE_PHASES) * len(T.TF_LIVE_INDICATORS) * 32

    def streams():
        ds = ["ma:30", "ema:20", "atr:14", "rsi:14", "macd", "ma:30>ema:20"]
        configs = [
            (4096, ds, None, None),
            (4096, ds, 127, None),
            (20000, ["atr:14"], 30, "15m"),
        ]
        for n, directives, window, tf in configs:
            run, setup = W.stream(
                {c: a for c, a in W.generated_ohlcv(n, SEED).items() if c in W.COLUMNS},
                directives,
                window,
                tf,
            )
            args, kw = setup()
            df = run(*args, **kw)
            for d in directives:
                check(df[d].to_numpy(dtype="float64"))
            if window:
                assert len(df) <= window
            _ = df[-20:][directives[0]]
        return sum(c[0] - 1 for c in configs)

    def boundaries():
        for n, reps in ((4096, 8), (1000000, 2)):
            for fn in W.interop(n).values():
                for _ in range(reps):
                    assert fn() is not None
        data = W.generated_ohlcv(2048, SEED)
        df = DataFrame(data)
        for _ in range(64):
            _ = (
                DataFrame(data),
                df["close"],
                df[100:1900],
                df[df["close"] > 100],
                df.copy(),
                df[-100:],
                df[800:1000],
            )
            df["scratch"] = data["close"]
            df.iat[100, 0] = 1.5
            _ = (
                df.to_numpy(),
                df["close"].to_numpy(),
                DataFrame({"close": data["close"][::2]}),
            )
        return 6 * 10 + 64 * 12

    def ranges():
        t = np.arange(1000000, dtype=float)
        c = 100 + np.sin(t * 0.03) + np.cos(t * 0.017)
        data = {"high": c + 1.5, "low": c - 0.5, "close": c}
        for d in ("atr:14", "natr:14", "adx:14", "sar", "sarext"):
            check(DataFrame(data).exec(d, dtype="float64"))
            run, setup = W.append({k: v[:2000] for k, v in data.items()}, d)
            for _ in range(32):
                args, kw = setup()
                run(*args, **kw)
        return 5 * 33

    def exceptional():
        for shard in ("trend", "flat", "missing"):
            data = W.generated_ohlcv(2048, SEED)
            if shard == "flat":
                for c in W.COLUMNS[:-1]:
                    data[c][:] = 100
            if shard == "missing":
                for c in W.COLUMNS:
                    data[c][[50, 100, 150]] = [np.nan, np.inf, -np.inf]
            for d in forms:
                check(DataFrame(data).exec(d, dtype="float64"))
        return 3 * len(forms)

    for name, fn in [
        ("batch", batch),
        ("append", append),
        ("first-cache", first),
        ("tf-live", live),
        ("streams", streams),
        ("boundaries", boundaries),
        ("range", ranges),
        ("exceptional", exceptional),
    ]:
        if stage is None or stage == name:
            record(name, lambda fn=fn: sum(fn() for _ in range(TRAIN_PASSES)))
    return {
        "seed": SEED,
        "passes": TRAIN_PASSES,
        "directives": forms,
        "inventory": inventory,
        "inventory_total": len(forms),
        "stages": stages,
        "result_sha256": checks.hexdigest(),
        "holdout": HOLDOUT,
        "data_sha256": {
            str(n): hashlib.sha256(
                b"".join(a.tobytes() for a in W.generated_ohlcv(n, SEED).values())
            ).hexdigest()
            for n in (2048, 20000)
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", type=Path, required=True)
    ap.add_argument("--native-sha256", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--stage", choices=STAGES)
    ap.add_argument("--identity", type=Path, required=True)
    args = ap.parse_args()
    ident = json.loads(args.identity.read_text())
    for name, expected in ident["training"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"training provenance mismatch: {name}")
    native = load_site(args.site.resolve(), args.native_sha256)
    for line in (ROOT / "scripts/pgo-requirements.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            name, version = line.split("==")
            assert importlib.metadata.version(name) == version, (
                f"training dependency mismatch: {name}"
            )
    start = time.monotonic()
    if args.stage is None:
        parts = []
        for stage in STAGES:
            phase_output = args.output.with_name(
                args.output.stem + "." + stage + ".json"
            )
            env = dict(os.environ)
            if "LLVM_PROFILE_FILE" in env:
                raw = Path(env["LLVM_PROFILE_FILE"])
                env["LLVM_PROFILE_FILE"] = str(raw.with_name(stage + "-%m-%p.profraw"))
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--site",
                    str(args.site),
                    "--native-sha256",
                    args.native_sha256,
                    "--output",
                    str(phase_output),
                    "--stage",
                    stage,
                    "--identity",
                    str(args.identity),
                ],
                env=env,
                check=True,
                timeout=max(1, 600 - (time.monotonic() - start)),
            )
            parts.append(json.loads(phase_output.read_text()))
        result = parts[0]
        result["stages"] = [s for p in parts for s in p["stages"]]
        result["stage_checksums"] = {
            s: p["result_sha256"] for s, p in zip(STAGES, parts)
        }
    else:
        result = train(args.stage)
    result.update(
        native=native.__file__,
        native_sha256=args.native_sha256,
        wall_s=time.monotonic() - start,
    )
    if result["wall_s"] > 600:
        raise ValueError("training exceeds 10 minute cap")
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
