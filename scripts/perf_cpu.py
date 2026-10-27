"""CPU paired A/B of exact current benchmark workloads, keyed by saved pytest rows."""

import importlib
import inspect
import json
import os
import pathlib
import statistics
import subprocess

import sys

from build_pair import validate
from perf_runtime import Capture, allocator_settings, environment, measure

if __package__:
    from . import build_config as processes
else:
    import build_config as processes

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / "target/numba"))


if sys.argv[1] == "worker":
    sys.path[:0] = [sys.argv[2], str(ROOT)]
    from test import test_benchmark as B
    import volas_rs

    native = importlib.import_module("volas_rs.volas_rs")
    if str(pathlib.Path(sys.argv[2]).resolve()) not in native.__file__:
        raise AssertionError(native.__file__)
    if B.VolasDataFrame is not native.DataFrame:
        raise AssertionError()
    standard = B.VolasDataFrame(
        {**{c: B.ARR[c] for c in B.COLUMNS}, "periods": B.PERIODS}
    )
    ranges = None
    extended = {}
    interop = {}
    cache = {}
    print(
        json.dumps(
            {
                "ready": True,
                "module": volas_rs.__file__,
                "native": native.__file__,
                "pid": os.getpid(),
            }
        ),
        flush=True,
    )
    for line in sys.stdin:
        req = json.loads(line)
        row = req["row"]
        name = row["name"]
        reps = req["reps"]
        if name not in cache:
            function = name.split("[", 1)[0]
            module = (
                B
                if hasattr(B, function)
                else importlib.import_module("test.test_benchmark_true_range")
            )
            target = getattr(module, function)
            p = row.get("params") or {}
            cap = Capture()
            args = {}
            for arg in inspect.signature(target).parameters:
                if arg == "benchmark":
                    args[arg] = cap
                elif arg == "states":
                    args[arg] = {"volas": standard}
                elif arg == "length_states":
                    n = int(p["length"])
                    if n not in extended:
                        data = B._generated_ohlcv(n)
                        extended[n] = {"data": data, "volas": B.VolasDataFrame(data)}
                    args[arg] = extended
                elif arg == "interop_registry":
                    length = row.get("extra_info", {}).get("length") or p.get(arg)
                    if length is None:
                        raise AssertionError((name, p))
                    n = int(length)
                    if n not in interop:
                        from test.bench_interop import build

                        interop[n] = build(n, B.pl)
                    args[arg] = (n, interop[n])
                elif arg == "range_prices":
                    if ranges is None:
                        fixture = getattr(module, arg)
                        factory = getattr(fixture, "__wrapped__", None) or getattr(
                            fixture, "_fixture_function", None
                        )
                        ranges = factory()
                    args[arg] = ranges
                else:
                    args[arg] = p[arg]
            target(**args)
            if cap.fn is None:
                raise AssertionError(name)
            for _ in range(5):
                a, k = cap.setup() if cap.setup else (cap.args, cap.kwargs)
                cap.fn(*a, **k)
            cache[name] = cap
        cap = cache[name]
        print(json.dumps(measure(cap, reps)), flush=True)
    sys.exit()
base, head, out, manifest = sys.argv[1:5]

provenance = validate(base, head, os.environ.get("PERF_COMPARE", "regression"))
pathlib.Path(out + ".builds.json").write_text(json.dumps(provenance, indent=2))
pathlib.Path(out + ".runtime.json").write_text(json.dumps(allocator_settings(), indent=2))
requests = json.loads(pathlib.Path(manifest).read_text())
workers = []
for site in (base, head):
    p = processes.popen_child(
        [sys.executable, __file__, "worker", site],
        env=environment(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    ready = json.loads(p.stdout.readline())
    if str(pathlib.Path(site).resolve()) not in ready["module"]:
        raise AssertionError(ready)
    workers.append(p)
    print(ready, flush=True)
results = []
try:
    for request in requests:
        row = request["row"]
        name = row["name"]
        function = name.split("[", 1)[0]
        params = row.get("params") or {}
        reps = request.get("reps", 300)
        if "reps" not in request:
            if function == "test_first_cache":
                reps = 3 if int(params["indicator"].split("@n=")[1]) >= 1000000 else 20
            elif function in (
                "test_append",
                "test_tf_live",
                "test_coverage_after_append",
                "test_range_single_row",
            ):
                reps = 100
            elif function in ("test_windowed_stream", "test_range_batch_million"):
                reps = 1 if function == "test_windowed_stream" else 3
            elif (
                function == "test_coverage_extended" and int(params["length"]) >= 20000
            ):
                reps = 100
        sides = [[], []]
        for pair in range(200):
            for side in [0, 1] if pair % 2 == 0 else [1, 0]:
                p = workers[side]
                p.stdin.write(json.dumps({"row": row, "reps": reps}) + "\n")
                p.stdin.flush()
                result = p.stdout.readline()
                if not result:
                    raise AssertionError((name, side, p.poll()))
                sides[side].append(json.loads(result))
        result = {"item": name, "params": params, "reps": reps, "samples": sides}
        for clock in ("cpu", "wall"):
            ratios = [h[clock] / b[clock] for b, h in zip(*sides)]
            q = statistics.quantiles(ratios, n=4, method="inclusive")
            result[clock] = {
                "median": statistics.median(ratios),
                "p25": q[0],
                "p75": q[2],
                "ratios": ratios,
                "base_ns": statistics.median(v[clock] for v in sides[0]),
                "head_ns": statistics.median(v[clock] for v in sides[1]),
            }
        results.append(result)
        pathlib.Path(out).write_text(json.dumps(results, indent=2))
        print(
            name,
            *(
                f"{k}: {result[k]['median']:.5f} [{result[k]['p25']:.5f},{result[k]['p75']:.5f}]"
                for k in ("cpu", "wall")
            ),
            flush=True,
        )
finally:
    for p in workers:
        p.stdin.close()
    for p in workers:
        if p.wait() != 0:
            raise AssertionError()
