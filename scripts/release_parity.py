#!/usr/bin/env python3
"""Export deterministic release-wheel results for bit-exact cross-build checks."""

import argparse
from pathlib import Path

from build_pair import manifest
from pgo_train import load_site


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--compare", type=Path)
    args = ap.parse_args()
    meta = manifest(args.site)
    load_site(args.site.resolve(), meta["native_sha256"])
    import numpy as np
    import pyarrow as pa
    from volas import DataFrame
    from test.bench_workloads import generated_ohlcv, append, stream, COLUMNS
    from test.bench_tf_live import build, TF_LIVE_PHASES, TF_LIVE_INDICATORS
    import json

    data = generated_ohlcv(257, 20261021)
    results = {}
    for d in json.loads(
        (Path(__file__).resolve().parents[1] / "test/pgo_inventory.json").read_text()
    ).values():
        results[f"batch/{d}"] = DataFrame(data).exec(d, dtype="float64")
        run, setup = append(data, d)
        (df,), kw = setup()
        run(df, **kw)
        results[f"append/{d}"] = df[d].to_numpy(dtype="float64")
        df["close"] = data["close"] * 1.001
        results[f"mutation/{d}"] = df[d].to_numpy(dtype="float64")
    for phase in TF_LIVE_PHASES:
        for d in TF_LIVE_INDICATORS:
            run, setup = build(data, COLUMNS, d, phase)
            (df,), kw = setup()
            run(df, **kw)
            results[f"{phase}/{d}"] = df[d].to_numpy(dtype="float64")
    run, setup = stream(
        {k: v for k, v in data.items() if k in COLUMNS}, ["ma:7", "ema:37"], 47
    )
    (df,), kw = setup()
    run(df, **kw)
    for d in ("ma:7", "ema:37"):
        results[f"window/{d}"] = df[d].to_numpy(dtype="float64")
    array = pa.chunked_array([pa.array([1.0, None, 3.0]), pa.array([4.0, 5.0])]).slice(
        1, 3
    )
    frame = DataFrame.from_arrow(pa.table({"close": array}))
    results["arrow/null-chunked-sliced"] = frame["close"].to_numpy()
    results["dlpack"] = np.from_dlpack(DataFrame({"close": data["close"]})["close"])
    np.savez(args.output, **results)
    if args.compare:
        with np.load(args.compare) as reference:
            if set(reference.files) != set(results):
                raise AssertionError()
            for name, values in results.items():
                expected = reference[name]
                np.testing.assert_array_equal(
                    np.isnan(values), np.isnan(expected), err_msg=name
                )
                finite = ~np.isnan(expected)
                np.testing.assert_array_equal(
                    values[finite].view("uint64"),
                    expected[finite].view("uint64"),
                    err_msg=name,
                )
    print(f"{len(results)} numeric/mutation/live/window/interop arrays verified")


if __name__ == "__main__":
    main()
