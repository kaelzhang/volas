"""Append-refresh microbenchmark for state-carry.

For each directive and frame size N: build an N-row frame, cache the indicator
(materialize + store its recursive state), then repeatedly {append ONE bar; read
df[directive] -> incremental refresh}. Report the min-of-many per-bar refresh in
microseconds.

A state-carry-converted recursive indicator refreshes in O(new rows) == O(1), so
its per-bar time is FLAT as N grows. An indicator left on the correct O(n)
full-recompute fallback would grow ~linearly with N. A finite-memory indicator
(ma) was always O(window) and is the flat control.

Every recursive / absolute-position family is now state-carry-converted (including
the index family `maxindex` and the composite `stochrsi`), so all of them are
expected FLAT; nothing remains on the O(n) fallback.
"""

import time

import numpy as np

import volas

COLS = ['open', 'high', 'low', 'close', 'volume']

# Converted families (all expected FLAT): EMA / Wilder / cumulative / SAR / HT, the
# windowed-but-absolute index family, and the composite StochRSI.
CONVERTED = ['ema:12', 'macd', 'rsi:14', 'atr:14', 'adx:14', 'obv', 'sar',
             'dema:30', 'kama:30', 'ht_trendline', 'maxindex:30', 'stochrsi.k']
CONTROL_FINITE = ['ma:20']            # finite-memory: always O(window), flat
DIRECTIVES = CONVERTED + CONTROL_FINITE

NS = [2_000, 20_000, 100_000]
REPEAT = 200          # append+read iterations; frame grows by this over the run
WARM = 20             # warm iterations to absorb capacity growth / first-touch


def _rng_frame(n, seed=0):
    rs = np.random.default_rng(seed)
    base = np.cumsum(rs.standard_normal(n)) + 1000.0
    high = base + np.abs(rs.standard_normal(n))
    low = base - np.abs(rs.standard_normal(n))
    close = base + rs.standard_normal(n) * 0.3
    vol = np.abs(rs.standard_normal(n)) * 1e6 + 1.0
    return {'open': base.copy(), 'high': high, 'low': low,
            'close': close, 'volume': vol}


def _one_bar(prev_close):
    # a single plausible next bar
    c = prev_close + 0.1
    return volas.DataFrame({'open': [prev_close], 'high': [c + 1.0],
                            'low': [c - 1.0], 'close': [c], 'volume': [1.0e6]})


def per_bar_us(directive, n):
    """Time the pure incremental refresh of ONE appended bar.

    Uses append + fulfill() (the refresh path) and reads only the latest cell via
    iloc[-1] — this is what a live tick actually does. Crucially it avoids the
    O(n) full-column to_numpy() copy, so the number reflects the refresh COMPUTE,
    not array materialization.
    """
    data = _rng_frame(n)
    df = volas.DataFrame(data)
    # cache: materialize + populate recursive state
    _ = df[directive].to_numpy()
    last = float(data['close'][-1])

    # warm: absorb first-touch / Vec capacity growth
    for _ in range(WARM):
        df.append(_one_bar(last))
        df.fulfill()
        last += 0.1

    best = float('inf')
    sink = 0.0
    for _ in range(REPEAT):
        bar = _one_bar(last)
        last += 0.1
        t0 = time.perf_counter()
        df.append(bar)
        df.fulfill()                       # performs the incremental refresh
        v = df[directive].iloc[-1]         # read latest cell only (no full copy)
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
        sink += 0.0 if v != v else v       # touch result; ignore NaN
    return best * 1e6


def main():
    print(f"append-refresh per-bar microseconds (min of {REPEAT}, warm {WARM})")
    header = f"{'directive':<16}" + "".join(f"{('N=' + str(n)):>14}" for n in NS) + f"{'ratio Nmax/Nmin':>18}"
    print(header)
    print("-" * len(header))

    def emit(group, names):
        print(f"# {group}")
        for d in names:
            row = []
            for n in NS:
                row.append(per_bar_us(d, n))
            ratio = row[-1] / row[0] if row[0] > 0 else float('nan')
            cells = "".join(f"{us:>14.3f}" for us in row)
            print(f"{d:<16}{cells}{ratio:>18.2f}")

    emit("CONVERTED (state-carry -> expect FLAT, ratio ~1)", CONVERTED)
    emit("CONTROL finite-memory (always flat)", CONTROL_FINITE)


if __name__ == '__main__':
    main()
