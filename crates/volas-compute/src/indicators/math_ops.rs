// ---------------------------------------------------------------------------
// Math operators — rolling reductions over a fixed window
// ---------------------------------------------------------------------------

/// Rolling sum over `period` (TA-Lib SUM), O(n) via a sliding total (TA-Lib's exact
/// order: add current, emit, drop trailing). Lookback `period-1`.
pub fn sum(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || period > n {
        return out;
    }
    let mut total = 0.0;
    for &x in &data[..period - 1] {
        total += x;
    }
    let mut trailing = 0;
    for i in (period - 1)..n {
        total += data[i];
        out[i] = total;
        total -= data[trailing];
        trailing += 1;
    }
    out
}

/// Absolute index (as a float) of the highest value in each trailing `period`-bar
/// window (TA-Lib MAXINDEX). Tie-breaking mirrors TA-Lib's incremental tracker: the
/// earliest index when the running max is rebuilt after the old one leaves (strict
/// `>`), the latest when a fresh bar matches the current max (`>=`). Lookback `period-1`.
pub fn maxindex(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || period > n {
        return out;
    }
    let mut hi_idx = 0usize;
    let mut hi = f64::NEG_INFINITY;
    let mut first = true;
    let mut trailing = 0usize;
    for today in (period - 1)..n {
        let tmp = data[today];
        if first || hi_idx < trailing {
            // The tracked max left the window — rebuild over [trailing, today].
            hi_idx = trailing;
            hi = data[trailing];
            for i in (trailing + 1)..=today {
                if data[i] > hi {
                    hi_idx = i;
                    hi = data[i];
                }
            }
            first = false;
        } else if tmp >= hi {
            hi_idx = today;
            hi = tmp;
        }
        out[today] = hi_idx as f64;
        trailing += 1;
    }
    out
}

/// Absolute index (as a float) of the lowest value in each trailing `period`-bar
/// window (TA-Lib MININDEX). Mirror of [`maxindex`] (strict `<` on rebuild, `<=` on
/// a fresh matching bar). Lookback `period-1`.
pub fn minindex(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || period > n {
        return out;
    }
    let mut lo_idx = 0usize;
    let mut lo = f64::INFINITY;
    let mut first = true;
    let mut trailing = 0usize;
    for today in (period - 1)..n {
        let tmp = data[today];
        if first || lo_idx < trailing {
            lo_idx = trailing;
            lo = data[trailing];
            for i in (trailing + 1)..=today {
                if data[i] < lo {
                    lo_idx = i;
                    lo = data[i];
                }
            }
            first = false;
        } else if tmp <= lo {
            lo_idx = today;
            lo = tmp;
        }
        out[today] = lo_idx as f64;
        trailing += 1;
    }
    out
}
