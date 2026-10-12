//! Decimal arithmetic on binary numbers — the one owner of every "scale by a power of ten"
//! conversion: rounding a value to `k` decimal places, and reading a scaled decimal
//! (`mantissa · 10^-scale`, an Arrow `Decimal128` / `Decimal256`) as a float.
//!
//! Every result is the correctly rounded one: rounding is half-to-even on the float's
//! exact value at the `k`-th decimal place (Python's `round`), and a decimal becomes the
//! float nearest to it. Scaling by a float `10^k` cannot give that — `10^k` is inexact
//! past 22, `x·10^k` rounds before the integer step and overflows for large `x` (NumPy's
//! `round(1e300, 10)` is `inf`) — so the exact value is kept: an error-free product while
//! the scale is exact, integer arithmetic for whole numbers, and Rust's exact decimal
//! formatting and correctly rounded parsing beyond. A result past the float range is
//! `None`, an overflow for the caller to raise.

/// A float type decimal rounding produces (`f64` / `f32`), computed through `f64`, which
/// holds every `f32` exactly.
pub trait RoundFloat: Copy + std::str::FromStr {
    /// Significand bits, the implicit one included.
    const DIGITS: u32;
    /// The largest `k` for which `10^k` is exact in this type.
    const EXACT_POW10: i32;
    fn to_f64(self) -> f64;
    /// The value nearest to `x` (`x` already rounded to `f64`: callers pass only values
    /// for which that double rounding is provably exact).
    fn from_f64(x: f64) -> Self;
    /// The value nearest to the integer `n`.
    fn from_i128(n: i128) -> Self;
    fn is_finite(self) -> bool;
}

impl RoundFloat for f64 {
    const DIGITS: u32 = f64::MANTISSA_DIGITS;
    const EXACT_POW10: i32 = 22;
    fn to_f64(self) -> f64 {
        self
    }
    fn from_f64(x: f64) -> f64 {
        x
    }
    fn from_i128(n: i128) -> f64 {
        n as f64
    }
    fn is_finite(self) -> bool {
        self.is_finite()
    }
}

impl RoundFloat for f32 {
    const DIGITS: u32 = f32::MANTISSA_DIGITS;
    const EXACT_POW10: i32 = 10;
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_f64(x: f64) -> f32 {
        x as f32
    }
    fn from_i128(n: i128) -> f32 {
        n as f32
    }
    fn is_finite(self) -> bool {
        self.is_finite()
    }
}

/// Past this many decimal places every float is already whole in them: the smallest
/// `f64` step, `2^-1074 ≈ 4.9e-324`, exceeds `10^-324`.
const MAX_PLACES: i32 = 350;

/// `x` rounded half-to-even at the `k`-th decimal place (`k < 0`: to a multiple of
/// `10^-k`), the float nearest to that decimal; NaN / ±inf / ±0 pass through. `None`
/// when the result lies past the float range (`round(1.7e308, -308)` is `2e308`).
pub fn round<F: RoundFloat>(x: F, k: i32) -> Option<F> {
    let v = x.to_f64();
    if !x.is_finite() || v == 0.0 {
        return Some(x);
    }
    if k >= 0 { round_places(x, k) } else { round_tens(x, k.unsigned_abs()) }
}

/// [`round`] at `k >= 0` places.
fn round_places<F: RoundFloat>(x: F, k: i32) -> Option<F> {
    let v = x.to_f64();
    if k > MAX_PLACES || step_exceeds_place(v, F::DIGITS, k) {
        // the rounded decimal lies within half a place, under half a step: x itself
        return Some(x);
    }
    if k > 22 {
        return Some(parse_decimal(&format!("{v:.*}", k as usize))); // exact digits, half-even
    }
    let scale = POW10[k as usize];
    // the exact scaled value v·10^k = p + e (an error-free product; when `e` underflows
    // |p| < 2^-969, far under a half)
    let p = v * scale;
    let e = v.mul_add(scale, -p);
    let n = p.round_ties_even();
    let d = p - n; // exact: n is the integer nearest p
    // n is the answer unless p sits on a half the exact value lies past. Otherwise
    // |d + e| < 1/2 (d is a multiple of p's step, |e| at most half of it), or p is whole
    // and the exact value a half from it — then p, rounded from that half to even, is
    // the even neighbour already.
    let whole = if d.abs() == 0.5 && e != 0.0 && e.signum() == d.signum() { n + d.signum() } else { n };
    // |whole| <= 2^DIGITS (the step test bounds |v|·10^k): `whole / 10^k` is one
    // correctly rounded division while 10^k is exact in F, and for f32 the f64 quotient
    // rounds to f32 correctly (53 >= 2·24 + 2)
    Some(if k <= F::EXACT_POW10 {
        F::from_f64(whole / scale)
    } else {
        parse_decimal(&format!("{whole:.0}e-{k}"))
    })
}

/// Whether the step from `v` toward zero (for a `digits`-bit significand) is at least
/// `10^-k`: then the rounded decimal, within half a place of `v`, is nearer
/// `v` than any other float (or `v` is whole and `k == 0`). Exact: `2^q > 10^-k` ⇔
/// `5^k > 2^(-q-k)`, compared through `k·log2 5`, which for `1 <= k <= 350` is never
/// within 1e-4 of an integer (`log2_5_multiples_stay_off_integers`).
fn step_exceeds_place(v: f64, digits: u32, k: i32) -> bool {
    // the step toward zero: at a power of two the floats below are twice as dense
    let power_of_two = v.to_bits() & ((1 << 52) - 1) == 0;
    let q = (exponent(v) - digits as i32 + 1 - i32::from(power_of_two)).max(min_step_exponent(digits));
    let m = -q - k;
    m <= 0 || k as f64 * LOG2_5 > m as f64
}

/// `log2 5`.
const LOG2_5: f64 = std::f64::consts::LOG2_10 - 1.0;

/// The binary exponent of a finite non-zero `v` (`2^e <= |v| < 2^(e+1)`).
fn exponent(v: f64) -> i32 {
    let bits = v.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    if biased == 0 {
        // subnormal: the highest set bit of the fraction
        -1074 + (63 - (bits & ((1 << 52) - 1)).leading_zeros() as i32)
    } else {
        biased - 1023
    }
}

/// The exponent of the smallest step of a `digits`-bit float type (`f64`: −1074).
fn min_step_exponent(digits: u32) -> i32 {
    if digits == f64::MANTISSA_DIGITS { -1074 } else { -149 }
}

/// [`round`] at `-j` places: to the nearest multiple of `10^j`, computed exactly — on
/// the whole part in `u64` / `u128` (the fraction, exact too, only decides a half), from
/// the exact decimal digits of a whole number past `2^126`.
fn round_tens<F: RoundFloat>(x: F, j: u32) -> Option<F> {
    let v = x.to_f64();
    let zero = Some(F::from_f64(0.0f64.copysign(v)));
    // |v| < 0.5 rounds to 0 at any j >= 1, and so does every |v| < 1.8e308 at j >= 309
    if v.abs() < 0.5 || j >= 309 {
        return zero;
    }
    let finish = |n: u128| {
        let out = F::from_i128(n as i128 * if v < 0.0 { -1 } else { 1 });
        if n == 0 { zero } else { out.is_finite().then_some(out) }
    };
    let (whole, frac) = (v.abs().trunc(), v.abs().fract()); // both exact
    // the multiple is at most whole + 10^j: under 2^64 and 2^127 respectively
    if whole < 2f64.powi(63) && j <= 18 {
        return finish(nearest_multiple_u64(whole as u64, frac > 0.0, 10u64.pow(j)) as u128);
    }
    if whole < 2f64.powi(126) && j <= 37 {
        return finish(nearest_multiple_u128(whole as u128, frac > 0.0, 10u128.pow(j)));
    }
    // a whole number past 2^126, or a multiple of 10^38 and beyond: its exact digits,
    // rounded half-even at the j-th from the right
    let digits = format!("{:.0}", v.abs());
    let j = j as usize;
    if digits.len() < j {
        return zero; // |v| < 10^(j-1), under half of 10^j
    }
    let (head, tail) = digits.split_at(digits.len() - j);
    let mut head = head.as_bytes().to_vec();
    let first = tail.as_bytes()[0];
    let rest_zero = tail.bytes().skip(1).all(|b| b == b'0');
    let odd = head.last().is_some_and(|&b| (b - b'0') % 2 == 1);
    if first > b'5' || (first == b'5' && (!rest_zero || odd)) {
        increment(&mut head);
    }
    if head.iter().all(|&b| b == b'0') {
        return zero;
    }
    let sign = if v < 0.0 { "-" } else { "" };
    let out: F = parse_decimal(&format!("{sign}{}e{j}", String::from_utf8(head).unwrap()));
    out.is_finite().then_some(out)
}

macro_rules! nearest_multiple {
    ($name:ident, $t:ty) => {
        /// The multiple of `t` (even) nearest to `whole` plus a fraction in `[0, 1)` that
        /// is non-zero when `past`: a remainder of exactly half goes up past it, and to
        /// the even multiple without one.
        fn $name(whole: $t, past: bool, t: $t) -> $t {
            let (q, r) = (whole / t, whole % t);
            let half = t / 2;
            let up = r > half || (r == half && (past || q % 2 == 1));
            (q + <$t>::from(up)) * t
        }
    };
}
nearest_multiple!(nearest_multiple_u64, u64);
nearest_multiple!(nearest_multiple_u128, u128);

/// Add one to a decimal digit string, carrying (`"99"` → `"100"`).
fn increment(digits: &mut Vec<u8>) {
    for b in digits.iter_mut().rev() {
        if *b == b'9' {
            *b = b'0';
        } else {
            *b += 1;
            return;
        }
    }
    digits.insert(0, b'1');
}

/// The float nearest to a decimal text Rust's formatting produced.
fn parse_decimal<F: RoundFloat>(text: &str) -> F {
    text.parse().ok().expect("formatted decimals parse")
}

/// Exact powers of ten `10^0 ..= 10^22`, every one a `f64`.
const POW10: [f64; 23] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
    1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
];

/// The float nearest to the scaled decimal `mantissa · 10^-scale` (an Arrow
/// `Decimal128`); `None` past the `f64` range.
pub fn scaled_to_f64(mantissa: i128, scale: i32) -> Option<f64> {
    // one correctly rounded operation on exact operands, while both are exact
    if mantissa.unsigned_abs() <= 1 << 53 && scale.unsigned_abs() <= 22 {
        let (m, p) = (mantissa as f64, POW10[scale.unsigned_abs() as usize]);
        return Some(if scale >= 0 { m / p } else { m * p });
    }
    scaled_text_to_f64(&mantissa.to_string(), scale)
}

/// [`scaled_to_f64`] of a mantissa given as its decimal digits (a `Decimal256`, wider
/// than `i128`).
pub fn scaled_text_to_f64(mantissa: &str, scale: i32) -> Option<f64> {
    let out: f64 = parse_decimal(&format!("{mantissa}e{}", -(scale as i64)));
    out.is_finite().then_some(out)
}

/// Round an int to `decimals` places: identity for `decimals >= 0`; for negative
/// `decimals`, banker's round to the nearest `10^|decimals|` multiple (integer
/// arithmetic, exact for all i64 — no f64 round-trip). `None` when that multiple
/// lies past `i64` (`round(i64::MAX, -1)`, or `round(6e18, -19)` = 1e19), an
/// overflow (contract #25).
///
/// The rounding runs on the magnitude in `u128` ([`nearest_multiple_u128`]; banker's
/// rounding is symmetric), where `10^k` and the rounded multiple are exact for every `k`
/// up to 38 — past `i64` from `k = 19` on, so the result's fit is checked rather than
/// assumed. A `10^k` beyond `u128` (`k >= 39`) exceeds twice any `i64`, so every value
/// rounds to 0 there.
pub(crate) fn round_i64(x: i64, decimals: i32) -> Option<i64> {
    if decimals >= 0 {
        return Some(x);
    }
    let Some(factor) = 10u128.checked_pow(decimals.unsigned_abs()) else {
        return Some(0);
    };
    // |x| + 10^38 < 2^127: exact
    let m = nearest_multiple_u128(u128::from(x.unsigned_abs()), false, factor) as i128;
    i64::try_from(if x < 0 { -m } else { m }).ok()
}

#[cfg(test)]
mod tests;
