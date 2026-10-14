# volas-compute

Numeric kernels accept `&[f64]` and return owned `Vec<f64>` buffers. Callers borrow
contiguous inputs directly and transfer output ownership without copying:

```rust
use volas_compute::kernels;

let close = [1.0, 2.0, 3.0, 4.0];
let mean: Vec<f64> = kernels::sma(&close, 2, 0);
assert_eq!(&mean[1..], &[1.5, 2.5, 3.5]);
```

Array adapters belong in the interop layer: gather a non-contiguous array there
before calling a kernel. Compute has no ndarray dependency or array-view adapters.
The Python interface is unchanged.

`rolling_mean_std` returns both columns when both are requested.
`rolling_dispersion_pass` projects the same mean/std arithmetic into one column,
with optional terminal-state publication through `StateSink`. Regression kernels
likewise emit only the selected line. Exact sums retain their original arithmetic
and rounding; their common expansions stay inline and larger expansions spill to
heap storage without a precision limit.

The published `window` helpers return `Result<_, window::WindowError>`. They
validate a positive rolling window (`usize::MAX` for expanding),
`min_periods <= window`, finite `0 < alpha <= 1`, `q` in `[0, 1]`, the quantile
interpolation and rank method, equal pairwise input lengths, and rank padding.
Validation applies even to empty inputs. Valid inputs retain the same arithmetic:

```rust
use volas_compute::window;
assert_eq!(window::ewm_mean(&[1.0, 3.0], 0.5, false, false, 0).unwrap(), vec![1.0, 2.0]);
assert_eq!(window::quantile(&[1.0, 3.0], 2, 1, f64::INFINITY, "linear"), Err(window::WindowError::Quantile));
```
