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
