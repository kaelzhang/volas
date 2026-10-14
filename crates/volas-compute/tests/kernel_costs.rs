//! Allocation contracts for compute kernels. Counts are local to the test thread.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::time::Instant;
use volas_compute::{raw::indicators as ind, window};

type CostCase<'a> = (&'a str, Box<dyn Fn() -> Vec<f64> + 'a>);

struct Counting;
thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static COUNTS: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
}
fn record(bytes: usize) {
    if ACTIVE.try_with(Cell::get).unwrap_or(false) {
        let _ = COUNTS.try_with(|c| {
            let (n, b) = c.get();
            c.set((n + 1, b + bytes));
        });
    }
}
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        System.alloc(layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        System.alloc_zeroed(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record(size);
        System.realloc(ptr, layout, size)
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;
fn counted<T>(f: impl FnOnce() -> T) -> (T, usize, usize) {
    COUNTS.with(|c| c.set((0, 0)));
    ACTIVE.with(|c| c.set(true));
    let out = f();
    ACTIVE.with(|c| c.set(false));
    let (n, b) = COUNTS.with(Cell::get);
    (out, n, b)
}
fn prices(n: usize) -> Vec<f64> {
    (0..n).map(|i| 100.0 + (i as f64 * 0.1).sin()).collect()
}
fn checksum(x: &[f64]) -> u64 {
    x.iter().fold(0xcbf29ce484222325, |h, x| {
        (h ^ x.to_bits()).wrapping_mul(0x100000001b3)
    })
}
#[test]
fn low_variance_has_no_per_window_heap_allocation() {
    for n in [100, 10_000] {
        for data in [
            vec![12345.67; n],
            (0..n)
                .map(|i| 12345.67 + (i as f64 * 0.1).sin() * 1e-6)
                .collect(),
        ] {
            let (_, count, _) = counted(|| window::var(&data, 20, 20, 1).unwrap());
            assert_eq!(count, 1, "only the result may allocate, n={n}");
        }
    }
}
#[test]
fn directional_capture_allocates_only_the_terminal_state() {
    for n in [100, 10_000] {
        let c = prices(n);
        let h: Vec<_> = c.iter().map(|x| x + 1.0).collect();
        let l: Vec<_> = c.iter().map(|x| x - 1.0).collect();
        for capture in [ind::adx_final_state, ind::dx_final_state] {
            let (s, count, _) = counted(|| capture(&h, &l, &c, 14));
            assert!(s.is_some());
            assert_eq!(count, 1, "n={n}");
        }
    }
}
#[test]
fn single_output_kernels_allocate_one_column() {
    let c = prices(10_000);
    for (name, kernel) in [
        ("wma", ind::wma as fn(&[f64], usize, usize) -> Vec<f64>),
        ("boll", ind::boll),
        ("bbw", ind::bbw),
        ("linearreg", ind::linearreg),
        ("slope", ind::linearreg_slope),
        ("intercept", ind::linearreg_intercept),
        ("angle", ind::linearreg_angle),
        ("tsf", ind::tsf),
    ] {
        let (_, count, bytes) = counted(|| kernel(&c, 20, 0));
        assert_eq!(count, 1, "{name}: {bytes} bytes");
    }
    let (_, count, _) = counted(|| ind::macd(&c, 12, 26));
    assert_eq!(count, 1, "MACD line");
    let (_, count, _) = counted(|| ind::macd_signal(&c, 12, 26, 9));
    assert_eq!(count, 1, "MACD signal");
    let (_, count, _) = counted(|| ind::macd_histogram(&c, 12, 26, 9));
    assert_eq!(count, 2, "MACD histogram retains both subtraction operands");
    let (_, count, _) = counted(|| ind::boll_upper(&c, 20, 2.0, 0));
    assert_eq!(count, 1, "upper band");
    let (_, count, _) = counted(|| ind::boll_lower(&c, 20, 2.0, 0));
    assert_eq!(count, 1, "lower band");
}
#[test]
fn finite_moments_borrow_the_windows() {
    let c = prices(10_000);
    for kernel in [window::skew, window::kurt] {
        let (_, count, bytes) = counted(|| kernel(&c, 2000, 2000));
        assert!(
            count == 1 && bytes == 8 * c.len(),
            "{count} allocations, {bytes} bytes"
        );
    }
}
/// Run explicitly in release mode to record allocation counts and min-of-seven
/// latency on prepared inputs; timing assertions would depend on host load.
#[test]
#[ignore]
fn measure_kernel_costs() {
    for n in [10_000, 100_000, 1_000_000] {
        let c = prices(n);
        let h: Vec<_> = c.iter().map(|x| x + 1.0).collect();
        let l: Vec<_> = c.iter().map(|x| x - 1.0).collect();
        let flat = vec![12345.67; n];
        let low: Vec<_> = (0..n)
            .map(|i| 12345.67 + (i as f64 * 0.1).sin() * 1e-6)
            .collect();
        let mut cases: Vec<CostCase<'_>> = vec![
            ("var_flat", Box::new(|| window::var(&flat, 20, 20, 1).unwrap())),
            ("var_low", Box::new(|| window::var(&low, 20, 20, 1).unwrap())),
            (
                "adx_capture",
                Box::new(|| ind::adx_final_state(&h, &l, &c, 14).unwrap()),
            ),
            (
                "dx_capture",
                Box::new(|| ind::dx_final_state(&h, &l, &c, 14).unwrap()),
            ),
            ("wma", Box::new(|| ind::wma(&c, 20, 0))),
            ("macd", Box::new(|| ind::macd(&c, 12, 26))),
            ("boll_upper", Box::new(|| ind::boll_upper(&c, 20, 2.0, 0))),
            ("linearreg", Box::new(|| ind::linearreg(&c, 20, 0))),
            ("slope", Box::new(|| ind::linearreg_slope(&c, 20, 0))),
        ];
        if n <= 100_000 {
            for p in [20, 200, 2000] {
                let name = match p {
                    20 => "skew20",
                    200 => "skew200",
                    _ => "skew2000",
                };
                let data = &c;
                cases.push((name, Box::new(move || window::skew(data, p, p).unwrap())));
            }
        }
        for (name, f) in cases {
            let (out, allocs, bytes) = counted(&f);
            let hash = checksum(&out);
            drop(out);
            let mut times = Vec::new();
            for _ in 0..7 {
                let t = Instant::now();
                drop(black_box(f()));
                times.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            times.sort_by(f64::total_cmp);
            println!(
                "COST,{name},{n},{:.6},{allocs},{bytes},{hash:016x}",
                times[0]
            );
        }
    }
}

/// An exact-bit fingerprint matrix for paired builds, including NaN payloads,
/// signed zeros, non-finite windows, origins and terminal recurrence states.
#[test]
#[ignore]
fn snapshot_kernel_bits() {
    for fixture in 0..12 {
        let mut c = prices(137);
        match fixture {
            1 => c.fill(12345.67),
            2 => {
                for x in &mut c {
                    *x = 12345.67 + (*x - 100.0) * 1e-6;
                }
            }
            3 => {
                for (i, x) in c.iter_mut().enumerate() {
                    *x = if i % 2 == 0 { -0.0 } else { 0.0 };
                }
            }
            4 => {
                c[0] = f64::NAN;
                c[7] = f64::from_bits(0x7ff8000000000028);
                c[81] = f64::NAN;
            }
            5 => {
                c[7] = f64::INFINITY;
                c[81] = f64::NEG_INFINITY;
            }
            6 => {
                c[7] = 1e300;
                c[81] = -1e300;
            }
            7 => {
                for (i, x) in c.iter_mut().enumerate() {
                    *x = (i as f64 - 68.0) * 1e-310;
                }
            }
            8 => {
                c.fill(f64::NAN);
            }
            9 => {
                c[0..135].fill(f64::NAN);
            }
            10 => {
                for (i, x) in c.iter_mut().enumerate() {
                    *x = (i as f64 * 0.33).sin() * 2.0f64.powi((i as i32 * 53 % 1800) - 900);
                }
            }
            11 => {
                c.fill(1e308);
                c[1] = -1e308;
            }
            _ => {}
        }
        let h: Vec<_> = c.iter().map(|x| x + 1.0).collect();
        let l: Vec<_> = c.iter().map(|x| x - 1.0).collect();
        for p in [1, 2, 3, 4, 20, 200, usize::MAX] {
            for min in [0, 1, 3, 20] {
                if min > p {
                    assert_eq!(window::skew(&c, p, min), Err(window::WindowError::MinPeriods));
                    assert_eq!(window::kurt(&c, p, min), Err(window::WindowError::MinPeriods));
                    assert_eq!(window::var(&c, p, min, 1), Err(window::WindowError::MinPeriods));
                    continue;
                }
                for (name, kernel) in [
                    ("skew", window::skew as fn(&[f64], usize, usize) -> Result<Vec<f64>, window::WindowError>),
                    ("kurt", window::kurt),
                ] {
                    println!(
                        "BITS,{fixture},{p},{min},{name},{:016x}",
                        checksum(&kernel(&c, p, min).unwrap())
                    );
                }
                println!(
                    "BITS,{fixture},{p},{min},var,{:016x}",
                    checksum(&window::var(&c, p, min, 1).unwrap())
                );
            }
        }
        for p in [0, 1, 2, 3, 20, 200] {
            for origin in [0, 3, 511] {
                for (name, kernel) in [
                    ("wma", ind::wma as fn(&[f64], usize, usize) -> Vec<f64>),
                    ("boll", ind::boll),
                    ("bbw", ind::bbw),
                    ("linearreg", ind::linearreg),
                    ("slope", ind::linearreg_slope),
                    ("intercept", ind::linearreg_intercept),
                    ("angle", ind::linearreg_angle),
                    ("tsf", ind::tsf),
                ] {
                    println!(
                        "BITS,{fixture},{p},{origin},{name},{:016x}",
                        checksum(&kernel(&c, p, origin))
                    );
                }
                for (name, kernel) in [
                    (
                        "upper",
                        ind::boll_upper as fn(&[f64], usize, f64, usize) -> Vec<f64>,
                    ),
                    ("lower", ind::boll_lower),
                    ("hv", ind::hv),
                ] {
                    println!(
                        "BITS,{fixture},{p},{origin},{name},{:016x}",
                        checksum(&kernel(&c, p, 2.0, origin))
                    );
                }
            }
        }
        for (name, kernel) in [
            (
                "adx",
                ind::adx_final_state as fn(&[f64], &[f64], &[f64], usize) -> Option<Vec<f64>>,
            ),
            ("dx", ind::dx_final_state),
        ] {
            println!(
                "BITS,{fixture},14,0,{name},{:016x}",
                checksum(&kernel(&h, &l, &c, 14).unwrap_or_default())
            );
        }
        for p in [1, 2, 20, 200] {
            for (name, out) in [
                ("bbi", ind::bbi(&c, p, 3, 5, 20, 0)),
                ("kdj_k", ind::kdj_k(&h, &l, &c, p, 3, 50.0)),
                ("kdj_d", ind::kdj_d(&h, &l, &c, p, 3, 3, 50.0)),
                ("kdj_j", ind::kdj_j(&h, &l, &c, p, 3, 3, 50.0)),
                ("donchian", ind::donchian(&h, &l, p)),
                ("midpoint", ind::midpoint(&c, p)),
                ("midprice", ind::midprice(&h, &l, p)),
                ("psy", ind::psy(&c, p, 0)),
                ("mass_index", ind::mass_index(&h, &l, p, 0)),
                ("chop", ind::chop(&h, &l, &c, p, 0)),
                ("vr", ind::vr(&c, &h, p, 0)),
                ("wvad", ind::wvad(&l, &h, &l, &c, &h, p, 0)),
            ] {
                println!("BITS,{fixture},{p},0,{name},{:016x}", checksum(&out));
            }
        }
        for fast in [3, 12, 26] {
            println!(
                "BITS,{fixture},{fast},0,macd,{:016x}",
                checksum(&ind::macd(&c, fast, 12))
            );
            println!(
                "BITS,{fixture},{fast},0,signal,{:016x}",
                checksum(&ind::macd_signal(&c, fast, 12, 9))
            );
            println!(
                "BITS,{fixture},{fast},0,histogram,{:016x}",
                checksum(&ind::macd_histogram(&c, fast, 12, 9))
            );
        }
    }
}
