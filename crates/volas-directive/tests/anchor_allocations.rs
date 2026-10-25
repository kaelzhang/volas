use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use volas_core::{Column, DataFrame};
use volas_directive::parse;

struct Counter;
static ENABLED: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
#[global_allocator]
static GLOBAL: Counter = Counter;

fn count() {
    if ENABLED.load(Ordering::Relaxed) {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
    }
}

// Forward every allocation unchanged to System; the counter itself never allocates.
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        System.alloc(layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count();
        System.alloc_zeroed(layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count();
        System.realloc(ptr, layout, size)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

// Foreign producer storage intentionally never memoizes a NaN fact. A producing
// pass already owns the complete-row classification used by its snapshot encoder.
#[test]
fn same_pass_foreign_snapshots_do_not_allocate_per_anchor() {
    use std::{ptr::NonNull, sync::Arc};
    use volas_core::Buffer;
    use volas_directive::exec::execute_with_states_and_anchors;
    for mode in ["clean", "leading", "interior"] {
        let mut values: Vec<_> = (0..65_536).map(|i| 100.0 + (i as f64 * 0.01).sin()).collect();
        if mode == "leading" { values[..1027].fill(f64::NAN); }
        if mode == "interior" { for i in (0..values.len()).step_by(97) { values[i] = f64::NAN; } }
        let data = Arc::new(values);
        let ptr = NonNull::new(data.as_ptr() as *mut f64).unwrap();
        // SAFETY: data owns the immutable allocation and the buffer retains its Arc.
        let foreign = unsafe { Buffer::from_foreign(ptr, data.len(), data.clone()) };
        let frames = [Column::f64(data.as_ref().clone()), Column::from_f64_buffer(foreign)].map(|col| {
            DataFrame::new(vec!["close".into()], vec![col], None).unwrap()
        });
        for name in ["ema:20", "ma:20", "crsi", "close>ma:20"] {
            let node = parse(name).unwrap();
            let counts = frames.each_ref().map(|df| {
                ALLOCS.store(0, Ordering::Relaxed);
                ENABLED.store(true, Ordering::Relaxed);
                let result = execute_with_states_and_anchors(df, &node).unwrap();
                ENABLED.store(false, Ordering::Relaxed);
                let count = ALLOCS.load(Ordering::Relaxed);
                assert!(result.3.len() > 100);
                let expected = if std::ptr::eq(df, &frames[0]) { Some(mode != "clean") } else { None };
                assert_eq!(df.column("close").unwrap().known_has_nan(), expected);
                count
            });
            println!("{mode} {name}: owned={} foreign={}", counts[0], counts[1]);
            assert!(counts[1] <= counts[0] + 12, "{mode} {name}: owned={} foreign={}", counts[0], counts[1]);
        }
    }
}
