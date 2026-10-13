//! Thread-local allocation accounting: input construction stays outside the measured call.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static TOTAL: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
}
struct Counting;
fn record(bytes: usize) {
    if ACTIVE.try_with(Cell::get).unwrap_or(false) {
        TOTAL.with(|v| {
            let (calls, size) = v.get();
            v.set((calls + 1, size + bytes));
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
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record(size);
        System.realloc(ptr, layout, size)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;

pub fn measure<T>(f: impl FnOnce() -> T) -> (T, usize, usize) {
    TOTAL.with(|v| v.set((0, 0)));
    ACTIVE.with(|v| v.set(true));
    let out = f();
    ACTIVE.with(|v| v.set(false));
    let (calls, bytes) = TOTAL.with(Cell::get);
    (out, calls, bytes)
}
