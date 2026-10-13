//! CSV token storage grows by column capacity, never by one allocation per cell.
#[path = "support/allocation.rs"]
mod allocation;
#[test]
fn csv_allocations_do_not_scale_with_cells() {
    let path = std::env::temp_dir().join(format!("volas_csv_cost_{}.csv", std::process::id()));
    let data = "1.25,2,hello\n".repeat(30_000);
    std::fs::write(&path, format!("a,b,c\n{data}")).unwrap();
    let (df, calls, _) =
        allocation::measure(|| volas_io::read_csv(&path, &Default::default()).unwrap());
    std::fs::remove_file(path).unwrap();
    assert_eq!((df.height(), df.width()), (30_000, 3));
    assert!(calls < 1000, "{calls} allocations for 90k cells");
}
