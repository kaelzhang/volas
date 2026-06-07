//! File readers for volas. Currently a high-performance CSV reader built on the
//! `csv` crate, with per-column type inference.

use std::collections::HashSet;

use volas_core::{Column, DataFrame, Result, VolasError};

/// pandas's default missing-value tokens (`STR_NA_VALUES`). A numeric column
/// with any of these (trimmed) entries upcasts to `f64` with `NaN`, exactly as
/// `pandas.read_csv` does.
pub const DEFAULT_NA_VALUES: &[&str] = &[
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN",
    "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null",
];

/// Options controlling [`read_csv`].
pub struct ReadCsvOptions {
    /// Field delimiter (default `b','`).
    pub delimiter: u8,
    /// Whether the first row is a header. When `false`, columns are named
    /// `"0".."n-1"` by position.
    pub has_header: bool,
    /// Additional strings to treat as missing (beyond [`DEFAULT_NA_VALUES`]).
    pub na_values: Vec<String>,
    /// Whether [`DEFAULT_NA_VALUES`] are recognized at all.
    pub keep_default_na: bool,
}

impl Default for ReadCsvOptions {
    fn default() -> Self {
        ReadCsvOptions {
            delimiter: b',',
            has_header: true,
            na_values: Vec::new(),
            keep_default_na: true,
        }
    }
}

/// Read a CSV file into a [`DataFrame`], inferring each column's dtype
/// (`i64` -> `f64` -> `bool` -> `str`).
pub fn read_csv(path: &str, opts: &ReadCsvOptions) -> Result<DataFrame> {
    let na_set = build_na_set(opts);
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(opts.delimiter)
        .has_headers(opts.has_header)
        .from_path(path)
        .map_err(|e| VolasError::Value(format!("cannot open CSV {path:?}: {e}")))?;

    // Resolve the header / column count up front so a data-less file still
    // yields the right number of (empty) columns.
    let mut headers: Vec<String> = if opts.has_header {
        rdr.headers()
            .map_err(|e| VolasError::Value(format!("bad CSV header: {e}")))?
            .iter()
            .map(String::from)
            .collect()
    } else {
        Vec::new()
    };
    let mut ncol = headers.len();
    let mut raw: Vec<Vec<String>> = if ncol > 0 {
        vec![Vec::new(); ncol]
    } else {
        Vec::new()
    };

    for rec in rdr.records() {
        let rec = rec.map_err(|e| VolasError::Value(format!("bad CSV record: {e}")))?;
        if raw.is_empty() {
            ncol = rec.len();
            raw = vec![Vec::new(); ncol];
        }
        for (j, cell) in raw.iter_mut().enumerate() {
            cell.push(rec.get(j).unwrap_or("").to_string());
        }
    }

    if !opts.has_header {
        headers = (0..ncol).map(|i| i.to_string()).collect();
    }

    let columns: Vec<Column> = raw.into_iter().map(|c| infer_column(c, &na_set)).collect();
    DataFrame::new(headers, columns, None)
}

/// The effective set of missing-value tokens for these options.
fn build_na_set(opts: &ReadCsvOptions) -> HashSet<String> {
    let mut set = HashSet::new();
    if opts.keep_default_na {
        set.extend(DEFAULT_NA_VALUES.iter().map(|s| s.to_string()));
    }
    set.extend(opts.na_values.iter().cloned());
    set
}

/// Infer the most specific column type from string cells
/// (`i64` -> `f64` -> `bool` -> `str`), treating NA tokens as missing.
fn infer_column(cells: Vec<String>, na: &HashSet<String>) -> Column {
    if cells.is_empty() {
        return Column::str(cells);
    }
    let trimmed: Vec<&str> = cells.iter().map(|c| c.trim()).collect();
    let is_na = |t: &str| na.contains(t);
    // i64: every cell present and integral.
    if trimmed
        .iter()
        .all(|t| !is_na(t) && t.parse::<i64>().is_ok())
    {
        return Column::i64(trimmed.iter().map(|t| t.parse().unwrap()).collect());
    }
    // f64: every cell is either missing (-> NaN) or float-parseable.
    if trimmed.iter().all(|t| is_na(t) || t.parse::<f64>().is_ok()) {
        return Column::f64(
            trimmed
                .iter()
                .map(|t| {
                    if is_na(t) {
                        f64::NAN
                    } else {
                        t.parse().unwrap()
                    }
                })
                .collect(),
        );
    }
    // bool: every cell present and a true/false literal (any case).
    if trimmed
        .iter()
        .all(|t| matches!(t.to_ascii_lowercase().as_str(), "true" | "false"))
    {
        return Column::bool(
            trimmed
                .iter()
                .map(|t| t.eq_ignore_ascii_case("true"))
                .collect(),
        );
    }
    Column::str(cells)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn default_na() -> HashSet<String> {
        build_na_set(&ReadCsvOptions::default())
    }

    fn infer(cells: &[&str]) -> Column {
        infer_column(cells.iter().map(|s| s.to_string()).collect(), &default_na())
    }

    #[test]
    fn infers_i64() {
        assert_eq!(infer(&["1", "2", "3"]), Column::i64(vec![1, 2, 3]));
    }

    #[test]
    fn blank_or_na_upcasts_int_to_f64_nan() {
        for cells in [&["1", "", "3"][..], &["1", "NA", "null"][..]] {
            match infer(cells) {
                Column::F64(v) => {
                    assert_eq!(v[0], 1.0);
                    assert!(v[1].is_nan());
                }
                other => panic!("expected F64, got {other:?}"), // LCOV_EXCL_LINE
            }
        }
    }

    #[test]
    fn all_na_column_is_f64_all_nan() {
        match infer(&["NA", "null", ""]) {
            Column::F64(v) => assert!(v.iter().all(|x| x.is_nan())),
            other => panic!("expected F64, got {other:?}"), // LCOV_EXCL_LINE
        }
    }

    #[test]
    fn scientific_and_negative_are_f64() {
        match infer(&["-1.5", "1e3"]) {
            Column::F64(v) => assert_eq!(**v, vec![-1.5, 1000.0]),
            other => panic!("expected F64, got {other:?}"), // LCOV_EXCL_LINE
        }
    }

    #[test]
    fn infers_bool_any_case() {
        assert_eq!(
            infer(&["True", "false", "TRUE"]),
            Column::bool(vec![true, false, true])
        );
    }

    #[test]
    fn non_numeric_is_str() {
        assert_eq!(
            infer(&["a", "b"]),
            Column::str(vec!["a".into(), "b".into()])
        );
    }

    #[test]
    fn empty_column_is_str() {
        assert_eq!(
            infer_column(Vec::new(), &default_na()),
            Column::str(Vec::new())
        );
    }

    #[test]
    fn na_disabled_keeps_tokens_as_str() {
        let na = build_na_set(&ReadCsvOptions {
            keep_default_na: false,
            ..Default::default()
        });
        // "NA" is no longer missing -> the column is object/string, not float.
        match infer_column(vec!["1".into(), "NA".into()], &na) {
            Column::Str(v) => assert_eq!(**v, vec!["1".to_string(), "NA".to_string()]),
            other => panic!("expected Str, got {other:?}"), // LCOV_EXCL_LINE
        }
    }

    #[test]
    fn custom_na_value_upcasts() {
        let na = build_na_set(&ReadCsvOptions {
            na_values: vec!["MISSING".into()],
            ..Default::default()
        });
        match infer_column(vec!["1".into(), "MISSING".into()], &na) {
            Column::F64(v) => {
                assert_eq!(v[0], 1.0);
                assert!(v[1].is_nan());
            }
            other => panic!("expected F64, got {other:?}"), // LCOV_EXCL_LINE
        }
    }

    #[test]
    fn reads_csv_file_with_mixed_dtypes() {
        let path = std::env::temp_dir().join("volas_io_read_csv_mixed.csv");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(f, "a,b,c").unwrap();
            writeln!(f, "1,1.5,x").unwrap();
            writeln!(f, "2,,y").unwrap();
        }
        let df = read_csv(path.to_str().unwrap(), &ReadCsvOptions::default()).unwrap();
        assert_eq!((df.height(), df.width()), (2, 3));
        assert_eq!(df.column("a").unwrap().as_i64().unwrap(), &[1, 2]);
        let b = df.column("b").unwrap().as_f64().unwrap();
        assert_eq!(b[0], 1.5);
        assert!(b[1].is_nan());
        assert_eq!(
            df.column("c").unwrap().as_str().unwrap(),
            &["x".to_string(), "y".to_string()]
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn reads_headerless_and_tab_delimited() {
        let path = std::env::temp_dir().join("volas_io_read_csv_headerless.tsv");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(f, "1\t2\t3").unwrap();
            writeln!(f, "4\t5\t6").unwrap();
        }
        let opts = ReadCsvOptions {
            delimiter: b'\t',
            has_header: false,
            ..Default::default()
        };
        let df = read_csv(path.to_str().unwrap(), &opts).unwrap();
        assert_eq!((df.height(), df.width()), (2, 3));
        assert_eq!(
            df.names(),
            &["0".to_string(), "1".to_string(), "2".to_string()]
        );
        assert_eq!(df.column("0").unwrap().as_i64().unwrap(), &[1, 4]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn missing_path_errors() {
        assert!(read_csv(
            "/no/such/dir/volas_io_missing.csv",
            &ReadCsvOptions::default()
        )
        .is_err());
    }
}
