//! File readers for volas. Currently a high-performance CSV reader built on the
//! `csv` crate, with per-column type inference.

use volas_core::{Column, DataFrame, Result, VolasError};

/// Read a CSV file into a [`DataFrame`], inferring each column's dtype
/// (`i64` -> `f64` -> `bool` -> `str`).
pub fn read_csv(path: &str) -> Result<DataFrame> {
    let mut rdr = csv::ReaderBuilder::new()
        .from_path(path)
        .map_err(|e| VolasError::Value(format!("cannot open CSV {path:?}: {e}")))?;
    let headers: Vec<String> = rdr
        .headers()
        .map_err(|e| VolasError::Value(format!("bad CSV header: {e}")))?
        .iter()
        .map(String::from)
        .collect();
    let ncol = headers.len();
    let mut raw: Vec<Vec<String>> = vec![Vec::new(); ncol];
    for rec in rdr.records() {
        let rec = rec.map_err(|e| VolasError::Value(format!("bad CSV record: {e}")))?;
        for (j, cell) in raw.iter_mut().enumerate() {
            cell.push(rec.get(j).unwrap_or("").to_string());
        }
    }
    let columns: Vec<Column> = raw.into_iter().map(infer_column).collect();
    DataFrame::new(headers, columns, None)
}

/// Whether a (trimmed) cell is one of pandas's default missing-value tokens.
///
/// Mirrors pandas's `STR_NA_VALUES` so a numeric column with `NA` / `null` /
/// `N/A` entries upcasts to `f64` with `NaN`, exactly as `pandas.read_csv` does.
fn is_na(s: &str) -> bool {
    matches!(
        s,
        "" | "#N/A"
            | "#N/A N/A"
            | "#NA"
            | "-1.#IND"
            | "-1.#QNAN"
            | "-NaN"
            | "-nan"
            | "1.#IND"
            | "1.#QNAN"
            | "<NA>"
            | "N/A"
            | "NA"
            | "NULL"
            | "NaN"
            | "None"
            | "n/a"
            | "nan"
            | "null"
    )
}

/// Infer the most specific column type from string cells
/// (`i64` -> `f64` -> `bool` -> `str`), treating NA tokens as missing.
fn infer_column(cells: Vec<String>) -> Column {
    if cells.is_empty() {
        return Column::Str(cells);
    }
    let trimmed: Vec<&str> = cells.iter().map(|c| c.trim()).collect();
    // i64: every cell present and integral.
    if trimmed.iter().all(|t| !is_na(t) && t.parse::<i64>().is_ok()) {
        return Column::I64(trimmed.iter().map(|t| t.parse().unwrap()).collect());
    }
    // f64: every cell is either missing (-> NaN) or float-parseable.
    if trimmed.iter().all(|t| is_na(t) || t.parse::<f64>().is_ok()) {
        return Column::F64(
            trimmed
                .iter()
                .map(|t| if is_na(t) { f64::NAN } else { t.parse().unwrap() })
                .collect(),
        );
    }
    // bool: every cell present and a true/false literal (any case).
    if trimmed
        .iter()
        .all(|t| matches!(t.to_ascii_lowercase().as_str(), "true" | "false"))
    {
        return Column::Bool(trimmed.iter().map(|t| t.eq_ignore_ascii_case("true")).collect());
    }
    Column::Str(cells)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn infer(cells: &[&str]) -> Column {
        infer_column(cells.iter().map(|s| s.to_string()).collect())
    }

    #[test]
    fn infers_i64() {
        assert_eq!(infer(&["1", "2", "3"]), Column::I64(vec![1, 2, 3]));
    }

    #[test]
    fn blank_or_na_upcasts_int_to_f64_nan() {
        for cells in [&["1", "", "3"][..], &["1", "NA", "null"][..]] {
            match infer(cells) {
                Column::F64(v) => {
                    assert_eq!(v[0], 1.0);
                    assert!(v[1].is_nan());
                }
                other => panic!("expected F64, got {other:?}"),
            }
        }
    }

    #[test]
    fn all_na_column_is_f64_all_nan() {
        match infer(&["NA", "null", ""]) {
            Column::F64(v) => assert!(v.iter().all(|x| x.is_nan())),
            other => panic!("expected F64, got {other:?}"),
        }
    }

    #[test]
    fn scientific_and_negative_are_f64() {
        match infer(&["-1.5", "1e3"]) {
            Column::F64(v) => assert_eq!(v, vec![-1.5, 1000.0]),
            other => panic!("expected F64, got {other:?}"),
        }
    }

    #[test]
    fn infers_bool_any_case() {
        assert_eq!(
            infer(&["True", "false", "TRUE"]),
            Column::Bool(vec![true, false, true])
        );
    }

    #[test]
    fn non_numeric_is_str() {
        assert_eq!(
            infer(&["a", "b"]),
            Column::Str(vec!["a".into(), "b".into()])
        );
    }

    #[test]
    fn empty_column_is_str() {
        assert_eq!(infer_column(Vec::new()), Column::Str(Vec::new()));
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
        let df = read_csv(path.to_str().unwrap()).unwrap();
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
    fn missing_path_errors() {
        assert!(read_csv("/no/such/dir/volas_io_missing.csv").is_err());
    }
}
