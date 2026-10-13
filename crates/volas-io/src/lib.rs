//! File readers for volas. Currently a high-performance CSV reader built on the
//! `csv` crate, with per-column type inference.

use std::collections::HashSet;
use std::fmt::Write;
use std::path::Path;

use volas_core::{datetime, Column, DataFrame, Result, StrBuffer, StrBufferBuilder, Tz, Validity, VolasError};
use volas_core::numeric::i64_to_f64_exact;

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
pub fn read_csv(path: impl AsRef<Path>, opts: &ReadCsvOptions) -> Result<DataFrame> {
    read_csv_table(path, opts)?.into_frame(&[])
}

/// How a column declared as dates is read from its text tokens.
#[derive(Debug, Clone, PartialEq)]
pub enum DateParse {
    /// Datetime text; a naive token is a wall clock in the zone, an offset-aware
    /// token is absolute.
    Text(Tz),
    /// A decimal epoch number in the unit (`"s"` / `"ms"` / `"us"` / `"ns"`), read
    /// exactly from the token.
    Epoch(String),
}

/// A parsed CSV file before typing: its (deduplicated) header and every column's
/// raw text tokens. [`CsvTable::into_frame`] types it — declared date columns
/// from their own tokens, every other column by inference.
pub struct CsvTable {
    headers: Vec<String>,
    cells: Vec<StrBuffer>,
    na: HashSet<String>,
}

impl CsvTable {
    /// The column names, in file order.
    pub fn names(&self) -> &[String] {
        &self.headers
    }

    /// Type the table into a frame: each `(name, parse)` in `dates` is a datetime
    /// column parsed from its tokens (an NA or blank token is `NaT`) BEFORE any
    /// inference could reinterpret it (an all-missing date column as float64, an
    /// epoch token through float64); every other column is inferred. An unknown
    /// date column name is a `ColumnNotFound` error.
    pub fn into_frame(self, dates: &[(String, DateParse)]) -> Result<DataFrame> {
        for (name, _) in dates {
            if !self.headers.contains(name) {
                return Err(VolasError::ColumnNotFound(name.clone()));
            }
        }
        let na = self.na;
        let columns = self
            .headers
            .iter()
            .zip(self.cells)
            .map(|(name, cells)| match dates.iter().find(|(n, _)| n == name) {
                Some((_, parse)) => date_column(&cells, parse, &na),
                None => Ok(infer_column(cells, &na)),
            })
            .collect::<Result<Vec<Column>>>()?;
        DataFrame::new(self.headers, columns, None)
    }
}

/// A declared date column from its raw tokens (see [`CsvTable::into_frame`]).
fn date_column(cells: &StrBuffer, parse: &DateParse, na: &HashSet<String>) -> Result<Column> {
    cells
        .iter()
        .map(|cell| {
            let t = cell.trim();
            if t.is_empty() || na.contains(t) {
                return Ok(datetime::NAT);
            }
            match parse {
                DateParse::Text(tz) => datetime::parse_ns_in_tz(t, tz),
                DateParse::Epoch(unit) => datetime::epoch_text_to_ns(t, unit),
            }
        })
        .collect::<Result<Vec<i64>>>()
        .map(Column::datetime)
}

/// Read a CSV file into a [`CsvTable`] (header + raw tokens), to be typed by
/// [`CsvTable::into_frame`].
pub fn read_csv_table(path: impl AsRef<Path>, opts: &ReadCsvOptions) -> Result<CsvTable> {
    let path = path.as_ref();
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
            .enumerate()
            // An empty header field becomes `Unnamed: {i}` (pandas parity) — e.g.
            // the empty index column `DataFrame.to_csv` writes for an unnamed index.
            .map(|(i, h)| {
                if h.is_empty() {
                    format!("Unnamed: {i}")
                } else {
                    h.to_string()
                }
            })
            .collect()
    } else {
        Vec::new()
    };
    // F46: duplicate header names are disambiguated pandas-style (a, a.1, a.2) —
    // erroring would leave the user no way to load the file; duplicate columns
    // would breach the unique-name contract.
    {
        let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut used: HashSet<String> = HashSet::new();
        for h in headers.iter_mut() {
            if used.contains(h.as_str()) {
                let base = h.clone();
                let mut k = *counts.get(&base).unwrap_or(&0);
                loop {
                    k += 1;
                    let cand = format!("{base}.{k}");
                    if !used.contains(&cand) {
                        *h = cand;
                        break;
                    }
                }
                counts.insert(base, k);
            }
            used.insert(h.clone());
        }
    }
    let mut ncol = headers.len();
    let mut raw: Vec<StrBufferBuilder> = (0..ncol)
        .map(|_| StrBufferBuilder::with_capacity(0)).collect();
    let mut rec = csv::StringRecord::new();
    while rdr.read_record(&mut rec)
        .map_err(|e| VolasError::Value(format!("bad CSV record: {e}")))?
    {
        if raw.is_empty() {
            ncol = rec.len();
            raw = (0..ncol).map(|_| StrBufferBuilder::with_capacity(0)).collect();
        }
        for (j, cells) in raw.iter_mut().enumerate() {
            cells.push(rec.get(j).unwrap_or(""));
        }
    }

    if !opts.has_header {
        headers = (0..ncol).map(|i| i.to_string()).collect();
    }
    Ok(CsvTable { headers, cells: raw.into_iter().map(StrBufferBuilder::finish).collect(), na: na_set })
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

/// Parsed values for the still-possible column dtype. Only an integer prefix
/// needs promotion; each token is parsed once for its chosen numeric candidate.
/// Raw tokens stay in the compact string buffer for dates and late text fallback.
enum Inferred {
    Pending,
    Int(Vec<i64>),
    Float(Vec<f64>),
    Bool(Vec<bool>),
    Text,
}

impl Inferred {
    fn missing(&mut self) {
        match self {
            Self::Int(v) => v.push(0),
            Self::Float(v) => v.push(f64::NAN),
            Self::Bool(v) => v.push(false),
            Self::Pending | Self::Text => {}
        }
    }
}

/// Promote parsed integers without reparsing their tokens. An integer the float
/// cannot hold instead selects text; a negative zero token keeps its float sign.
fn float_prefix(ints: &[i64], cells: &StrBuffer, valid: &Validity) -> Option<Vec<f64>> {
    let mut out = Vec::with_capacity(cells.len());
    for (i, &x) in ints.iter().enumerate() {
        out.push(if !valid.is_valid(i) {
            f64::NAN
        } else if x == 0 && cells.get(i).trim().starts_with('-') {
            -0.0
        } else {
            i64_to_f64_exact(x)?
        });
    }
    Some(out)
}

/// Infer i64 -> f64 -> bool -> str, preserving the original untrimmed strings
/// on fallback. The final numeric vector is built as the tokens are inspected;
/// no validation pass followed by a second parse of every token.
fn infer_column(cells: StrBuffer, na: &HashSet<String>) -> Column {
    let n = cells.len();
    if n == 0 {
        return Column::Str(cells, Validity::dense());
    }
    let mut inferred = Inferred::Pending;
    let mut valid = Validity::dense();
    let mut exact = String::new();
    for (i, cell) in cells.iter().enumerate() {
        let t = cell.trim();
        if na.contains(t) {
            valid.set(i, n, false);
            inferred.missing();
            continue;
        }
        let boolean = || {
            if t.eq_ignore_ascii_case("true") { Some(true) }
            else if t.eq_ignore_ascii_case("false") { Some(false) }
            else { None }
        };
        inferred = match inferred {
            Inferred::Pending => {
                if let Ok(x) = t.parse::<i64>() {
                    let mut v = Vec::with_capacity(n);
                    v.resize(i, 0);
                    v.push(x);
                    Inferred::Int(v)
                } else if let Some(x) = float_token(t, &mut exact) {
                    let mut v = Vec::with_capacity(n);
                    v.resize(i, f64::NAN);
                    v.push(x);
                    Inferred::Float(v)
                } else if let Some(b) = boolean() {
                    let mut v = Vec::with_capacity(n);
                    v.resize(i, false);
                    v.push(b);
                    Inferred::Bool(v)
                } else {
                    Inferred::Text
                }
            }
            Inferred::Int(mut v) => {
                if let Ok(x) = t.parse::<i64>() {
                    v.push(x);
                    Inferred::Int(v)
                } else if let Some(x) = float_token(t, &mut exact) {
                    match float_prefix(&v, &cells, &valid) {
                        Some(mut floats) => { floats.push(x); Inferred::Float(floats) }
                        None => Inferred::Text,
                    }
                } else {
                    Inferred::Text
                }
            }
            Inferred::Float(mut v) => match float_token(t, &mut exact) {
                Some(x) => { v.push(x); Inferred::Float(v) }
                None => Inferred::Text,
            },
            Inferred::Bool(mut v) => match boolean() {
                Some(x) => { v.push(x); Inferred::Bool(v) }
                None => Inferred::Text,
            },
            Inferred::Text => Inferred::Text,
        };
    }
    match inferred {
        Inferred::Int(v) => Column::i64_with(v, valid).expect("CSV validity covers every row"),
        Inferred::Pending => Column::f64(vec![f64::NAN; n]),
        Inferred::Float(v) => Column::f64(v),
        Inferred::Bool(v) => Column::bool_with(v, valid).expect("CSV validity covers every row"),
        Inferred::Text => {
            // Keep the original text, but normalize missing payloads to the same
            // empty spans as the nullable string constructor.
            let cells = if valid.has_nulls() {
                cells.iter().enumerate().map(|(i, s)| if valid.is_valid(i) { s } else { "" }).collect()
            } else {
                cells
            };
            Column::Str(cells, valid)
        }
    }
}

/// A present cell as the float64 it denotes, or `None` when it is not a number
/// float64 holds faithfully: not float-parseable, a `nan` spelling (only an NA
/// token is missing), or an integer literal float64 cannot hold exactly
/// (`9007199254740993`, `9223372036854775809`). An integer literal is exact when
/// the float's exact decimal expansion equals it.
fn float_token(t: &str, exact: &mut String) -> Option<f64> {
    let x = t.parse::<f64>().ok().filter(|x| !x.is_nan())?;
    let digits = t.strip_prefix(['+', '-']).unwrap_or(t);
    if !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Some(x); // a decimal / exponent / inf literal: float precision applies
    }
    let magnitude = digits.trim_start_matches('0');
    exact.clear();
    write!(exact, "{:.0}", x.abs()).expect("writing to a String cannot fail");
    (exact == if magnitude.is_empty() { "0" } else { magnitude }).then_some(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn default_na() -> HashSet<String> {
        build_na_set(&ReadCsvOptions::default())
    }

    fn infer(cells: &[&str]) -> Column {
        infer_column(cells.iter().copied().collect(), &default_na())
    }

    #[test]
    fn late_type_changes_keep_tokens_missing_and_signed_zero() {
        let c = infer(&["NA", "-000", " +007 ", "1.25", "-0", "null"]);
        let v = c.as_f64().unwrap();
        assert!(v[0].is_nan() && v[5].is_nan());
        assert_eq!((v[1].to_bits(), v[4].to_bits()), ((-0f64).to_bits(), (-0f64).to_bits()));
        assert_eq!((v[2], v[3]), (7., 1.25));
        for tokens in [
            vec![" 003 ", "-0", "NA", "1.25", " late text "],
            vec![" 9007199254740993 ", "NA", "1.25"],
            vec!["1.25", "NA", "9007199254740993"],
            vec!["NA", "TRUE", "false", " late text "],
            vec!["NA", "1", "true"],
        ] {
            let c = infer(&tokens);
            let Column::Str(text, val) = c else { panic!("expected text") };
            for (i, token) in tokens.iter().enumerate() {
                if *token == "NA" {
                    assert!(!val.is_valid(i));
                    assert_eq!(text.get(i), "");
                } else {
                    assert!(val.is_valid(i));
                    assert_eq!(text.get(i), *token);
                }
            }
        }
        let b = infer(&["NA", "false", "NA", "TRUE"]);
        assert_eq!(b.as_bool().unwrap(), &[false, false, false, true]);
        assert!(!b.is_valid(0) && !b.is_valid(2));
        assert_eq!(infer(&["-0", "+000"]).as_i64().unwrap(), &[0, 0]);
        let c = infer(&["NA", "-9223372036854775808", "1.25"]);
        assert_eq!(c.as_f64().unwrap()[1], i64::MIN as f64);
    }

    #[test]
    fn infers_i64() {
        assert_eq!(infer(&["1", "2", "3"]), Column::i64(vec![1, 2, 3]));
    }

    #[test]
    fn blank_or_na_keeps_int_with_validity() {
        // F35: an integral column with NA tokens stays int64 + validity (the
        // native-NA model, matching the constructor) — no legacy float demotion.
        for cells in [&["1", "", "3"][..], &["1", "NA", "null"][..]] {
            match infer(cells) {
                Column::I64(v, val) => {
                    assert_eq!(v[0], 1);
                    assert!(!val.is_valid(1));
                }
                other => panic!("expected I64, got {other:?}"), // LCOV_EXCL_LINE
            }
        }
    }

    #[test]
    fn str_column_na_tokens_are_missing() {
        // a str column marks NA tokens (blank / NA / null) as missing via validity,
        // keeping present cells — so a parse_dates column with blanks -> NaT.
        let c = infer(&["x", "", "z", "NA"]);
        assert!(matches!(c, Column::Str(..)));
        assert!(c.is_valid(0) && !c.is_valid(1) && c.is_valid(2) && !c.is_valid(3));
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
            Column::F64(v) => assert_eq!(v.to_vec(), vec![-1.5, 1000.0]),
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
    fn bool_inference_honors_na_tokens() {
        let col = infer(&["true", "", "FALSE"]);
        assert_eq!(col.dtype(), volas_core::DType::Bool);
        assert!(col.is_valid(0) && !col.is_valid(1) && col.as_bool().unwrap() == [true, false, false]);
        let na: HashSet<String> = ["True".to_string()].into();
        let col = infer_column(["True", "False"].into_iter().collect(), &na);
        assert!(!col.is_valid(0) && !col.as_bool().unwrap()[1]);
    }

    #[test]
    fn integers_float64_cannot_hold_stay_text() {
        // past int64, and past 2^53 next to a float: never rounded
        assert!(matches!(infer(&["9223372036854775809"]), Column::Str(..)));
        assert!(matches!(infer(&["9007199254740993", "1.5"]), Column::Str(..)));
        // exact integer literals (any spelling) next to a float are float64
        assert!(infer(&["9007199254740992", "-007", "+0", "1.5", "18446744073709551616"])
            .equals(&Column::f64(vec![9007199254740992.0, -7.0, 0.0, 1.5, 18446744073709551616.0])));
        // a `nan` spelling that is not an NA token is text
        let na: HashSet<String> = HashSet::new();
        assert!(matches!(infer_column(["1", "nan"].into_iter().collect(), &na), Column::Str(..)));
        assert_eq!(float_token("-", &mut String::new()), None);
        assert_eq!(float_token("1e3", &mut String::new()), Some(1000.0));
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
            infer_column(StrBuffer::from_vec(Vec::new()), &default_na()),
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
        match infer_column(["1", "NA"].into_iter().collect(), &na) {
            Column::Str(v, _) => assert_eq!(v.to_vec(), vec!["1".to_string(), "NA".to_string()]),
            other => panic!("expected Str, got {other:?}"), // LCOV_EXCL_LINE
        }
    }

    #[test]
    fn custom_na_value_upcasts() {
        let na = build_na_set(&ReadCsvOptions {
            na_values: vec!["MISSING".into()],
            ..Default::default()
        });
        match infer_column(["1", "MISSING"].into_iter().collect(), &na) {
            Column::I64(v, val) => {
                assert_eq!(v[0], 1);
                assert!(!val.is_valid(1));          // F35: custom NA token -> int64 + NA
            }
            other => panic!("expected I64, got {other:?}"), // LCOV_EXCL_LINE
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
        let c = df.column("c").unwrap();
        assert_eq!((c.str_at(0).unwrap(), c.str_at(1).unwrap()), ("x", "y"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn declared_date_columns_parse_from_their_tokens() {
        let path = std::env::temp_dir().join("volas_io_declared_dates.csv");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(f, "t,e,x").unwrap();
            writeln!(f, "NA,1704067200000000001,1").unwrap();
            writeln!(f, ",1704067200000000500.0,2").unwrap();
        }
        let table = read_csv_table(&path, &ReadCsvOptions::default()).unwrap();
        assert_eq!(table.names(), &["t", "e", "x"]);
        let dates = [
            ("t".to_string(), DateParse::Text(Tz::Naive)),
            ("e".to_string(), DateParse::Epoch("ns".into())),
        ];
        let df = table.into_frame(&dates).unwrap();
        // an all-missing date column is NaT, epoch tokens are exact
        assert_eq!(df.column("t").unwrap().as_datetime().unwrap(), &[datetime::NAT; 2]);
        assert_eq!(
            df.column("e").unwrap().as_datetime().unwrap(),
            &[1_704_067_200_000_000_001, 1_704_067_200_000_000_500]
        );
        let missing = [("nope".to_string(), DateParse::Text(Tz::Utc))];
        let err = read_csv_table(&path, &ReadCsvOptions::default()).unwrap().into_frame(&missing);
        assert!(matches!(err, Err(VolasError::ColumnNotFound(_))));
        let text = [("x".to_string(), DateParse::Text(Tz::Utc))];
        assert!(read_csv_table(&path, &ReadCsvOptions::default()).unwrap().into_frame(&text).is_err());
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
