//! pandas-compatible text / HTML rendering for `Series` / `DataFrame` / `Row`,
//! plus the lower-level CSV cell formatting shared with `to_csv`.
//!
//! The text layout reproduces pandas's console formatting closely (per-column
//! float decimals, a leading sign-space on numeric / bool / string cells, a
//! left-justified index column, right-justified data columns joined by a single
//! space, head/tail truncation, and the `[N rows x M columns]` footer). It is
//! kept out of `lib.rs` so the binding layer stays a thin set of `#[pymethods]`
//! that delegate here.

use volas_core::{datetime, Column, DataFrame, Index, IndexKind, Series, Tz};

// ===========================================================================
// CSV cell formatting (shared with `DataFrame.to_csv`)
// ===========================================================================

/// Format the `i`-th cell of a column as a CSV field (`na_rep` for NaN).
pub(crate) fn cell_to_csv(
    col: &Column,
    i: usize,
    na_rep: &str,
    ff: Option<(Option<usize>, char)>,
) -> String {
    match col {
        Column::F64(v) => {
            if v[i].is_nan() {
                na_rep.to_string()
            } else {
                match ff {
                    Some((prec, kind)) => fmt_f64_with(prec, kind, v[i]),
                    // Default: the shortest round-trippable form that keeps the
                    // decimal point, so `1.0` writes as "1.0" (Rust's `to_string`
                    // drops it to "1").
                    None => format!("{:?}", v[i]),
                }
            }
        }
        Column::F32(v) => {
            if v[i].is_nan() {
                na_rep.to_string()
            } else {
                match ff {
                    Some((prec, kind)) => fmt_f64_with(prec, kind, v[i] as f64),
                    None => format!("{:?}", v[i]),
                }
            }
        }
        Column::Bool(v, _) => if v[i] { "True" } else { "False" }.to_string(),
        Column::I64(v, _) => v[i].to_string(),
        Column::I32(v, _) => v[i].to_string(),
        Column::Str(v) => v[i].clone(),
        Column::Datetime(v) => datetime::format_ns(v[i]),
    }
}

/// Format the `i`-th index label as a CSV field.
pub(crate) fn index_label_csv(index: &Index, i: usize) -> String {
    match index.kind() {
        IndexKind::Range(_) => i.to_string(),
        IndexKind::Int64(v) => v[i].to_string(),
        IndexKind::Datetime(v, tz) => datetime::format_ns_tz(v[i], *tz),
        IndexKind::Str(v) => v[i].clone(),
    }
}

/// Parse a printf-style float format `%[.prec](f|e|g)` (the common pandas
/// `float_format` forms) into `(precision, kind)`; `None` if unrecognized.
pub(crate) fn parse_float_format(fmt: &str) -> Option<(Option<usize>, char)> {
    let body = fmt.strip_prefix('%')?;
    let kind = body.chars().last()?;
    if !matches!(kind, 'f' | 'e' | 'g') {
        return None;
    }
    let prec = match body.strip_prefix('.') {
        Some(rest) => Some(rest[..rest.len() - 1].parse().ok()?),
        None if body.len() == 1 => None,
        None => return None,
    };
    Some((prec, kind))
}

/// Apply a parsed [`parse_float_format`] spec to `x`.
fn fmt_f64_with(prec: Option<usize>, kind: char, x: f64) -> String {
    match kind {
        'e' => match prec {
            Some(p) => format!("{x:.p$e}"),
            None => format!("{x:e}"),
        },
        'g' => match prec {
            Some(p) => format!("{x:.p$}"),
            None => format!("{x:?}"),
        },
        // 'f' — the only remaining kind `parse_float_format` admits.
        _ => match prec {
            Some(p) => format!("{x:.p$}"),
            None => format!("{x:.6}"),
        },
    }
}

// ===========================================================================
// pandas-style table rendering
// ===========================================================================

/// When to print the `[N rows x M columns]` dimensions footer.
#[derive(Clone, Copy)]
pub(crate) enum Dimensions {
    /// Never (a non-truncated `to_string` with `show_dimensions=False`).
    Never,
    /// Always (`to_string` with `show_dimensions=True`).
    Always,
    /// Only when the table is truncated (the repr default — pandas's
    /// `display.show_dimensions='truncate'`).
    OnTruncate,
}

/// Display options threaded through the table renderer.
pub(crate) struct DisplayOpts<'a> {
    /// Whether to print the column-name header row.
    pub header: bool,
    /// Whether to print the index column.
    pub index: bool,
    /// String used for missing (NaN) cells.
    pub na_rep: &'a str,
    /// Parsed `float_format` spec; `None` uses the per-column decimal format.
    pub float_format: Option<(Option<usize>, char)>,
    /// When to print the dimensions footer.
    pub dimensions: Dimensions,
    /// `Some(k)` → show the first `k` and last `k` rows with an ellipsis row in
    /// between; `None` → no truncation.
    pub truncate: Option<usize>,
}

/// Left-justify `s` to `w` columns (padding on the right).
fn ljust(s: &str, w: usize) -> String {
    let pad = w.saturating_sub(s.chars().count());
    format!("{s}{}", " ".repeat(pad))
}

/// Right-justify `s` to `w` columns (padding on the left).
fn rjust(s: &str, w: usize) -> String {
    let pad = w.saturating_sub(s.chars().count());
    format!("{}{s}", " ".repeat(pad))
}

/// Center `s` in `w` columns, matching Python's `str.center` (which biases the
/// extra space left or right via the `marg & width & 1` term). Used for the
/// Series truncation ellipsis.
fn py_center(s: &str, w: usize) -> String {
    let n = s.chars().count();
    if w <= n {
        return s.to_string();
    }
    let marg = w - n;
    let left = marg / 2 + (marg & w & 1);
    format!("{}{s}{}", " ".repeat(left), " ".repeat(marg - left))
}

/// The displayed row positions and whether truncation occurred. When truncated,
/// the result is the first `k` followed by the last `k` positions.
fn displayed_rows(n: usize, truncate: Option<usize>) -> (Vec<usize>, bool) {
    match truncate {
        // Callers only pass `Some(k)` with `1 <= k` and `2 * k < n`.
        Some(k) => {
            let mut rows: Vec<usize> = (0..k).collect();
            rows.extend((n - k)..n);
            (rows, true)
        }
        None => ((0..n).collect(), false),
    }
}

/// Whether a UTC-ns instant falls exactly on midnight in `tz` (so a datetime
/// index of pure dates renders date-only, like pandas).
fn is_midnight(ns: i64, tz: Tz) -> bool {
    let (_, _, _, h, mi, s) = datetime::civil_parts_tz(ns, tz);
    h == 0 && mi == 0 && s == 0 && ns.rem_euclid(1_000_000_000) == 0
}

/// Format a UTC-ns instant as a `YYYY-MM-DD` date in `tz`.
fn fmt_date(ns: i64, tz: Tz) -> String {
    let (y, mo, d, _, _, _) = datetime::civil_parts_tz(ns, tz);
    format!("{y:04}-{mo:02}-{d:02}")
}

/// Format the index labels at `rows` (a datetime index renders date-only when
/// every displayed instant is midnight, else the full timestamp).
fn index_cells(index: &Index, rows: &[usize]) -> Vec<String> {
    match index.kind() {
        IndexKind::Range(_) => rows.iter().map(|&i| i.to_string()).collect(),
        IndexKind::Int64(v) => rows.iter().map(|&i| v[i].to_string()).collect(),
        IndexKind::Str(v) => rows.iter().map(|&i| v[i].clone()).collect(),
        IndexKind::Datetime(v, tz) => {
            let all_midnight = rows.iter().all(|&i| is_midnight(v[i], *tz));
            rows.iter()
                .map(|&i| {
                    if all_midnight {
                        fmt_date(v[i], *tz)
                    } else {
                        datetime::format_ns_tz(v[i], *tz)
                    }
                })
                .collect()
        }
    }
}

/// The number of fractional digits to show for a float column: the max needed
/// (after rounding to 6 places and trimming trailing zeros) over the finite
/// values, clamped to `[1, 6]` — pandas's default column-uniform float format.
fn float_decimals(vals: &[f64]) -> usize {
    let mut d = 1usize;
    for &x in vals {
        if x.is_finite() {
            // "{:.6}" always emits exactly six fractional digits.
            let s = format!("{x:.6}");
            let frac = s[s.len() - 6..].trim_end_matches('0');
            d = d.max(frac.len().max(1));
        }
    }
    d.min(6)
}

/// Prepend a leading sign-space to a non-negative numeric string (pandas's
/// `leading_space`, reserving the slot a `-` would occupy).
fn lead_num(neg: bool, s: String) -> String {
    if neg {
        s
    } else {
        format!(" {s}")
    }
}

/// Format a data column's cells at `rows`, pandas-style: per-column float
/// decimals (or `float_format`), a leading sign-space on numeric / bool / string
/// cells, `na_rep` for NaN, and date-or-full datetimes. Datetime cells get no
/// leading space (matching pandas).
fn data_cells(
    col: &Column,
    rows: &[usize],
    na_rep: &str,
    ff: Option<(Option<usize>, char)>,
) -> Vec<String> {
    match col {
        Column::F64(v) => {
            let d = float_decimals(v);
            rows.iter()
                .map(|&i| {
                    let x = v[i];
                    if x.is_nan() {
                        na_rep.to_string()
                    } else {
                        let s = match ff {
                            Some((prec, kind)) => fmt_f64_with(prec, kind, x),
                            None => format!("{x:.d$}"),
                        };
                        lead_num(x < 0.0, s)
                    }
                })
                .collect()
        }
        Column::F32(v) => {
            let asf: Vec<f64> = v.iter().map(|&x| x as f64).collect();
            let d = float_decimals(&asf);
            rows.iter()
                .map(|&i| {
                    let x = v[i] as f64;
                    if x.is_nan() {
                        na_rep.to_string()
                    } else {
                        let s = match ff {
                            Some((prec, kind)) => fmt_f64_with(prec, kind, x),
                            None => format!("{x:.d$}"),
                        };
                        lead_num(x < 0.0, s)
                    }
                })
                .collect()
        }
        Column::I64(v, _) => rows
            .iter()
            .map(|&i| lead_num(v[i] < 0, v[i].to_string()))
            .collect(),
        Column::I32(v, _) => rows
            .iter()
            .map(|&i| lead_num(v[i] < 0, v[i].to_string()))
            .collect(),
        Column::Bool(v, _) => rows
            .iter()
            .map(|&i| format!(" {}", if v[i] { "True" } else { "False" }))
            .collect(),
        Column::Str(v) => rows.iter().map(|&i| format!(" {}", v[i])).collect(),
        Column::Datetime(v) => {
            let all_midnight = rows.iter().all(|&i| is_midnight(v[i], Tz::Utc));
            rows.iter()
                .map(|&i| {
                    if all_midnight {
                        fmt_date(v[i], Tz::Utc)
                    } else {
                        datetime::format_ns(v[i])
                    }
                })
                .collect()
        }
    }
}

/// The empty-frame placeholder (pandas's `Empty DataFrame\nColumns: [...]\n
/// Index: [...]`), used when the frame has no rows or no selected columns.
fn empty_frame(df: &DataFrame, col_pos: &[usize]) -> String {
    let names: Vec<&str> = col_pos.iter().map(|&j| df.names()[j].as_str()).collect();
    let index = if df.height() == 0 {
        "[]".to_string()
    } else {
        let labels: Vec<String> = (0..df.height())
            .map(|i| index_label_csv(df.index(), i))
            .collect();
        format!("[{}]", labels.join(", "))
    };
    format!(
        "Empty DataFrame\nColumns: [{}]\nIndex: {index}",
        names.join(", ")
    )
}

/// Render a frame as a pandas-style aligned text table. `col_pos` selects which
/// columns to show (in order).
pub(crate) fn render_frame(df: &DataFrame, col_pos: &[usize], opts: &DisplayOpts) -> String {
    let n = df.height();
    if n == 0 || col_pos.is_empty() {
        return empty_frame(df, col_pos);
    }
    let (rows, truncated) = displayed_rows(n, opts.truncate);

    // Index column (left-justified). Its header is blank; the index name, if
    // any, occupies a second header row.
    let mut idx = index_cells(df.index(), &rows);
    // Data columns. The header is leading-spaced exactly like the cells (so a
    // non-datetime column reserves the same sign slot), and a datetime column —
    // which has no leading space — keeps its bare name.
    let mut headers: Vec<String> = Vec::with_capacity(col_pos.len());
    let mut bodies: Vec<Vec<String>> = Vec::with_capacity(col_pos.len());
    for &j in col_pos {
        let col = &df.columns()[j];
        let name = &df.names()[j];
        headers.push(if matches!(col, Column::Datetime(_)) {
            name.clone()
        } else {
            format!(" {name}")
        });
        bodies.push(data_cells(col, &rows, opts.na_rep, opts.float_format));
    }
    // Per-column value width (cells only), measured before the ellipsis row is
    // inserted — the ellipsis never widens a column.
    let vws: Vec<usize> = bodies
        .iter()
        .map(|cells| cells.iter().map(|s| s.chars().count()).max().unwrap_or(0))
        .collect();
    if truncated {
        let k = rows.len() / 2;
        // pandas uses "..." for a wide column/index, ".." for a narrow one.
        let iw = idx.iter().map(|s| s.chars().count()).max().unwrap_or(0);
        idx.insert(k, if iw > 3 { "..." } else { ".." }.to_string());
        for (cells, &vw) in bodies.iter_mut().zip(&vws) {
            cells.insert(k, if vw > 3 { "..." } else { ".." }.to_string());
        }
    }

    // Final widths: a data column also covers its leading-spaced header (when a
    // header row is shown); the index column covers its labels (incl. the `..`
    // ellipsis) and its name.
    let widths: Vec<usize> = headers
        .iter()
        .zip(&vws)
        .map(|(h, &vw)| {
            if opts.header {
                vw.max(h.chars().count())
            } else {
                vw
            }
        })
        .collect();
    let iname = df.index().name().unwrap_or("");
    let idx_w = idx
        .iter()
        .map(|s| s.chars().count())
        .chain(std::iter::once(iname.chars().count()))
        .max()
        .unwrap_or(0);

    let mut lines: Vec<String> = Vec::new();
    if opts.header {
        let mut h = String::new();
        if opts.index {
            h.push_str(&" ".repeat(idx_w));
        }
        for (j, name) in headers.iter().enumerate() {
            if opts.index || j > 0 {
                h.push(' ');
            }
            h.push_str(&rjust(name, widths[j]));
        }
        lines.push(h);
        // The index-name row (only when the index is named and shown).
        if opts.index && !iname.is_empty() {
            let mut h2 = ljust(iname, idx_w);
            for &w in &widths {
                h2.push(' ');
                h2.push_str(&" ".repeat(w));
            }
            lines.push(h2);
        }
    }
    for r in 0..idx.len() {
        let mut line = String::new();
        if opts.index {
            line.push_str(&ljust(&idx[r], idx_w));
        }
        for (j, cells) in bodies.iter().enumerate() {
            if opts.index || j > 0 {
                line.push(' ');
            }
            line.push_str(&rjust(&cells[r], widths[j]));
        }
        lines.push(line);
    }

    let mut out = lines.join("\n");
    let show_footer = match opts.dimensions {
        Dimensions::Always => true,
        Dimensions::OnTruncate => truncated,
        Dimensions::Never => false,
    };
    if show_footer {
        out.push_str(&format!("\n\n[{} rows x {} columns]", n, df.width()));
    }
    out
}

// --- vertical (Series / Row) rendering -------------------------------------

/// Assemble the vertical body shared by `Series` and `Row`: each row is
/// `<label>   <value>` with the labels left-justified and the leading-spaced
/// values right-justified.
fn vertical_body(labels: &[String], values: &[String]) -> String {
    let lw = labels.iter().map(|s| s.chars().count()).max().unwrap_or(0);
    let vw = values.iter().map(|s| s.chars().count()).max().unwrap_or(0);
    labels
        .iter()
        .zip(values)
        .map(|(l, v)| format!("{}   {}", ljust(l, lw), rjust(v, vw)))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Render a `Series` as pandas does: a vertical `label   value` table, with a
/// `Name: <name>, [Length: <n>,] dtype: <dtype>` footer when `footer` is set
/// (repr / str) and omitted for `to_string`.
pub(crate) fn render_series(
    s: &Series,
    na_rep: &str,
    ff: Option<(Option<usize>, char)>,
    truncate: Option<usize>,
    footer: bool,
) -> String {
    let n = s.len();
    let dtype = s.data.dtype().name();
    if n == 0 {
        return match &s.name {
            Some(name) => format!("Series([], Name: {name}, dtype: {dtype})"),
            None => format!("Series([], dtype: {dtype})"),
        };
    }
    let (rows, truncated) = displayed_rows(n, truncate);
    let mut labels = index_cells(&s.index, &rows);
    let mut values = data_cells(&s.data, &rows, na_rep, ff);
    if truncated {
        let k = rows.len() / 2;
        let vw = values.iter().map(|s| s.chars().count()).max().unwrap_or(0);
        labels.insert(k, String::new());
        // pandas center-justifies the Series ellipsis within the value width.
        values.insert(k, py_center(if vw > 3 { "..." } else { ".." }, vw));
    }
    let mut out = String::new();
    // A named index prints its name on a leading line (pandas).
    if let Some(name) = s.index.name() {
        out.push_str(name);
        out.push('\n');
    }
    out.push_str(&vertical_body(&labels, &values));
    if footer {
        let mut parts: Vec<String> = Vec::new();
        if let Some(name) = &s.name {
            parts.push(format!("Name: {name}"));
        }
        if truncated {
            parts.push(format!("Length: {n}"));
        }
        parts.push(format!("dtype: {dtype}"));
        out.push('\n');
        out.push_str(&parts.join(", "));
    }
    out
}

/// Render a `Row` (a 1-row frame) as pandas renders a row Series: vertical,
/// `column   value`, with a `Name: <row label>, dtype: <dtype>` footer. The
/// dtype is the row's single column type when uniform, else `object` (pandas's
/// upcast).
pub(crate) fn render_row(df: &DataFrame, footer: bool) -> String {
    let labels: Vec<String> = df.names().to_vec();
    // A uniform row formats its values in that one dtype (floats share decimals);
    // a mixed row is `object`, each value formatted in its own dtype.
    let first = df.columns().first().map(|c| c.dtype());
    let uniform = first.filter(|d0| df.columns().iter().all(|c| c.dtype() == *d0));
    let values: Vec<String> = df
        .columns()
        .iter()
        .map(|c| data_cells(c, &[0], "NaN", None).pop().unwrap())
        .collect();
    // When uniform-float, re-derive decimals across the whole row for alignment.
    let values = match uniform {
        Some(volas_core::DType::F64) => {
            let xs: Vec<f64> = df
                .columns()
                .iter()
                .map(|c| c.as_f64().unwrap()[0])
                .collect();
            let d = float_decimals(&xs);
            xs.iter()
                .map(|&x| {
                    if x.is_nan() {
                        "NaN".to_string()
                    } else {
                        lead_num(x < 0.0, format!("{x:.d$}"))
                    }
                })
                .collect()
        }
        _ => values,
    };
    let dtype = match uniform {
        Some(d) => d.name().to_string(),
        None => "object".to_string(),
    };
    let mut out = vertical_body(&labels, &values);
    if footer {
        let label = index_label_csv(df.index(), 0);
        out.push_str(&format!("\nName: {label}, dtype: {dtype}"));
    }
    out
}

// --- HTML (Jupyter) --------------------------------------------------------

/// HTML-escape the five XML special characters.
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}

/// Render a frame as the Jupyter HTML table pandas emits (`_repr_html_`). Cell
/// text uses the same per-column float / datetime formatting as the text table,
/// but without the leading sign-space (HTML right-aligns via CSS).
pub(crate) fn render_frame_html(df: &DataFrame) -> String {
    let n = df.height();
    let cols: Vec<usize> = (0..df.width()).collect();
    // Truncation in HTML mirrors the text repr (max_rows = 60, min_rows = 10).
    let truncate = if n > 60 { Some(5) } else { None };
    let (rows, truncated) = displayed_rows(n, truncate);

    let mut idx = index_cells(df.index(), &rows);
    let mut body: Vec<Vec<String>> = cols
        .iter()
        .map(|&j| {
            data_cells(&df.columns()[j], &rows, "NaN", None)
                .into_iter()
                // strip the leading sign-space; HTML aligns via CSS, not padding
                .map(|s| s.trim_start().to_string())
                .collect()
        })
        .collect();
    if truncated {
        let k = rows.len() / 2;
        idx.insert(k, "...".to_string());
        for col in &mut body {
            col.insert(k, "...".to_string());
        }
    }

    let mut s = String::from(
        "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n",
    );
    for &j in &cols {
        s.push_str(&format!("      <th>{}</th>\n", html_escape(&df.names()[j])));
    }
    s.push_str("    </tr>\n");
    if let Some(name) = df.index().name() {
        s.push_str(&format!("    <tr>\n      <th>{}</th>\n", html_escape(name)));
        for _ in &cols {
            s.push_str("      <th></th>\n");
        }
        s.push_str("    </tr>\n");
    }
    s.push_str("  </thead>\n  <tbody>\n");
    for r in 0..idx.len() {
        s.push_str(&format!(
            "    <tr>\n      <th>{}</th>\n",
            html_escape(&idx[r])
        ));
        for col in &body {
            s.push_str(&format!("      <td>{}</td>\n", html_escape(&col[r])));
        }
        s.push_str("    </tr>\n");
    }
    s.push_str("  </tbody>\n</table>\n");
    if truncated {
        s.push_str(&format!("<p>{} rows × {} columns</p>\n", n, df.width()));
    }
    s.push_str("</div>");
    s
}
