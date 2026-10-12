//! One name per bound command: every spelling of every output line that binds to the
//! same command — its head and aliases in any case, with or without an empty argument
//! list, each argument written as its default or as any spelling of its value
//! (zero-padded, signed, as a float), defaults left empty or given — renders to one
//! canonical name, and that name binds back to the same command.

mod matrix;

use std::collections::BTreeMap;

use volas_directive::spec::{canon_sub, ArgBound};
use volas_directive::types::{ArgValue, Node};
use volas_directive::{bind_command, parse, stringify};

/// A deterministic pseudo-random sequence (xorshift; no RNG dependency).
struct Rng(u64);

impl Rng {
    fn next(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }
}

/// Spellings of an argument value: its token and other spellings of the same value.
fn spell(value: &ArgValue, rng: &mut Rng) -> String {
    let token = value.to_token();
    let forms = match value {
        ArgValue::Usize(_) => vec![token.clone(), format!("0{token}"), format!("+{token}"), format!("{token}.0")],
        ArgValue::F64(v) if *v >= 0.0 => vec![token.clone(), format!("{v:.6}"), format!("+{token}")],
        _ => vec![token.clone()],
    };
    forms[rng.next(forms.len())].clone()
}

/// The bound command a directive names: `(name, sub, argument tokens)` — a bare name
/// that is a command bound as that command.
fn identity(text: &str) -> Option<(String, Option<String>, Vec<String>)> {
    let node = parse(text).ok()?;
    let cmd = match &*node {
        Node::Command(c) => c.clone(),
        Node::Name(name) => bind_command(name, None, &[], &[]).ok()?,
        _ => return None,
    };
    if !cmd.series.iter().all(|s| matches!(s, Node::Name(n) if n.is_empty() || n == "close" || n == "open" || n == "volume")) {
        return None;
    }
    // an empty slot reads the slot's default column
    let series: Vec<String> = cmd
        .series
        .iter()
        .enumerate()
        .map(|(i, s)| match s {
            Node::Name(n) if n.is_empty() => cmd.args.spec.series[i].to_string(),
            other => stringify(other),
        })
        .collect();
    let mut args: Vec<String> = cmd.args.values.iter().map(ArgValue::to_token).collect();
    args.extend(series);
    Some((cmd.name, cmd.sub, args))
}

#[test]
fn every_spelling_of_a_bound_command_has_one_name() {
    let mut names: BTreeMap<(String, Option<String>, Vec<String>), BTreeMap<String, String>> = BTreeMap::new();
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let mut spellings = 0usize;
    for line in matrix::lines() {
        let (cmd, sub) = line.head.split_once('.').map_or((line.head.as_str(), None), |(c, s)| (c, Some(s)));
        let canonical_sub = canon_sub(cmd, sub).map(|s| s.into_owned());
        let heads = [line.head.clone(), line.head.to_uppercase(), {
            let mut t = line.head.clone();
            t[..1].make_ascii_uppercase();
            t
        }];
        // the default argument values (a required one at its bound's minimum, at least 2)
        let defaults: Vec<ArgValue> = line
            .args
            .iter()
            .map(|a| {
                a.default.value().unwrap_or_else(|| match a.bound {
                    ArgBound::IntMin(m) | ArgBound::IntRange(m, _) => ArgValue::Usize(m.max(2)),
                    _ => unreachable!("every required argument is an integer"),
                })
            })
            .collect();
        let mut texts = Vec::new();
        for head in &heads {
            texts.push(format!("{head}{}", line.series));
            texts.push(format!("{head}:{}", line.series));
            for _ in 0..12 {
                let toks: Vec<String> = defaults
                    .iter()
                    .zip(&line.args)
                    .map(|(v, a)| {
                        if a.default.value().is_some() && rng.next(3) == 0 {
                            String::new() // left to its default
                        } else {
                            spell(v, &mut rng)
                        }
                    })
                    .collect();
                let cut = rng.next(toks.len() + 1);
                texts.push(format!("{head}:{}{}", toks[..cut].join(","), line.series));
            }
        }
        for text in texts {
            let Some(id) = identity(&text) else { continue };
            assert_eq!(id.1, canonical_sub, "{text}");
            let name = stringify(&parse(&text).unwrap());
            spellings += 1;
            names.entry(id).or_default().insert(name, text);
        }
    }
    assert!(spellings > 2000, "only {spellings} spellings bound");
    let split: Vec<_> = names.iter().filter(|(_, n)| n.len() > 1).collect();
    assert!(split.is_empty(), "{} bound command(s) with several names, e.g. {:?}", split.len(), split.first());
    // and each name binds back to its command
    for (id, by_name) in &names {
        for name in by_name.keys() {
            assert_eq!(identity(name).as_ref(), Some(id), "{name}");
        }
    }
}
