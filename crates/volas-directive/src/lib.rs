//! volas-directive: parse `"ma:5@close"`-style strings into an AST and execute
//! them against a `volas_core::DataFrame`.

pub mod exec;
pub mod lookback;
pub mod parser;
pub mod types;

pub use exec::execute;
pub use parser::parse;
pub use types::{Command, Node, Op};
