//! volas-directive: parse `"ma:5@close"`-style strings into an AST and execute
//! them against a `volas_core::DataFrame`.

mod bind;
pub mod exec;
mod exec_resume;
pub mod lookback;
pub mod parser;
pub mod spec;
pub mod stringify;
pub mod types;

pub use bind::{check_bare, check_form};
pub use exec::execute;
pub use parser::{parse, parse_cst};
pub use stringify::stringify;
pub use types::{Ast, Op};
