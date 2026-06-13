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

pub use bind::check_bare_commands;
pub use exec::execute;
pub use parser::parse;
pub use stringify::stringify;
pub use types::{Ast, Op};
