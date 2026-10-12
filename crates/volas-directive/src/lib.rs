//! volas-directive: parse `"ma:5@close"`-style strings into an AST and execute
//! them against a `volas_core::DataFrame`.

mod bind;
mod complete;
mod operands;
pub mod exec;
mod exec_resume;
pub mod lookback;
pub mod parser;
pub mod spec;
pub mod stringify;
pub mod types;

pub use bind::{bind_command, bind_form, bind_frame, check_bare, is_bare_command, uses_bare_command};
pub use exec::execute;
pub use parser::{parse, parse_cst};
pub use stringify::stringify;
pub use types::{Ast, Op};
