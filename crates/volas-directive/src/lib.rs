//! volas-directive: parse `"ma:5@close"`-style strings into an AST and execute
//! them against a `volas_core::DataFrame`.

mod bind;
mod complete;
mod operands;
mod replay;
pub mod exec;
mod exec_resume;
pub mod lookback;
pub mod parser;
pub mod spec;
pub mod stringify;
pub mod types;

pub use bind::{check_ambiguous_names, check_bare, check_form, is_bare_command, uses_bare_command};
pub use exec::execute;
pub use parser::{parse, parse_cst};
pub use replay::{checkpoint_at, refresh_frame, resume_states, states_at, uses_default_series, Replayer, ResumeStates};
pub use stringify::stringify;
pub use types::{Ast, Op};
