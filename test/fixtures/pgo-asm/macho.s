.text
.globl "_volas_core::frame::execute"
"_volas_core::frame::execute":
adrp x0, _named@PAGE
add x0, x0, _named@PAGEOFF
adrp x1, _hidden@PAGE
bl _foreign
ret
.globl "_volas_core::frame::dispatch::closure"
"_volas_core::frame::dispatch::closure":
ret
.data
.globl _named
_named:
.quad 42
.space 0x5000
_hidden:
.quad 43
