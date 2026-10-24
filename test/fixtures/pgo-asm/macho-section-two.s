.text
.globl "_volas_core::frame::execute"
"_volas_core::frame::execute":
adrp x0, _two@PAGE
add x0, x0, _two@PAGEOFF
ret
.section __TEXT,__const
.p2align 12
_one:
.quad 42
.section __DATA_CONST,__const
.p2align 12
_two:
.quad 43
