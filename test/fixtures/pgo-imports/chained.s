.text
.globl "volas_core::frame::execute"
"volas_core::frame::execute":
bl foreign
adrp x0, foreign@GOTPAGE
ldr x0, [x0, foreign@GOTPAGEOFF]
adrp x1, slot@PAGE
ldr x1, [x1, slot@PAGEOFF]
ret
.data
.p2align 3
slot:
.quad foreign+ADDEND
