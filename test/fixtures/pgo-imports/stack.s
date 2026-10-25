.text
.globl "volas_core::frame::execute"
"volas_core::frame::execute":
stp x29, x30, [sp, #-128]!
add x29, sp, #80
ldur q0, [sp, #60]
stur q0, [sp, #60]
stp x3, x4, [x29, #-16]!
ldp x3, x4, [x29], #16
ldur x1, [x29, #-8]
stur x1, [x29, #-8]
str x2, [sp, #8]!
ldr x2, [sp], #8
ldp x3, x4, [sp, #16]
stnp x3, x4, [sp, #16]
ldnp x3, x4, [sp, #16]
ldrsb x5, [sp, #32]
ldrsh x5, [x29, #-16]
ldrsw x5, [sp, #32]
ld1 {v0.16b}, [sp]
st1 {v0.16b}, [sp]
prfm pldl1keep, [sp]
str x0, [x19]
mov x29, x19
ldur x0, [x29, #-8]
ldp x29, x30, [sp], #128
ret
.globl "volas_core::frame::dispatch"
"volas_core::frame::dispatch":
ldur x0, [x29, #-8]
ret
.globl "volas_core::frame::merge"
"volas_core::frame::merge":
mov x29, sp
cbz x0, 1f
mov x29, x19
1:
ldur x0, [x29, #-8]
str x0, [sp]
ret
.globl "volas_core::frame::indirect"
"volas_core::frame::indirect":
mov x29, sp
br x0
ldur x1, [x29, #-8]
str x1, [sp]
ldp x29, x30, [sp]
b "volas_core::frame::dispatch"
