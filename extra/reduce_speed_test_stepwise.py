import numpy as np
import functools
from dataclasses import dataclass
from tinygrad import Tensor
from tinygrad.device import Device, Buffer
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.uop.ops import pyrender, KernelInfo, graph_rewrite, UOp, Ops, UPat, PatternMatcher, identity_element
from tinygrad.uop.symbolic import sym, pm_move_where_on_load, gep_pushing
from tinygrad.codegen.late.expander import expander, pm_pre_expander, pm_group_for_reduce
from tinygrad.codegen.late.devectorizer import load_store_folding, load_store_indexing, correct_load_store, pm_add_loads, pm_render, devectorize, devectorize_buf_and_index
from tinygrad.schedule.rangeify import pm_add_buffers_local, rangeify_codegen, pm_mops
from tinygrad.dtype import dtypes, DType, AddrSpace, PtrDType
from tinygrad.helpers import flatten
from tinygrad.renderer import Renderer
from tinygrad.codegen.late.linearizer import linearize, pm_add_control_flow, CFGContext
from tinygrad.codegen.simplify import pm_simplify_ranges, pm_flatten_range, pm_split_ranges, pm_load_collapse, pm_split_store
from tinygrad.codegen.opt.postrange import apply_opts, make_images
from tinygrad.uop.ops import pm_lower_index_dtype

# Setup - get AST with opts applied (same as reduce_speed_opts.py)
print("=" * 80)
print("STEP 0: Initial Setup")
print("=" * 80)
a = Tensor(np.random.default_rng().random((4096, 4096), dtype=np.float32)).realize()
out = a.sum()
s = out.schedule()
big_kernel_ast = s[0].ast.replace(arg=KernelInfo(opts_to_apply=(Opt(OptOps.UPCAST, 0, 8), Opt(OptOps.UNROLL, 0, 8))))
renderer = Device[Device.DEFAULT].renderer

print(f'Initial AST:\n{pyrender(big_kernel_ast.replace(arg=None))}\n')

# ============================================================================
# COPY OF REDUCE FUNCTIONS - MODIFY THESE TO TEST CHANGES
# ============================================================================

@dataclass
class ReduceContext:
  acc_num: int = 0

def horizontal_reduce(inp:UOp, out_dtype:DType) -> list[UOp]:
  # if this has a horizontal reduction component, do that first
  if inp.dtype != out_dtype:
    # NOTE: [0 1 2 3 4 5 6 7] -> [0+4, 1+5, 2+6, 3+7]
    horizontal_amount = inp.dtype.count//out_dtype.count
    return [inp.gep(tuple(range(i, inp.dtype.count, horizontal_amount))) for i in range(0, horizontal_amount)]
  return [inp]

def reduce_to_acc(ctx:ReduceContext, red:UOp):
  inp, reduce_range = red.src[0], red.src[1:]
  lst = horizontal_reduce(inp, red.dtype)
  assert all(x.dtype == red.dtype for x in lst), f"horizontal reduction mismatch {lst[0].dtype} != {red.dtype}"
  # if we have a range
  if len(reduce_range) != 0:
    topo = inp.toposort()
    ended_ranges = flatten([x.ended_ranges for x in topo if x.op is Ops.END])
    input_ranges = tuple([x for x in topo if x.op is Ops.RANGE and x not in reduce_range and x not in ended_ranges])
    identity = red.const(red.dtype, identity_element(red.arg, red.dtype.scalar()))
    acc = UOp(Ops.DEFINE_REG, red.dtype.ptr(size=1, addrspace=AddrSpace.REG), arg=ctx.acc_num)
    acc_init = acc.after(*input_ranges).index(UOp.const(dtypes.int, 0)).store(identity) if len(input_ranges) else \
               acc.index(UOp.const(dtypes.int, 0)).store(identity)
    lst = [acc.after(acc_init, *reduce_range).index(UOp.const(dtypes.int, 0))] + lst  # put acc as the first element
    ctx.acc_num += 1
  ret = functools.reduce(lambda x,y: x.alu(red.arg, y), lst)
  if len(reduce_range) == 0: return ret
  return acc.after(acc.index(UOp.const(dtypes.int, 0)).store(ret).end(*reduce_range)).index(UOp.const(dtypes.int, 0))

# Custom pm_reduce that uses our local reduce_to_acc function
pm_reduce_custom = PatternMatcher([
  # REDUCE -> DEFINE_ACC+ASSIGN
  (UPat(Ops.REDUCE, name="red"), reduce_to_acc),
])

# ============================================================================
# STEP-BY-STEP TRANSFORMATIONS
# ============================================================================

sink = big_kernel_ast

# Preprocessing - needed before optimization
print("=" * 80)
print("STEP 0.5: Preprocessing")
print("=" * 80)

pm_syntactic_sugar = PatternMatcher([
  # INDEX on ptr INDEX concats them
  (UPat(Ops.INDEX, name="i1").f(Ops.INDEX, name="i2", allow_any_len=True),
   lambda i1,i2: i2.replace(src=i1.src+i2.src[1:]) if isinstance(i1.dtype, PtrDType) and not isinstance(i2.dtype, PtrDType) else None),
])

sink = graph_rewrite(sink, pm_mops+pm_syntactic_sugar, name="early movement ops", bottom_up=True)

print(f'AST after preprocessing:\n{pyrender(sink.replace(arg=None))[:500]}...\n')

# Optimization phase - this is where opts_to_apply gets applied
print("=" * 80)
print("STEP 0.6: Optimization Phase (applies opts_to_apply)")
print("=" * 80)

# collapse loads reduce
sink = graph_rewrite(sink, pm_load_collapse, name="load collapse")

# split ranges
sink = graph_rewrite(sink, pm_split_ranges+pm_flatten_range, ctx={}, name="split ranges")

# symbolic (required for pm_simplify_ranges)
sink = graph_rewrite(sink, sym+pm_flatten_range, name="initial symbolic")

# optimize (schedule) the AST
sink = graph_rewrite(sink, pm_simplify_ranges, name="simplify ranges")

# split store range
sink = graph_rewrite(sink, pm_split_store, ctx=renderer.device, name="cut store ranges")

# create image buffers
sink = make_images(sink, renderer)

# do postrange optimization - THIS APPLIES opts_to_apply!
sink = apply_opts(sink, renderer)

print(f'AST after apply_opts (opts should be applied now):\n{pyrender(sink.replace(arg=None))}\n')

print("=" * 80)
print("STEP 1: After Expander (UNROLL creates vectorized inputs)")
print("=" * 80)

# Apply expander transformations
sink = graph_rewrite(sink, sym+pm_move_where_on_load, name="postopt symbolic")
sink = graph_rewrite(sink, sym+pm_pre_expander+pm_group_for_reduce+expander, ctx=renderer, name="expander")

print(f'AST after expander:\n{pyrender(sink.replace(arg=None))}\n')

print("=" * 80)
print("STEP 2: After Adding Local Buffers")
print("=" * 80)

# Add locals
from itertools import count
sink = graph_rewrite(sink, pm_add_buffers_local+rangeify_codegen, ctx=count(0), name="add local buffers")

print(f'AST after adding buffers:\n{pyrender(sink.replace(arg=None))}\n')

print("=" * 80)
print("STEP 3: After reduce_to_acc (REDUCE -> DEFINE_ACC)")
print("=" * 80)

# Apply pm_reduce with our custom reduce_to_acc
sink = graph_rewrite(sink, pm_reduce_custom+gep_pushing, ctx=ReduceContext(), name="remove_reduce")

print(f'AST after reduce_to_acc:\n{pyrender(sink.replace(arg=None))}\n')

print("=" * 80)
print("STEP 4: After Adding Loads")
print("=" * 80)

# Add loads
sink = graph_rewrite(sink, pm_add_loads, name="add loads")

print(f'AST after adding loads:\n{pyrender(sink.replace(arg=None))}\n')

print("=" * 80)
print("STEP 5: After Devectorization")
print("=" * 80)

# Apply devectorization - includes devectorize which has no_vectorized_buf
from tinygrad.helpers import DEVECTORIZE
if DEVECTORIZE >= 2: pm_devectorize = sym+load_store_folding+load_store_indexing
elif DEVECTORIZE: pm_devectorize = sym+devectorize+load_store_folding+correct_load_store+load_store_indexing
else: pm_devectorize = sym+load_store_folding+correct_load_store+load_store_indexing
sink = graph_rewrite(sink, pm_devectorize, ctx=renderer, name="devectorize")

print(f'AST after devectorize:\n{pyrender(sink.replace(arg=None))}\n')

print("=" * 80)
print("STEP 6: Additional Transformations for Rendering")
print("=" * 80)

# Need to add more transformations before rendering
from tinygrad.uop.ops import pm_lower_index_dtype
from tinygrad.uop.symbolic import symbolic
from tinygrad.uop.decompositions import get_late_rewrite_patterns
from tinygrad.helpers import TRANSCENDENTAL, DEVECTORIZE
from tinygrad.codegen.late.devectorizer import devectorize
from tinygrad.uop.symbolic import symbolic_simple

# Lower index dtype
sink = graph_rewrite(sink, pm_lower_index_dtype+load_store_indexing, ctx=renderer.device, name="lower all index dtypes")
sink = graph_rewrite(sink, symbolic, name="post index symbolic")

# Decompositions
supported_ops = tuple(renderer.code_for_op.keys())
pm_decomp = symbolic_simple+get_late_rewrite_patterns(supported_ops, TRANSCENDENTAL>=2)
sink = graph_rewrite(sink, pm_decomp, ctx=renderer.device, name="decompositions")

# Final rewrite with renderer
extra_matcher = renderer.extra_matcher if renderer.extra_matcher is not None else PatternMatcher([])
from tinygrad.codegen.late.linearizer import pm_split_ends
pm_final_rewrite = pm_decomp+pm_render+extra_matcher+pm_split_ends
sink = graph_rewrite(sink, pm_final_rewrite, ctx=renderer.device, name="final rewrite")

# Add control flow
sink = graph_rewrite(sink, pm_add_control_flow, ctx=CFGContext(sink), name="add control flow", bottom_up=True)

print(f'AST after final transformations:\n{pyrender(sink.replace(arg=None))[:2000]}...\n')

print("=" * 80)
print("STEP 7: Linearize and Render to Code")
print("=" * 80)

# Create PROGRAM structure and linearize
from tinygrad.codegen.__init__ import pm_linearize_cleanups, line_rewrite, pm_to_program, do_linearize, do_render
prg = UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=renderer.device)))
prg = graph_rewrite(prg, pm_to_program, ctx=renderer, name="linearize/render")

# Extract rendered code
for u in prg.toposort():
  if u.op == Ops.SOURCE:
    print(f'Generated code:\n{u.arg}\n')
    break
  elif u.op == Ops.LINEAR:
    print(f'Linearized to {len(u.src)} uops')
    print(f'First 20 uops:')
    for i, uop in enumerate(u.src[:20]):
      print(f'  {i}: {uop.op} {uop.dtype}')
    print()

# Create ProgramSpec (same as get_program does)
from tinygrad.renderer import ProgramSpec
prg_spec = ProgramSpec.from_uop(prg)
print("=" * 80)
print("STEP 8: ProgramSpec Created")
print("=" * 80)
print(f'ProgramSpec created: name={prg_spec.name}, device={prg_spec.device}')
print(f'Global size: {prg_spec.global_size}, Local size: {prg_spec.local_size}')
print(f'Applied opts: {prg_spec.applied_opts}')
print(f'\nThis ProgramSpec is what would be passed to CompiledRunner\n')

from tinygrad.engine.realize import CompiledRunner
from dataclasses import replace

bufs = [Buffer(b.device, b.size, b.dtype).ensure_allocated() if b is not None else None for b in s[0].bufs]

# Use prg_spec (ProgramSpec) instead of prg (UOp) - CompiledRunner expects ProgramSpec
runner = CompiledRunner(replace(prg_spec, device=bufs[0].device))

# Warmup
for _ in range(2):
  runner(bufs)
Device[bufs[0].device].synchronize()

# Benchmark
times = []
for _ in range(8):
  et = runner(bufs, wait=True)
  times.append(et)

best = min(times)
print(f"Benchmark | best: {best*1e6:8.2f} µs | opts: {prg_spec.applied_opts}")
