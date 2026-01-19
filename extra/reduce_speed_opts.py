import numpy as np
from tinygrad import Tensor
from tinygrad.device import Device, Buffer
from tinygrad.engine.realize import CompiledRunner
from tinygrad.codegen import get_program
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.uop.ops import pyrender, KernelInfo
from dataclasses import replace

a = Tensor(np.random.default_rng().random((4096, 4096), dtype=np.float32)).realize()
out = a.sum()
s = out.schedule()

print(f'AST:\n{pyrender(s[0].ast.replace(arg=None))}')

renderer = Device[Device.DEFAULT].renderer
big_kernel_ast = s[0].ast

bufs = [Buffer(b.device, b.size, b.dtype).ensure_allocated() if b is not None else None for b in s[0].bufs]

def test_opts(opts_list, name="custom"):
  """Test optimizations: compile, run 8 times, return best time."""
  if opts_list is None:
    prg = get_program(big_kernel_ast, renderer=renderer)
  else:
    ast = big_kernel_ast.replace(arg=KernelInfo(opts_to_apply=tuple(opts_list)))
    prg = get_program(ast, renderer=renderer)

  runner = CompiledRunner(replace(prg, device=bufs[0].device))

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
  print(f"{name:20s} | best: {best*1e6:8.2f} µs | opts: {prg.applied_opts}")
  return best

print("=" * 80)
baseline = test_opts((), "no opts (baseline)")

test_opts([Opt(OptOps.UPCAST, 0, 8), Opt(OptOps.UNROLL, 0, 8)], "UP8 + UNROLL8")

# print("\n--- GROUP only ---")
# test_opts([Opt(OptOps.GROUPTOP, 0, 16)], "GROUPTOP16")
# test_opts([Opt(OptOps.GROUPTOP, 0, 32)], "GROUPTOP32")

# print("\n--- GROUP + UPCAST/UNROLL ---")
# test_opts([Opt(OptOps.GROUPTOP, 0, 16), Opt(OptOps.UPCAST, 0, 4)], "GROUPTOP16 + UP4")
# test_opts([Opt(OptOps.GROUPTOP, 0, 32), Opt(OptOps.UNROLL, 0, 4)], "GROUPTOP32 + UNROLL4")

print("\n--- BEAM/default ---")
beam_time = test_opts(None, "BEAM/default")

print("=" * 80)
print(f"Baseline: {baseline*1e6:.2f} µs | BEAM: {beam_time*1e6:.2f} µs | speedup: {baseline/beam_time:.2f}x")
