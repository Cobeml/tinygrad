import numpy as np
from tinygrad import Tensor, GlobalCounters
import time

if __name__ == "__main__":
  a = Tensor(np_array:=np.random.default_rng().random((4096, 4096), dtype=np.float32)).realize()

  for _ in range(10):
    a.sum().realize()

  GlobalCounters.reset()
  st = time.perf_counter()
  out = a.sum().realize()
  et = time.perf_counter() - st
  print(f"final run: {et*1000:.2f} ms")

  np.testing.assert_allclose(out.item(), np_array.sum(), atol=1e-6, rtol=1e-5)
