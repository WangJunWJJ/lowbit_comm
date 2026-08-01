# GPU-first sharded reduce-scatter smoke

This auxiliary smoke verifies that the Task 0 sharded benchmark emits the same
versioned result schema on both 2 and 4 A6000 GPUs. It is not part of the 12-run
all-reduce performance gate.

| GPUs | dtype | elements | native full-reduce + shard ms | CCDL compressed shard ms | ratio | relative L2 |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 2 | FP16 | 8,388,608 | 2.4904 | 0.8306 | 0.334 | 0.005943 |
| 4 | FP16 | 8,388,608 | 3.6000 | 1.2857 | 0.357 | 0.005944 |

Both runs used commit `7850148`, group size 64, 5 warm-up iterations, and 10
measured iterations. All four standardized records contain zero non-finite
values.
