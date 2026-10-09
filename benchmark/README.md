# CPU inference benchmark

Run from the repository root:

```sh
julia -t 4 --project=. benchmark/cpu.jl
julia -t 4 --project=. benchmark/cpu.jl --checkpoint /path/to/local/checkpoint --require-speedup 10
```

The default is a deterministic four-layer hybrid backbone with 128-dimensional
DeltaNet heads. `--checkpoint` measures a loaded model without network access.
The measured operation is `backbone_hidden`, including embedding, all layers,
forward workspace allocation, and final normalization. Both paths return every
token, including masked positions. Loading and compilation are excluded by
warming both paths.

The baseline invokes the original generic all-token forward, scalar recurrent
step, unfused convolution, and original BLAS thread policy. The optimized path
uses the public API with the current CPU defaults. With four Julia threads,
the original policy uses one BLAS thread and the new policy uses four. The
benchmark changes BLAS settings only between completed forwards; the library
sets its policy at initialization, never during inference.

Measurement order alternates over 21 samples per path. The script reports
median wall-clock time and checks all hidden states at `atol=2e-5, rtol=2e-5`.
GC runs before each sample; allocations within a forward are timed. The
aggregate is the sum of the medians for 64, 128, 256, and 512 tokens, representing
one sequence of each length. `--require-speedup 10` fails if aggregate speedup
is below 10%. Improvements depend on the model, input length, hardware and
machine load; run comparisons on an idle machine.

`--last` measures `backbone_last_hidden` instead. Its baseline retains the
existing last-token shortcut and workspaces, with the original recurrent step,
convolution and BLAS policy. This reports the independent effect on last-token
inference without changing its output contract.

A [validated 0.8B model run](results/cpu-0.8b-4threads.txt) on an AMD Ryzen 9 PRO
8945HS with Julia 1.13.1 and four Julia threads reduced the aggregate from
4,341.6 ms to 3,873.1 ms: 12.1% higher inference speed, or 10.8% less time.
The machine had other workloads running. These are paired median measurements,
not a guarantee of the same improvement for every model or input length.
The full offline suite passed all 4,505 tests with both one and four Julia
threads, including all-token output/ownership and recurrent/convolution checks.
