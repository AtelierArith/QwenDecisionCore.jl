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

## Single-thread CPU and CUDA revision comparison

`inference.jl` measures the public `backbone_last_hidden` API, including input
preparation, device transfers, workspace management, and owned host output.
Julia and BLAS both use **one thread**, independently of the library's automatic
BLAS policy. The deterministic synthetic hybrid model and lengths match the
CPU benchmark above. Every length has four warm forwards, followed by 21 timed
samples with GC before each sample. Compilation, initial workspace construction,
and CUDA graph recording are excluded. The aggregate is the sum of the four
medians; speedup is `100 * (baseline / current - 1)`.

Use an unchanged checkout of the baseline revision and the **same script** for
both measurements, on an idle machine:

```sh
julia -t 1 --project=/path/to/baseline benchmark/inference.jl --output /tmp/cpu-before.json
julia -t 1 --project=. benchmark/inference.jl --baseline /tmp/cpu-before.json --require-speedup 15
```

For CUDA, create two temporary environments with the same CUDA version and
other dependencies, developing the respective checkouts into each environment.
Then run:

```sh
julia -t 1 --project=/path/to/cuda-baseline-env benchmark/inference.jl --device cuda --output /tmp/cuda-before.json
julia -t 1 --project=/path/to/cuda-current-env benchmark/inference.jl --device cuda --baseline /tmp/cuda-before.json --require-speedup 15
julia -t 1 --project=/path/to/cuda-current-env test/cuda.jl
```

Both comparisons check every output against the saved baseline with
`atol=2e-5, rtol=2e-5`. `--all` measures the independent all-token API;
`--checkpoint /local/model` measures an actual checkpoint without downloading
weights. The 15% gate applies to the selected benchmark aggregate, not every
individual sequence length or checkpoint. CUDA is the GPU backend measured
here; Metal and AMDGPU performance is not claimed.

CPU forwards retain one task-private scratch set, replaced on shape or policy
changes; no model weights are retained, and returned arrays remain owned by the
caller. Nested and concurrent tasks use independent scratch. The internal
`:reuse_workspace => false` diagnostic scope disables reuse. CUDA keeps one
warmed computation graph per model workspace and invalidates it on buffer,
layer, shape, math-policy, algorithm, or all-token-path changes. Input uploads
and host readback remain outside capture. `QDC_CUDA_GRAPH=0` disables recording
and replay for diagnostics. Cold starts and frequently changing input lengths
may benefit less than repeated warmed forwards.

A [2026-10-24 revision comparison](results/inference-1thread-cuda-2026-10-24.txt)
on a Xeon E5-2699 v3 and RTX 3060 passed both 15% gates:

| Backend | Baseline aggregate | Current aggregate | Higher inference speed |
| --- | ---: | ---: | ---: |
| CPU, Julia/BLAS one thread | 134.145 ms | 115.953 ms | 15.7% |
| CUDA, RTX 3060 | 7.599 ms | 6.421 ms | 18.3% |

The baseline is `e8a3db6`. The CPU path also reuses final-query attention
projections and RoPE tables. For longer CUDA sequences, one wider state product
replaces separate correction and query products; the packed output is converted
back to the original token layout before normalization and gating.
