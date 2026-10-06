# QwenDecisionCore.jl

Shared, dependency-light core for native-Julia decision models built on
Qwen3.5 / Qwen3.8 hybrid backbones. It owns

- the backbone forward pass (partial RoPE, full attention, Gated DeltaNet,
  RMS normalization, MLP),
- the safetensors reader **and** writer,
- Hugging Face checkpoint resolution and caching,
- the automatic CPU policy and the accelerator extensions (Metal, CUDA,
  AppleAccelerate, LoopVectorization, Octavian, SIMD),
- the ordered `ChoiceQuestion` / `NoulQuestion` / `ScoreQuestion` types and
  their TypeSafe confidence formulas.

Decision heads (a linear readout, a pointer head, …), tokenization and
checkpoint-format specifics belong to client packages such as
[KevClient.jl](https://github.com/AtelierArith/KevClient.jl) and
[JeffClient.jl](https://github.com/AtelierArith/JeffClient.jl).

## Installation

Not registered on the General registry. Add it from GitHub at a pinned tag, or
develop a local checkout:

```julia
using Pkg
Pkg.add(url = "https://github.com/AtelierArith/QwenDecisionCore.jl", rev = "v0.1.0")
# or: Pkg.develop(path = "QwenDecisionCore.jl")
```

## Usage

```julia
using QwenDecisionCore

backbone = QwenBackbone("Qwen/Qwen3.5-0.8B-Base")   # local dir or Hub ID
states = backbone_hidden(backbone, input_ids, attention_mask)
# → (hidden, length) final-normalized hidden states; apply your own head

resolve_checkpoint("org/model"; revision = "dc7cdfe2…")  # local directory
read_native_weights("model.safetensors")                # Dict{String,Any}
write_native_weights("out.safetensors", tensors)        # round trips exactly
```

`QwenBackbone(directory; device = :cpu)` also accepts `:metal`, `:cuda` and
`:amdgpu` after importing Metal / CUDA / AMDGPU. The Apple Accelerate fast path
is an **opt-in weak dependency**: add `AppleAccelerate` to your environment and
load it (`using AppleAccelerate`); its extension forwards BLAS to Accelerate on
Apple silicon and re-applies the CPU policy, in either load order. With it
absent the portable CPU policy runs on every platform, including Linux. A Metal
extension can additionally expose a batched forward through
`batch_backbone_hidden` when `QDC_METAL_BATCHED=1`.

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The suite is offline: it checks the backbone against a committed PyTorch
reference with tiny fixtures, the safetensors round trip, the Hub resolver
against a `file://` endpoint, and the CPU policies and workspaces.
