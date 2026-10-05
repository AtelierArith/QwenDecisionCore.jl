# Offline tests for the shared Qwen3.5 / Qwen3.8 core: the safetensors reader,
# the Hugging Face resolver, the ordered question types, and the CPU backbone
# and its diagnostic policies. No network, no accelerator package and no
# decision head are required.
using Test
import JSON
import LinearAlgebra
using QwenDecisionCore
using QwenDecisionCore:
    QwenBackbone,
    backbone_hidden,
    backbone_last_hidden,
    read_native_weights,
    write_native_weights

const FIXTURE = joinpath(@__DIR__, "fixtures", "native")

fixture_readout() =
    read_native_weights(joinpath(FIXTURE, "readout.safetensors"))["weight"]

# A stand-in linear head: the core owns the backbone, so tests probe the final
# hidden state with the fixture readout to reproduce the PyTorch reference.
function probe_logits(backend, inputs, readout = fixture_readout())
    ids, mask = inputs["input_ids"], inputs["attention_mask"]
    result = Matrix{Float32}(undef, size(ids, 1), size(readout, 2))
    for row in axes(ids, 1)
        states = backbone_hidden(backend, vec(ids[row, :]), vec(mask[row, :]))
        result[row, :] .= transpose(readout) * states[:, end]
    end
    return result
end

include("questions.jl")
include("safetensors.jl")
include("hub.jl")
include("backbone.jl")
