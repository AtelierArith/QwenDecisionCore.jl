"""
    QwenDecisionCore

Shared core for native-Julia decision models built on
Qwen3.5 / Qwen3.8 hybrid backbones. It owns the backbone forward pass
(partial RoPE, full attention, Gated DeltaNet, RMS normalization, MLP), the
safetensors loader, Hugging Face checkpoint resolution, the CPU policy, the
accelerator extensions, and the ordered Choice / Noul / Score question types
with their TypeSafe confidence formulas. Decision heads (a linear readout, a
pointer head, ...), tokenization, and checkpoint-format specifics belong to
client packages such as KevClient.
"""
module QwenDecisionCore

import Downloads
import JSON
import Libdl
import Scratch
using LinearAlgebra
import LoopVectorization

export AbstractDecisionBackend
export AbstractQuestion, ChoiceQuestion, NoulQuestion, ScoreQuestion
export option_count, probabilities, answer
export QwenBackbone, backbone_hidden, backbone_last_hidden
export resolve_checkpoint, read_native_weights, write_native_weights

include("interfaces.jl")
include("questions.jl")
include("safetensors.jl")
include("hub.jl")
include("cpu_settings.jl")
include("backbone.jl")
include("backbone_cpu.jl")
include("cpu_mkl.jl")

function __init__()
    reset_cpu_mkl!()
    initialize_cpu!()
end

end # module QwenDecisionCore
