module QwenDecisionCoreCUDAExt

import QwenDecisionCore
import CUDA
using LinearAlgebra

function QwenDecisionCore.native_array(::Val{:cuda}, x)
    CUDA.functional() || throw(ArgumentError("CUDA is not available on this machine."))
    return CUDA.CuArray(x)
end

# Keep complete sequences by default for matched performance comparisons.
function QwenDecisionCore.native_sequence_start(::CUDA.CuArray, mask, row)
    start = first(axes(mask, 2))
    get(ENV, "QDC_CUDA_TRIM_PADDING", "0") == "1" || return start
    while start < last(axes(mask, 2)) && mask[row, start] == 0
        start += 1
    end
    return start
end

function QwenDecisionCore.delta_solve(system::CUDA.CuMatrix{Float32}, rhs::CUDA.CuMatrix{Float32})
    return UnitLowerTriangular(system) \ rhs
end

include("cuda_workspace.jl")
include("cuda_delta.jl")
include("cuda_attention.jl")
include("cuda_mlp.jl")

end
