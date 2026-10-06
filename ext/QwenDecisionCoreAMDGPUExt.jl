module QwenDecisionCoreAMDGPUExt

import QwenDecisionCore
import AMDGPU
using LinearAlgebra

function QwenDecisionCore.native_array(::Val{:amdgpu}, x)
    AMDGPU.functional() || throw(ArgumentError("AMDGPU is not available on this machine."))
    return AMDGPU.ROCArray(x)
end

# Trim the masked prefix by default, matching the automatic CPU policy the
# client already applies. Removing leading inactive tokens preserves the last
# token's relative positions, so the readout is unchanged while padded inputs
# skip the empty prefix. Set `QDC_AMDGPU_TRIM_PADDING=0` to keep whole
# sequences for matched full-length comparisons with the CUDA / Metal paths.
function QwenDecisionCore.native_sequence_start(::AMDGPU.ROCArray, mask, row)
    start = first(axes(mask, 2))
    get(ENV, "QDC_AMDGPU_TRIM_PADDING", "1") == "1" || return start
    while start < last(axes(mask, 2)) && mask[row, start] == 0
        start += 1
    end
    return start
end

function QwenDecisionCore.delta_solve(
    system::AMDGPU.ROCArray{Float32,2,AMDGPU.Runtime.Mem.HIPBuffer},
    rhs::AbstractMatrix{Float32},
)
    size(system, 1) == size(system, 2) ||
        throw(DimensionMismatch("Triangular system must be square."))
    output = similar(system, Float32, size(system, 1), size(rhs, 2))
    copyto!(output, rhs)
    # The algorithm treats the diagonal as unit; rocBLAS's 'U' diag matches
    # `UnitLowerTriangular(system) \ rhs` on the CPU.
    AMDGPU.rocBLAS.trsm!('L', 'L', 'N', 'U', 1.0f0, system, output)
    return output
end

include("amdgpu_workspace.jl")
include("amdgpu_delta.jl")
include("amdgpu_attention.jl")
include("amdgpu_mlp.jl")

end
