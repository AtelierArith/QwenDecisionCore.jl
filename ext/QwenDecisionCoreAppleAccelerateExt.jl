module QwenDecisionCoreAppleAccelerateExt

import QwenDecisionCore
import AppleAccelerate

function QwenDecisionCore.native_cpu_silu!(output::Matrix{Float32}, exponential::Matrix{Float32})
    QwenDecisionCore.cpu_portable_silu!(output) && return output
    if !Sys.isapple() || !QwenDecisionCore.cpu_setting(:vector_math)
        return invoke(QwenDecisionCore.native_cpu_silu!, Tuple{Any,Any}, output, exponential)
    end
    size(output) == size(exponential) ||
        throw(DimensionMismatch("SiLU scratch shapes differ."))
    exponential .= .-output
    AppleAccelerate.exp!(exponential, exponential)
    output .= output ./ (1.0f0 .+ exponential)
    return output
end

function QwenDecisionCore.cpu_owned_mlp_gate!(gate::Matrix{Float32}, up::Matrix{Float32})
    QwenDecisionCore.cpu_portable_gate!(gate, up) && return gate
    Base.mightalias(gate, up) &&
        return invoke(QwenDecisionCore.cpu_owned_mlp_gate!, Tuple{Any,Any}, gate, up)
    if !Sys.isapple() || !QwenDecisionCore.cpu_setting(:vector_math)
        return invoke(QwenDecisionCore.cpu_owned_mlp_gate!, Tuple{Any,Any}, gate, up)
    end
    size(gate) == size(up) || throw(DimensionMismatch("MLP gate shapes differ."))
    # Check before consuming either array: multiplying first can overflow even
    # when SiLU(gate) * up is finite. Preserve the scalar expression in that case.
    @inbounds for index in eachindex(gate, up)
        product = gate[index] * up[index]
        if isfinite(gate[index]) && isfinite(up[index]) && !isfinite(product)
            return invoke(QwenDecisionCore.cpu_owned_mlp_gate!, Tuple{Any,Any}, gate, up)
        end
    end
    up .*= gate
    gate .= .-gate
    AppleAccelerate.exp!(gate, gate)
    gate .= up ./ (1.0f0 .+ gate)
    return gate
end

# Loading AppleAccelerate forwards BLAS to Accelerate (AppleAccelerate's own
# `__init__`). Re-run the CPU policy here so `accelerate` / `vector_math` are set
# whichever order the caller loads QwenDecisionCore and AppleAccelerate in.
function __init__()
    QwenDecisionCore.initialize_cpu!()
    return nothing
end

end
