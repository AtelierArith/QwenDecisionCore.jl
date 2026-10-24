module QwenDecisionCoreLoopVectorizationExt

import QwenDecisionCore
using LoopVectorization

# Guard the fast-math loop against exceptional values, signed zeros, and tiny
# outputs. Outside this conservative domain use the unchanged scalar formula.
function eligible_silu(values, span = eachindex(values))
    valid = true
    @inbounds @simd for i in span
        x = values[i]
        valid &= isfinite(x) & (x > -20.0f0) & (x < 80.0f0) & (abs(x) > 1.0f-12)
    end
    return valid
end

function QwenDecisionCore.cpu_portable_softmax!(scores::Matrix{Float32})
    QwenDecisionCore.cpu_setting(:portable_vector_math) || return false
    isempty(scores) && return false
    valid = true
    @inbounds @simd for i in eachindex(scores)
        valid &= isfinite(scores[i])
    end
    valid || return false
    # Preserve rare subnormal probabilities with the scalar exp/division;
    # SIMD lanes use bounded inputs, including exact zero for masked scores.
    indices = Vector{Int}(undef, size(scores, 1))
    differences = Vector{Float32}(undef, size(scores, 1))
    for column in axes(scores, 2)
        largest = -Inf32
        @turbo for row in axes(scores, 1)
            largest = max(largest, scores[row, column])
        end
        count = 0
        @inbounds for row in axes(scores, 1)
            difference = scores[row, column] - largest
            if -104.0f0 <= difference < -80.0f0
                count += 1
                indices[count] = row
                differences[count] = difference
            end
        end
        @turbo for row in axes(scores, 1)
            difference = scores[row, column] - largest
            scores[row, column] =
                ifelse(difference >= -80.0f0, exp(max(difference, -80.0f0)), 0.0f0)
        end
        total = 0.0f0
        @turbo for row in axes(scores, 1)
            total += scores[row, column]
        end
        @turbo for row in axes(scores, 1)
            scores[row, column] /= total
        end
        @inbounds for j = 1:count
            scores[indices[j], column] = exp(differences[j]) / total
        end
    end
    return true
end

function eligible_up(values, span = eachindex(values))
    valid = true
    @inbounds @simd for i in span
        x = abs(values[i])
        valid &= isfinite(x) & (x > 1.0f-12) & (x < 1.0f12)
    end
    return valid
end

function zero_block(values, span)
    valid = true
    @inbounds @simd for i in span
        valid &= iszero(values[i])
    end
    return valid
end

# Leave exceptional lanes untouched in the SIMD pass. Their original values
# are then available for the scalar formula, including NaN and signed zero.
function mixed_silu_block!(output, first, last, exceptions)
    count = 0
    @inbounds for i = first:last
        x = output[i]
        if !(x > -20.0f0 && x < 80.0f0 && abs(x) > 1.0f-12)
            count += 1
            exceptions[count] = i
        end
    end
    @turbo for i = first:last
        x = output[i]
        valid = (x > -20.0f0) & (x < 80.0f0) & (abs(x) > 1.0f-12)
        safe = ifelse(valid, x, 1.0f0)
        output[i] = ifelse(valid, safe * inv(1.0f0 + exp(-safe)), x)
    end
    @inbounds for j = 1:count
        i = exceptions[j]
        x = output[i]
        output[i] = iszero(x) ? x : QwenDecisionCore.native_silu(x)
    end
    return output
end

function mixed_gate_block!(gate, up, first, last, exceptions)
    count = 0
    @inbounds for i = first:last
        x, u = gate[i], abs(up[i])
        if !(x > -20.0f0 && x < 80.0f0 && abs(x) > 1.0f-12 && u > 1.0f-12 && u < 1.0f12)
            count += 1
            exceptions[count] = i
        end
    end
    @turbo for i = first:last
        x, u = gate[i], up[i]
        valid =
            (x > -20.0f0) & (x < 80.0f0) & (abs(x) > 1.0f-12) & (abs(u) > 1.0f-12) &
            (abs(u) < 1.0f12)
        safe_x, safe_u = ifelse(valid, x, 1.0f0), ifelse(valid, u, 1.0f0)
        gate[i] = ifelse(valid, (safe_x * inv(1.0f0 + exp(-safe_x))) * safe_u, x)
    end
    @inbounds for j = 1:count
        i = exceptions[j]
        x = gate[i]
        gate[i] = (iszero(x) ? x : QwenDecisionCore.native_silu(x)) * up[i]
    end
    return gate
end

function silu_blocks!(output, starts)
    exceptions = Vector{Int}(undef, 256)
    for first in starts
        last = min(first + 255, length(output))
        if eligible_silu(output, first:last)
            @turbo for i = first:last
                output[i] = output[i] * inv(1.0f0 + exp(-output[i]))
            end
        elseif !zero_block(output, first:last)
            mixed_silu_block!(output, first, last, exceptions)
        end
    end
    return output
end

function gate_blocks!(gate, up, starts, exceptions = Vector{Int}(undef, 256))
    for first in starts
        last = min(first + 255, length(gate))
        if eligible_silu(gate, first:last) && eligible_up(up, first:last)
            @turbo for i = first:last
                gate[i] = (gate[i] * inv(1.0f0 + exp(-gate[i]))) * up[i]
            end
        elseif zero_block(gate, first:last)
            @inbounds @simd for i = first:last
                gate[i] = (gate[i] * 0.5f0) * up[i]
            end
        else
            mixed_gate_block!(gate, up, first, last, exceptions)
        end
    end
    return gate
end

function QwenDecisionCore.cpu_portable_silu!(output::Matrix{Float32})
    QwenDecisionCore.cpu_setting(:portable_vector_math) || return false
    isempty(output) && return true
    blocks = QwenDecisionCore.cpu_setting(:vector_math_blocks)
    workers = min(Threads.nthreads(:default), cld(length(output), 65536))
    if blocks && workers > 1
        # Interleave disjoint blocks so a long zero prefix cannot leave most
        # workers idle. Workers use owned data and never change BLAS settings.
        @sync for worker = 1:workers
            Threads.@spawn silu_blocks!(
                output,
                (1+256*(worker-1)):(256*workers):length(output),
            )
        end
        return true
    end
    if !eligible_silu(output)
        blocks || return false
        silu_blocks!(output, 1:256:length(output))
        return true
    end
    @turbo for i in eachindex(output)
        output[i] = output[i] * inv(1.0f0 + exp(-output[i]))
    end
    return true
end

function QwenDecisionCore.cpu_portable_gate!(gate::Matrix{Float32}, up::Matrix{Float32})
    QwenDecisionCore.cpu_setting(:portable_vector_math) || return false
    size(gate) == size(up) || return false
    Base.mightalias(gate, up) && return false
    isempty(gate) && return true
    blocks = QwenDecisionCore.cpu_setting(:vector_math_blocks)
    workers = min(Threads.nthreads(:default), cld(length(gate), 65536))
    if blocks && workers > 1
        @sync for worker = 1:workers
            Threads.@spawn gate_blocks!(
                gate,
                up,
                (1+256*(worker-1)):(256*workers):length(gate),
            )
        end
        return true
    end
    if !(eligible_silu(gate) && eligible_up(up))
        blocks || return false
        gate_blocks!(gate, up, 1:256:length(gate))
        return true
    end
    @turbo for i in eachindex(gate, up)
        gate[i] = (gate[i] * inv(1.0f0 + exp(-gate[i]))) * up[i]
    end
    return true
end

# Reduce along the key dimension while keeping each SIMD group of value rows
# in registers, rather than loading/storing the scratch vector for every key.
function QwenDecisionCore.cpu_delta_recurrent_step!(
    state::Matrix{Float32}, projected, result, q::Array{Float32,3},
    k::Array{Float32,3}, v::Array{Float32,3}, beta, factor, token, head, kh,
)
    if !QwenDecisionCore.cpu_setting(:recurrent_vector_math) || isempty(state)
        return invoke(
            QwenDecisionCore.cpu_delta_recurrent_step!,
            Tuple{Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any},
            state, projected, result, q, k, v, beta, factor, token, head, kh,
        )
    end
    @turbo for row in axes(state, 1)
        value = 0.0f0
        for column in axes(state, 2)
            value += state[row, column] * k[column, token, kh]
        end
        projected[row] = beta * (v[row, token, head] - factor * value)
    end
    @turbo for row in axes(state, 1)
        value = 0.0f0
        for column in axes(state, 2)
            updated = factor * state[row, column] + projected[row] * k[column, token, kh]
            state[row, column] = updated
            value += updated * q[column, token, kh]
        end
        result[row] = value
    end
    return nothing
end

# Four-row gate projections are cheaper as dot products than as packed GEMMs.
function QwenDecisionCore.cpu_small_projection!(output::Matrix{Float32}, weight::Matrix{Float32}, input::Matrix{Float32}, beta)
    QwenDecisionCore.cpu_setting(:portable_vector_math) &&
        Threads.nthreads(:default) == 1 && QwenDecisionCore.cpu_projection_blas_threads() == 1 &&
        1 <= size(output,1) <= 4 && size(input,1) >= 64 && size(input,2) >= 16 || return false
    if iszero(beta)
        @turbo for token in axes(output,2), row in axes(output,1)
            value = 0f0
            for inner in axes(input,1)
                value += weight[inner,row]*input[inner,token]
            end
            output[row,token] = value
        end
    else
        @turbo for token in axes(output,2), row in axes(output,1)
            value = 0f0
            for inner in axes(input,1)
                value += weight[inner,row]*input[inner,token]
            end
            output[row,token] = value+beta*output[row,token]
        end
    end
    return true
end

# Two-token updates read each L1-resident state tile twice instead of four
# times. Shared key/query dot products account for the first token's rank-one
# update when predicting the second; an odd tail uses the original recurrence.
function QwenDecisionCore.cpu_delta_tiled_scan!(
    out::Matrix{Float32}, q::Array{Float32,3}, k::Array{Float32,3},
    v::Array{Float32,3}, beta, decay, head, kh,
)
    QwenDecisionCore.cpu_setting(:recurrent_vector_math) || return false
    rows = size(v, 1)
    rows >= 32 && rows % 32 == 0 && 32 <= size(k, 1) <= 128 && size(out, 2) >= 16 || return false
    n = size(out, 2)
    # SIMD exp is accurate in the ordinary decay range. Keep scalar exp for
    # subnormals and exceptional values before deciding whether to fall back.
    factors = Vector{Float32}(undef, n)
    if decay isa Matrix{Float32} && all(d -> -80f0 <= d <= 0f0, @view(decay[head,:]))
        @turbo for token in 1:n
            factors[token] = exp(decay[head,token])
        end
    else
        factors .= exp.(@view decay[head,:])
    end
    pairs = n ÷ 2
    products = Matrix{Float32}(undef, 4, pairs)
    for pair in 1:pairs
        t = 2pair-1
        f1,f2 = factors[t],factors[t+1]
        if !isfinite(f1) || !isfinite(f2) || f1 > 1f0 || f2 > 1f0 ||
            (f1 > 0f0 && f2 > 0f0 && iszero(f1*f2))
            return false
        end
        kk, kq1, kq2, k2q2 = 0f0, 0f0, 0f0, 0f0
        @turbo for column in axes(k, 1)
            a, b = k[column, t, kh], k[column, t+1, kh]
            kk += a*b
            kq1 += a*q[column, t, kh]
            kq2 += a*q[column, t+1, kh]
            k2q2 += b*q[column, t+1, kh]
        end
        @inbounds begin
            products[1,pair] = kk
            products[2,pair] = kq1
            products[3,pair] = kq2
            products[4,pair] = k2q2
        end
    end
    tile = Matrix{Float32}(undef, 16, size(k, 1))
    p1, p2 = Vector{Float32}(undef, 16), Vector{Float32}(undef, 16)
    offset = (head - 1) * rows
    for first in 0:16:(rows-1)
        fill!(tile, 0f0)
        for pair in 1:pairs
            t = 2pair-1
            f1, f2 = factors[t], factors[t+1]
            f12 = f1*f2
            b1, b2 = beta[head, t], beta[head, t+1]
            kk, kq1, kq2, k2q2 = products[1,pair], products[2,pair], products[3,pair], products[4,pair]
            @turbo for row in 1:16
                sk1, sk2, sq1, sq2 = 0f0, 0f0, 0f0, 0f0
                for column in axes(tile, 2)
                    value = tile[row, column]
                    sk1 += value*k[column, t, kh]
                    sk2 += value*k[column, t+1, kh]
                    sq1 += value*q[column, t, kh]
                    sq2 += value*q[column, t+1, kh]
                end
                c1 = b1*(v[first+row, t, head]-f1*sk1)
                c2 = b2*(v[first+row, t+1, head]-f2*(f1*sk2+c1*kk))
                p1[row] = f2*c1
                p2[row] = c2
                out[offset+first+row, t] = f1*sq1+c1*kq1
                out[offset+first+row, t+1] = f12*sq2+f2*c1*kq2+c2*k2q2
            end
            @turbo for column in axes(tile, 2), row in 1:16
                tile[row, column] = f12*tile[row, column] + p1[row]*k[column, t, kh] + p2[row]*k[column, t+1, kh]
            end
        end
        if isodd(n)
            factor, gate = factors[n], beta[head, n]
            @turbo for row in 1:16
                value = 0f0
                for column in axes(tile, 2)
                    value += tile[row, column]*k[column, n, kh]
                end
                p1[row] = gate*(v[first+row, n, head]-factor*value)
            end
            @turbo for row in 1:16
                value = 0f0
                for column in axes(tile, 2)
                    updated = factor*tile[row, column]+p1[row]*k[column, n, kh]
                    tile[row, column] = updated
                    value += updated*q[column, n, kh]
                end
                out[offset+first+row, n] = value
            end
        end
    end
    return true
end

function QwenDecisionCore.cpu_delta_gate_output!(out::Matrix{Float32}, z::Array{Float32,3}, weight::AbstractVector{Float32}, eps, head)
    QwenDecisionCore.cpu_setting(:recurrent_vector_math) || return false
    LoopVectorization.check_args(weight) || return false
    width = size(z,1)
    width > 0 && size(out,2) > 0 || return false
    offset = (head-1)*width
    for token in axes(out,2)
        squared = 0f0
        @turbo for row in 1:width
            squared += abs2(out[offset+row,token])
        end
        scale = inv(sqrt(squared/width+eps))
        @turbo for row in 1:width
            out[offset+row,token] = ((out[offset+row,token]*scale)*weight[row])*z[row,token,head]
        end
    end
    return true
end

end
