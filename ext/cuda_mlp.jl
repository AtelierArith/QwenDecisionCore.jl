struct PackedMLP{M}
    gate_up::M
    down::M
end

# Store projection rows contiguously so inference uses cuBLAS NN products.
cuda_projection_storage(weight) = transpose(permutedims(weight))

function QwenDecisionCore.native_attention_weights(::Val{:cuda}, attention)
    if attention.kind == :full
        # One projection instead of three: the narrow K/V products alone run
        # at a fraction of cuBLAS's throughput.
        packed = hcat(attention.q, attention.k, attention.v)
        q = size(attention.q, 2)
        k = size(attention.k, 2)
        return merge(
            attention,
            (
                q = view(packed, :, 1:q),
                k = view(packed, :, (q+1):(q+k)),
                v = view(packed, :, (q+k+1):size(packed, 2)),
                packed_projection = packed,
            ),
        )
    end
    attention.kind == :delta || return attention
    packed = cuda_projection_storage(hcat(attention.qkv, attention.z, attention.a, attention.b))
    storage = parent(packed)
    q = size(attention.qkv, 2)
    z = size(attention.z, 2)
    a = size(attention.a, 2)
    return merge(
        attention,
        (
            qkv = transpose(view(storage, 1:q, :)),
            z = transpose(view(storage, (q+1):(q+z), :)),
            a = transpose(view(storage, (q+z+1):(q+z+a), :)),
            b = transpose(view(storage, (q+z+a+1):size(storage, 1), :)),
            out = cuda_projection_storage(attention.out),
            packed_projection = packed,
        ),
    )
end

function QwenDecisionCore.native_mlp_weights(::Val{:cuda}, gate, up, down)
    return PackedMLP(cuda_projection_storage(hcat(gate, up)), cuda_projection_storage(down))
end

function mlp_gate_kernel!(output, packed, width, tokens)
    index =
        (Int32(CUDA.blockIdx().x)-Int32(1))*Int32(CUDA.blockDim().x)+Int32(
            CUDA.threadIdx().x,
        )
    if index <= width*tokens
        row = rem(index-Int32(1), width)+Int32(1)
        token = (index-Int32(1))÷width+Int32(1)
        @inbounds output[index] =
            QwenDecisionCore.native_silu(packed[row, token])*packed[width+row, token]
    end
    return
end

function QwenDecisionCore.native_mlp(mlp::PackedMLP, x::CUDA.CuMatrix{Float32})
    projected = QwenDecisionCore.native_linear(mlp.gate_up, x)
    width = size(projected, 1)÷2
    gated = scratch(Float32, (width, size(x, 2)))
    CUDA.@cuda threads=256 blocks=cld(length(gated), 256) mlp_gate_kernel!(
        gated,
        projected,
        Int32(width),
        Int32(size(x, 2)),
    )
    return QwenDecisionCore.native_linear(mlp.down, gated)
end
