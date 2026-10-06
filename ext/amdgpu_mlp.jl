# rocBLAS is measurably faster for contiguous `N`,`N` products on RDNA
# integrated GPUs than for the transposed inputs the CPU and CUDA paths pass.
# The AMDGPU backend therefore materializes every projection weight transposed
# at load time and pairs it with a `native_linear` that does not transpose
# again.
struct PackedMLP{M}
    gate_up::M
    down::M
end

# Transpose one linear weight stored as `(in, out)` into `(out, in)`. Norm
# weights and other 1-D tensors are left untouched.
amd_transpose(weight) = permutedims(weight)

function QwenDecisionCore.native_attention_weights(::Val{:amdgpu}, attention)
    if attention.kind == :full
        return merge(
            attention,
            (
                q = amd_transpose(attention.q),
                k = amd_transpose(attention.k),
                v = amd_transpose(attention.v),
                out = amd_transpose(attention.out),
            ),
        )
    end
    qkv = amd_transpose(attention.qkv)
    z = amd_transpose(attention.z)
    a = amd_transpose(attention.a)
    b = amd_transpose(attention.b)
    # Stack the four projections along their output dimension, matching the
    # CUDA packing order so `projected` rows slice as [qkv z a b].
    packed = vcat(qkv, z, a, b)
    q = size(qkv, 1)
    zz = size(z, 1)
    aa = size(a, 1)
    return merge(
        attention,
        (
            qkv = view(packed, 1:q, :),
            z = view(packed, (q+1):(q+zz), :),
            a = view(packed, (q+zz+1):(q+zz+aa), :),
            b = view(packed, (q+zz+aa+1):size(packed, 1), :),
            out = amd_transpose(attention.out),
            packed_projection = packed,
        ),
    )
end

function QwenDecisionCore.native_mlp_weights(::Val{:amdgpu}, gate, up, down)
    return PackedMLP(vcat(amd_transpose(gate), amd_transpose(up)), amd_transpose(down))
end

# Every weight reaching this method was transposed at load time, so the linear
# product is a plain contiguous `N`,`N` GEMM.
function QwenDecisionCore.native_linear(
    weight::AMDGPU.ROCArray{Float32,2,AMDGPU.Runtime.Mem.HIPBuffer},
    x::AMDGPU.ROCArray{Float32},
)
    return QwenDecisionCore.native_matmul(weight, x)
end

function mlp_gate_kernel!(output, packed, width, tokens)
    index =
        (Int32(AMDGPU.blockIdx().x)-Int32(1))*Int32(AMDGPU.blockDim().x)+Int32(
            AMDGPU.threadIdx().x,
        )
    if index <= width*tokens
        row = rem(index-Int32(1), width)+Int32(1)
        token = (index-Int32(1))÷width+Int32(1)
        @inbounds output[index] =
            QwenDecisionCore.native_silu(packed[row, token])*packed[width+row, token]
    end
    return
end

function QwenDecisionCore.native_mlp(
    mlp::PackedMLP,
    x::AMDGPU.ROCArray{Float32,2,ROCBuffer},
)
    projected = QwenDecisionCore.native_linear(mlp.gate_up, x)
    width = size(projected, 1)÷2
    gated = scratch(Float32, (width, size(x, 2)))
    AMDGPU.@roc groupsize = 256 gridsize = cld(length(gated), 256) mlp_gate_kernel!(
        gated,
        projected,
        Int32(width),
        Int32(size(x, 2)),
    )
    return QwenDecisionCore.native_linear(mlp.down, gated)
end
