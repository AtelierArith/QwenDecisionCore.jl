function prepare_head_kernel!(
    output,
    projected,
    norm,
    dim,
    heads,
    source_heads,
    tokens,
    rotary,
    theta,
    eps,
    query,
    offset = Int32(0),
    row_offset = Int32(0),
)
    lane = Int32(CUDA.threadIdx().x)
    group = Int32(CUDA.blockIdx().x)-Int32(1)
    head = rem(group, heads)+Int32(1)
    token = group÷heads+Int32(1)
    source_head = query ? head : cld(head, heads÷source_heads)
    source_width = query ? Int32(2)*dim : dim
    base = (source_head-Int32(1))*source_width+row_offset
    total = 0.0f0
    @inbounds for row = lane:Int32(32):dim
        value = projected[base+row, token]
        total += value*value
    end
    scale = inv(sqrt(warp_sum(total)/Float32(dim)+eps))
    half = rotary÷Int32(2)
    @inbounds for row = lane:Int32(32):dim
        value = projected[base+row, token]*scale*(1.0f0+norm[row])
        if row <= rotary
            first_row = row <= half ? row : row-half
            partner = row <= half ? row+half : row-half
            other = projected[base+partner, token]*scale*(1.0f0+norm[partner])
            angle =
                Float32(token-Int32(1)+offset) /
                theta^(Float32(Int32(2)*(first_row-Int32(1)))/Float32(rotary))
            sine, cosine = sincos(angle)
            value = row <= half ? value*cosine-other*sine : value*cosine+other*sine
        end
        output[row, token, head] = value
    end
    return
end

function prepare_value_kernel!(output, input, dim, heads, source_heads, tokens, row_offset = Int32(0))
    index =
        (Int32(CUDA.blockIdx().x)-Int32(1))*Int32(CUDA.blockDim().x)+Int32(
            CUDA.threadIdx().x,
        )
    if index <= dim*tokens*heads
        row = rem(index-Int32(1), dim)+Int32(1)
        column = (index-Int32(1))÷dim
        token = rem(column, tokens)+Int32(1)
        head = column÷tokens+Int32(1)
        source_head = cld(head, heads÷source_heads)
        @inbounds output[index] = input[row_offset+(source_head-Int32(1))*dim+row, token]
    end
    return
end

# `offset` is the position of the first query column minus one.
function attention_softmax_kernel!(scores, mask, tokens, scale, offset = Int32(0))
    lane = Int32(CUDA.threadIdx().x)
    query = Int32(CUDA.blockIdx().x)
    head = Int32(CUDA.blockIdx().y)
    peak = -floatmax(Float32)
    @inbounds for key = lane:Int32(32):tokens
        value =
            key <= query+offset && mask[key] == 1 ? scores[key, query, head]*scale :
            -floatmax(Float32)
        scores[key, query, head] = value
        peak = max(peak, value)
    end
    for offset in (16, 8, 4, 2, 1)
        peak = max(peak, CUDA.shfl_xor_sync(CUDA.FULL_MASK, peak, offset))
    end
    total = 0.0f0
    @inbounds for key = lane:Int32(32):tokens
        value = exp(scores[key, query, head]-peak)
        scores[key, query, head] = value
        total += value
    end
    total = warp_sum(total)
    @inbounds for key = lane:Int32(32):tokens
        scores[key, query, head] /= total
    end
    return
end

function merge_gate_kernel!(output, values, qgate, dim, heads, tokens)
    index =
        (Int32(CUDA.blockIdx().x)-Int32(1))*Int32(CUDA.blockDim().x)+Int32(
            CUDA.threadIdx().x,
        )
    if index <= dim*heads*tokens
        row = rem(index-Int32(1), dim)+Int32(1)
        column = (index-Int32(1))÷dim
        head = rem(column, heads)+Int32(1)
        token = column÷heads+Int32(1)
        @inbounds output[index] =
            values[row, token, head]*QwenDecisionCore.native_sigmoid(
                qgate[(head-Int32(1))*Int32(2)*dim+dim+row, token],
            )
    end
    return
end

function QwenDecisionCore.full_attention(attention, x::CUDA.CuMatrix{Float32}, mask, cfg)
    tokens = size(x, 2)
    dim, heads, kv_heads = cfg.head_dim, cfg.heads, cfg.kv_heads
    if hasproperty(attention, :packed_projection)
        projected = QwenDecisionCore.native_linear(attention.packed_projection, x)
        q_width, k_width = size(attention.q, 2), size(attention.k, 2)
        # Kernels index the packed rows directly: SubArray rows halve their speed.
        qgate, kp, vp = projected, projected, projected
        k_offset, v_offset = q_width, q_width+k_width
    else
        qgate = QwenDecisionCore.native_linear(attention.q, x)
        kp = QwenDecisionCore.native_linear(attention.k, x)
        vp = QwenDecisionCore.native_linear(attention.v, x)
        k_offset, v_offset = 0, 0
    end
    q = scratch(Float32, (dim, tokens, heads))
    k = scratch(Float32, size(q))
    v = scratch(Float32, size(q))
    CUDA.@cuda threads=32 blocks=heads*tokens prepare_head_kernel!(
        q,
        qgate,
        attention.q_norm,
        Int32(dim),
        Int32(heads),
        Int32(heads),
        Int32(tokens),
        Int32(cfg.rotary_dim),
        cfg.rope_theta,
        cfg.eps,
        true,
    )
    CUDA.@cuda threads=32 blocks=heads*tokens prepare_head_kernel!(
        k,
        kp,
        attention.k_norm,
        Int32(dim),
        Int32(heads),
        Int32(kv_heads),
        Int32(tokens),
        Int32(cfg.rotary_dim),
        cfg.rope_theta,
        cfg.eps,
        false,
        Int32(0),
        Int32(k_offset),
    )
    CUDA.@cuda threads=256 blocks=cld(length(v), 256) prepare_value_kernel!(
        v,
        vp,
        Int32(dim),
        Int32(heads),
        Int32(kv_heads),
        Int32(tokens),
        Int32(v_offset),
    )
    scores = scratch(Float32, (tokens, tokens, heads))
    batched_matmul!(scores, k, q, 'T', 'N')
    device_mask = mask isa CUDA.CuArray ? mask : QwenDecisionCore.native_prepare_mask(x, mask)
    CUDA.@cuda threads=32 blocks=(tokens, heads) attention_softmax_kernel!(
        scores,
        device_mask,
        Int32(tokens),
        inv(sqrt(Float32(dim))),
    )
    values = scratch(Float32, size(q))
    batched_matmul!(values, v, scores, 'N', 'N')
    merged = scratch(Float32, (dim*heads, tokens))
    CUDA.@cuda threads=256 blocks=cld(length(merged), 256) merge_gate_kernel!(
        merged,
        values,
        qgate,
        Int32(dim),
        Int32(heads),
        Int32(tokens),
    )
    return QwenDecisionCore.native_linear(attention.out, merged)
end

# The readout needs only the last position of the final layer: K and V cover
# every token, while the query, output projection and MLP run on one column.
function full_attention_last(attention, x::CUDA.CuMatrix{Float32}, mask, cfg)
    tokens = size(x, 2)
    dim, heads, kv_heads = cfg.head_dim, cfg.heads, cfg.kv_heads
    packed = attention.packed_projection
    q_width, k_width = size(attention.q, 2), size(attention.k, 2)
    # Column blocks of the packed weight and the last column are dense views.
    kvp = QwenDecisionCore.native_linear(view(packed, :, (q_width+1):size(packed, 2)), x)
    qgate = QwenDecisionCore.native_linear(view(packed, :, 1:q_width), view(x, :, tokens:tokens))
    q = scratch(Float32, (dim, 1, heads))
    k = scratch(Float32, (dim, tokens, heads))
    v = scratch(Float32, size(k))
    CUDA.@cuda threads=32 blocks=heads prepare_head_kernel!(
        q,
        qgate,
        attention.q_norm,
        Int32(dim),
        Int32(heads),
        Int32(heads),
        Int32(1),
        Int32(cfg.rotary_dim),
        cfg.rope_theta,
        cfg.eps,
        true,
        Int32(tokens-1),
    )
    CUDA.@cuda threads=32 blocks=heads*tokens prepare_head_kernel!(
        k,
        kvp,
        attention.k_norm,
        Int32(dim),
        Int32(heads),
        Int32(kv_heads),
        Int32(tokens),
        Int32(cfg.rotary_dim),
        cfg.rope_theta,
        cfg.eps,
        false,
        Int32(0),
        Int32(0),
    )
    CUDA.@cuda threads=256 blocks=cld(length(v), 256) prepare_value_kernel!(
        v,
        kvp,
        Int32(dim),
        Int32(heads),
        Int32(kv_heads),
        Int32(tokens),
        Int32(k_width),
    )
    scores = scratch(Float32, (tokens, 1, heads))
    batched_matmul!(scores, k, q, 'T', 'N')
    device_mask = mask isa CUDA.CuArray ? mask : QwenDecisionCore.native_prepare_mask(x, mask)
    CUDA.@cuda threads=32 blocks=(1, heads) attention_softmax_kernel!(
        scores,
        device_mask,
        Int32(tokens),
        inv(sqrt(Float32(dim))),
        Int32(tokens-1),
    )
    values = scratch(Float32, size(q))
    batched_matmul!(values, v, scores, 'N', 'N')
    merged = scratch(Float32, (dim*heads, 1))
    CUDA.@cuda threads=256 blocks=cld(length(merged), 256) merge_gate_kernel!(
        merged,
        values,
        qgate,
        Int32(dim),
        Int32(heads),
        Int32(1),
    )
    return QwenDecisionCore.native_linear(attention.out, merged)
end

function last_token_layer(layer, hidden, mask, cfg)
    normalized = QwenDecisionCore.native_rms(hidden, layer.input_norm, cfg.eps)
    mixed = full_attention_last(layer.attention, normalized, mask, cfg)
    last = view(hidden, :, size(hidden, 2):size(hidden, 2))
    residual, normed = QwenDecisionCore.native_residual_rms(last, mixed, layer.post_norm, cfg.eps)
    return QwenDecisionCore.native_residual_add!(residual, QwenDecisionCore.native_mlp(layer.mlp, normed))
end
