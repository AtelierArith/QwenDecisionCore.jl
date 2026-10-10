@inline function warp_sum(value::Float32)
    for offset in (16, 8, 4, 2, 1)
        value += CUDA.shfl_xor_sync(CUDA.FULL_MASK, value, offset)
    end
    return value
end

function qk_normalize_kernel!(
    query,
    key,
    mixed,
    dim,
    heads,
    width,
    tokens,
    ::Val{PARTS},
) where {PARTS}
    lane = Int32(CUDA.threadIdx().x)
    group = Int32(CUDA.blockIdx().x) - Int32(1)
    head = rem(group, heads) + Int32(1)
    token = group ÷ heads + Int32(1)
    qs = ntuple(Val(PARTS)) do p
        channel = lane + Int32(32 * (p - 1))
        channel <= dim ? mixed[(head-Int32(1))*dim+channel, token] : 0.0f0
    end
    ks = ntuple(Val(PARTS)) do p
        channel = lane + Int32(32 * (p - 1))
        channel <= dim ? mixed[width+(head-Int32(1))*dim+channel, token] : 0.0f0
    end
    qscale = inv(sqrt(warp_sum(sum(map(abs2, qs))) + 1.0f-6) * sqrt(Float32(dim)))
    kscale = inv(sqrt(warp_sum(sum(map(abs2, ks))) + 1.0f-6))
    for p = 1:PARTS
        channel = lane + Int32(32 * (p - 1))
        if channel <= dim
            query[channel, head, token] = qs[p] * qscale
            key[channel, head, token] = ks[p] * kscale
        end
    end
    return
end

# Each warp owns one value row; each lane owns PARTS key components of
# its recurrent state. No heap arrays or global state buffers inside the scan.
function delta_recurrent_kernel!(
    output,
    query,
    key,
    mixed,
    beta,
    decay,
    key_dim,
    value_dim,
    value_start,
    groups,
    tokens,
    ::Val{PARTS},
) where {PARTS}
    lane = Int32(CUDA.threadIdx().x)
    row =
        (Int32(CUDA.blockIdx().x) - Int32(1))*Int32(CUDA.blockDim().y) +
        Int32(CUDA.threadIdx().y)
    head = Int32(CUDA.blockIdx().y)
    key_head = cld(head, groups)
    state = ntuple(_ -> 0.0f0, Val(PARTS))
    @inbounds for token = Int32(1):tokens
        keys = ntuple(Val(PARTS)) do p
            channel = lane + Int32(32 * (p - 1))
            channel <= key_dim ? key[channel, key_head, token] : 0.0f0
        end
        queries = ntuple(Val(PARTS)) do p
            channel = lane + Int32(32 * (p - 1))
            channel <= key_dim ? query[channel, key_head, token] : 0.0f0
        end
        factor = exp(decay[head, token])
        state = map(s -> s * factor, state)
        prediction = warp_sum(sum(map(*, state, keys)))
        value =
            row <= value_dim ? mixed[value_start+(head-Int32(1))*value_dim+row, token] :
            0.0f0
        correction = (value - prediction) * beta[head, token]
        state = map((s, k) -> s + correction*k, state, keys)
        result = warp_sum(sum(map(*, state, queries)))
        if lane == 1 && row <= value_dim
            output[row, head, token] = result
        end
    end
    return
end

# Per (head, token) column: RMS-normalize the delta output (uncentered weight)
# and gate it with silu(z), where z is a row block of `projected`.
function rms_gate_kernel!(gated, output, weight, projected, z_offset, dim, heads, eps)
    lane = Int32(CUDA.threadIdx().x)
    column = Int32(CUDA.blockIdx().x)
    base = (column-Int32(1))*dim
    head = rem(column-Int32(1), heads)
    token = (column-Int32(1)) ÷ heads + Int32(1)
    total = 0.0f0
    @inbounds for row = lane:Int32(32):dim
        value = output[row+base]
        total += value*value
    end
    scale = inv(sqrt(warp_sum(total)/Float32(dim)+eps))
    @inbounds for row = lane:Int32(32):dim
        z = projected[z_offset+head*dim+row, token]
        gated[row+base] = output[row+base]*scale*weight[row]*QwenDecisionCore.native_silu(z)
    end
    return
end

const CONV_TOKENS = 8

# Thread = one channel over CONV_TOKENS consecutive tokens, so each input is
# loaded once per window instead of once per tap.
function causal_depthwise_kernel!(output, input, weight, channels, tokens, ::Val{TAPS}) where {TAPS}
    channel = (Int32(CUDA.blockIdx().x)-Int32(1))*Int32(CUDA.blockDim().x) + Int32(CUDA.threadIdx().x)
    first = (Int32(CUDA.blockIdx().y)-Int32(1))*Int32(CONV_TOKENS)
    channel <= channels || return
    @inbounds begin
        taps = ntuple(tap -> weight[tap, channel], Val(TAPS))
        window = ntuple(Val(TAPS - 1)) do lag
            source = first - Int32(TAPS - 1) + Int32(lag)
            source >= Int32(1) ? input[channel, source] : 0.0f0
        end
        for t = Int32(1):Int32(CONV_TOKENS)
            token = first + t
            token <= tokens || break
            current = input[channel, token]
            values = (window..., current)
            value = 0.0f0
            for tap = 1:TAPS
                value += values[tap] * taps[tap]
            end
            output[channel, token] = QwenDecisionCore.native_silu(value)
            window = Base.tail(values)
        end
    end
    return
end

function QwenDecisionCore.causal_depthwise(input::CUDA.StridedCuMatrix{Float32}, weight)
    output = scratch(Float32, size(input))
    # A leading row block is indexed through its parent (same leading
    # dimension); SubArray indexing roughly halves the kernel's speed.
    source =
        input isa SubArray && first(parentindices(input)[1]) == 1 &&
        parentindices(input)[2] == axes(parent(input), 2) ? parent(input) : input
    CUDA.@cuda threads=128 blocks=(cld(size(input, 1), 128), cld(size(input, 2), CONV_TOKENS)) causal_depthwise_kernel!(
        output,
        source,
        weight,
        Int32(size(input, 1)),
        Int32(size(input, 2)),
        Val(size(weight, 1)),
    )
    return output
end

function QwenDecisionCore.delta_attention(attention, x::CUDA.CuMatrix{Float32}, mask, cfg)
    cfg.key_dim <= 256 || return invoke(
        QwenDecisionCore.delta_attention,
        Tuple{Any,Any,Any,Any},
        attention,
        x,
        mask,
        cfg,
    )
    tokens = size(x, 2)
    device_mask = mask isa CUDA.CuArray ? mask : QwenDecisionCore.native_prepare_mask(x, mask)
    masked = scratch(Float32, size(x))
    masked .= x .* reshape(device_mask, 1, :)
    projected = QwenDecisionCore.native_linear(attention.packed_projection, masked)
    qkv_width = size(attention.qkv, 2)
    z_width = size(attention.z, 2)
    a_width = size(attention.a, 2)
    mixed = QwenDecisionCore.causal_depthwise(view(projected, 1:qkv_width, :), attention.conv)
    query = scratch(Float32, (cfg.key_dim, cfg.key_heads, tokens))
    key = scratch(Float32, size(query))
    parts = Val(cld(cfg.key_dim, 32))
    width = cfg.key_dim*cfg.key_heads
    CUDA.@cuda threads=32 blocks=cfg.key_heads*tokens qk_normalize_kernel!(
        query,
        key,
        mixed,
        Int32(cfg.key_dim),
        Int32(cfg.key_heads),
        Int32(width),
        Int32(tokens),
        parts,
    )
    b = view(projected, (qkv_width+z_width+a_width+1):size(projected, 1), :)
    beta = scratch(Float32, size(b))
    beta .= QwenDecisionCore.native_sigmoid.(b)
    a = view(projected, (qkv_width+z_width+1):(qkv_width+z_width+a_width), :)
    decay = scratch(Float32, size(a))
    decay .= attention.a_decay .* QwenDecisionCore.native_softplus.(a .+ attention.dt_bias)
    z = reshape(
        view(projected, (qkv_width+1):(qkv_width+z_width), :),
        cfg.value_dim,
        cfg.value_heads,
        tokens,
    )
    owner = get(task_local_storage(), WORKSPACE_KEY, nothing)
    if owner !== nothing && get(ENV, "QDC_CUDA_DELTA", "chunked") != "recurrent"
        chunks = cld(tokens, DELTA_CHUNK)
        padded = scratch(Float32, (cfg.value_dim, cfg.value_heads, chunks * DELTA_CHUNK))
        delta_chunked!(owner::ForwardWorkspace, padded, query, key, mixed, beta, decay, cfg, tokens)
        # A leading slice of the padded columns is itself a dense CuArray.
        output = view(padded, :, :, 1:tokens)
    else
        output = scratch(Float32, (cfg.value_dim, cfg.value_heads, tokens))
        launch_delta_recurrent!(output, query, key, mixed, beta, decay, cfg, width, tokens, parts)
    end
    gated = scratch(Float32, (cfg.value_dim*cfg.value_heads, tokens))
    CUDA.@cuda threads=32 blocks=cfg.value_heads*tokens rms_gate_kernel!(
        gated,
        output,
        attention.norm,
        projected,
        Int32(qkv_width),
        Int32(cfg.value_dim),
        Int32(cfg.value_heads),
        cfg.eps,
    )
    return QwenDecisionCore.native_linear(attention.out, gated)
end

function launch_delta_recurrent!(output, query, key, mixed, beta, decay, cfg, width, tokens, parts)
    CUDA.@cuda threads=(32, 4) blocks=(cld(cfg.value_dim, 4), cfg.value_heads) delta_recurrent_kernel!(
        output,
        query,
        key,
        mixed,
        beta,
        decay,
        Int32(cfg.key_dim),
        Int32(cfg.value_dim),
        Int32(2width),
        Int32(cfg.value_heads÷cfg.key_heads),
        Int32(tokens),
        parts,
    )
    return output
end
