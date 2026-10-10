# Chunked gated delta rule (the WY form used by flash-linear-attention and the
# generic `delta_attention`). Within a chunk of DELTA_CHUNK tokens the
# recurrence becomes small batched products; only the state hand-off between
# chunks is sequential. All heads of a chunk run in one strided-batched cuBLAS
# call, and batch `b = head + (chunk - 1) * heads` throughout.
#
# Decays enter only as exp(G_t - G_s) with s <= t (or exp(G_t), G_t <= 0) for
# the cumulative log decay G inside a chunk, so no factor exceeds one.
const DELTA_CHUNK = 64

# Column-major strided-batched SGEMM on raw offsets, so that operands can be row
# blocks of a packed buffer and results can be written with any leading
# dimension. Coefficients are the workspace's device scalars.
function chunk_gemm!(
    ta,
    tb,
    m,
    n,
    k,
    alpha,
    a,
    offa,
    lda,
    sa,
    b,
    offb,
    ldb,
    sb,
    beta,
    c,
    offc,
    ldc,
    sc,
    batch,
)
    blas = CUDA.CUBLAS
    blas.cublasGemmStridedBatchedEx(
        blas.handle(),
        ta,
        tb,
        m,
        n,
        k,
        alpha,
        pointer(a, offa + 1),
        Float32,
        lda,
        sa,
        pointer(b, offb + 1),
        Float32,
        ldb,
        sb,
        beta,
        pointer(c, offc + 1),
        Float32,
        ldc,
        sc,
        batch,
        blas.CUBLAS_COMPUTE_32F,
        blas.CUBLAS_GEMM_DEFAULT,
    )
    return c
end

const DELTA_BLOCK = 16
const DELTA_PARTS = 4

# Inclusive prefix sum of the chunk's log decays into shared `logs` (C threads
# or more); `betas` receives the gates. Tokens past the sequence get zeros.
@inline function chunk_logs!(logs, betas, beta, decay, head, first, tokens, thread)
    C = Int32(DELTA_CHUNK)
    @inbounds if thread <= C
        token = first + thread
        valid = token <= tokens
        logs[thread] = valid ? decay[head, token] : 0.0f0
        betas[thread] = valid ? beta[head, token] : 0.0f0
    end
    offset = Int32(1)
    while offset < C
        CUDA.sync_threads()
        value = 0.0f0
        @inbounds if thread <= C && thread > offset
            value = logs[thread-offset]
        end
        CUDA.sync_threads()
        @inbounds if thread <= C && thread > offset
            logs[thread] += value
        end
        offset *= Int32(2)
    end
    CUDA.sync_threads()
    return
end

# Blocks (head, chunk) x DELTA_PARTS token ranges. Writes the chunk operands:
#   keys_queries[:, 1:C] = k, [:, C+1:2C] = q
#   scaled[1:dk] = beta * exp(G) * k, scaled[dk+1:end] = beta * v
#   decayed_queries = exp(G) * q, ending_keys = exp(G_C - G) * k
# Tokens past the sequence are zero with zero decay, so they leave the state
# unchanged and produce zero output.
function delta_chunk_prepare_kernel!(
    keys_queries,
    scaled,
    decayed_queries,
    ending_keys,
    cumulative,
    gates,
    growth_out,
    query,
    key,
    mixed,
    beta,
    decay,
    key_dim,
    value_dim,
    value_start,
    heads,
    groups,
    tokens,
)
    C = Int32(DELTA_CHUNK)
    thread = Int32(CUDA.threadIdx().x)
    threads = Int32(CUDA.blockDim().x)
    batch = Int32(CUDA.blockIdx().x)
    part = Int32(CUDA.blockIdx().y)
    head = rem(batch - Int32(1), heads) + Int32(1)
    first = (batch - Int32(1)) ÷ heads * C
    key_head = cld(head, groups)
    logs = CUDA.CuStaticSharedArray(Float32, DELTA_CHUNK)
    betas = CUDA.CuStaticSharedArray(Float32, DELTA_CHUNK)
    chunk_logs!(logs, betas, beta, decay, head, first, tokens, thread)
    @inbounds begin
        last_log = logs[C]
        if part == Int32(1) && thread <= C
            cumulative[thread, batch] = logs[thread]
            gates[thread, batch] = betas[thread]
            thread == C && (growth_out[batch] = exp(last_log))
        end
        span = C ÷ Int32(DELTA_PARTS)
        for t = ((part-Int32(1))*span+Int32(1)):(part*span)
            token = first + t
            valid = token <= tokens
            growth = exp(logs[t])
            ending = exp(last_log - logs[t])
            for channel = thread:threads:key_dim
                k = valid ? key[channel, key_head, token] : 0.0f0
                q = valid ? query[channel, key_head, token] : 0.0f0
                keys_queries[channel, t, batch] = k
                keys_queries[channel, C+t, batch] = q
                decayed_queries[channel, t, batch] = q * growth
                ending_keys[channel, t, batch] = k * ending
                scaled[channel, t, batch] = k * betas[t] * growth
            end
            for channel = thread:threads:value_dim
                v =
                    valid ? mixed[value_start+(head-Int32(1))*value_dim+channel, token] :
                    0.0f0
                scaled[key_dim+channel, t, batch] = v * betas[t]
            end
        end
    end
    return
end

# One block of 4C threads per (head, chunk). From products = [k'k  k'q]:
#   inverse = (I + L)^-1 with L[t, s] = beta_t exp(G_t - G_s) k_t'k_s (s < t)
#   intra[s, t] = exp(G_t - G_s) k_s'q_t (s <= t), zero otherwise.
# The inverse is blocked: diagonal DELTA_BLOCK blocks by forward substitution,
# then each block row as T_ij = -T_ii * sum_k L_ik T_kj.
function delta_chunk_inverse_kernel!(inverse, intra, products, cumulative, gates)
    C = Int32(DELTA_CHUNK)
    S = Int32(DELTA_BLOCK)
    thread = Int32(CUDA.threadIdx().x)
    threads = Int32(CUDA.blockDim().x)
    batch = Int32(CUDA.blockIdx().x)
    system = CUDA.CuStaticSharedArray(Float32, (DELTA_CHUNK + 1, DELTA_CHUNK))
    solved = CUDA.CuStaticSharedArray(Float32, (DELTA_CHUNK + 1, DELTA_CHUNK))
    partial = CUDA.CuStaticSharedArray(Float32, (DELTA_BLOCK + 1, DELTA_CHUNK))
    logs = CUDA.CuStaticSharedArray(Float32, DELTA_CHUNK)
    betas = CUDA.CuStaticSharedArray(Float32, DELTA_CHUNK)
    @inbounds begin
        if thread <= C
            logs[thread] = cumulative[thread, batch]
            betas[thread] = gates[thread, batch]
        end
        CUDA.sync_threads()
        for e = thread:threads:(C*C)
            i = rem(e - Int32(1), C) + Int32(1)
            s = (e - Int32(1)) ÷ C + Int32(1)
            system[i, s] =
                s < i ? betas[i] * exp(logs[i] - logs[s]) * products[i, s, batch] : 0.0f0
            intra[i, s, batch] =
                s >= i ? exp(logs[s] - logs[i]) * products[i, C+s, batch] : 0.0f0
            solved[i, s] = i == s ? 1.0f0 : 0.0f0
        end
        CUDA.sync_threads()
        if thread <= C
            j = thread
            stop = ((j - Int32(1)) ÷ S + Int32(1)) * S
            for t = (j+Int32(1)):stop
                total = 0.0f0
                for s = j:(t-Int32(1))
                    total += system[t, s] * solved[s, j]
                end
                solved[t, j] = -total
            end
        end
        for row_block = Int32(1):(C÷S-Int32(1))
            start = row_block * S          # rows start+1:start+S, columns 1:start
            CUDA.sync_threads()
            for e = thread:threads:(S*start)
                r = rem(e - Int32(1), S) + Int32(1)
                j = (e - Int32(1)) ÷ S + Int32(1)
                total = 0.0f0
                for s = j:start
                    total += system[start+r, s] * solved[s, j]
                end
                partial[r, j] = total
            end
            CUDA.sync_threads()
            for e = thread:threads:(S*start)
                r = rem(e - Int32(1), S) + Int32(1)
                j = (e - Int32(1)) ÷ S + Int32(1)
                total = 0.0f0
                for u = Int32(1):r
                    total += solved[start+r, start+u] * partial[u, j]
                end
                solved[start+r, j] = -total
            end
        end
        CUDA.sync_threads()
        for e = thread:threads:(C*C)
            i = rem(e - Int32(1), C) + Int32(1)
            s = (e - Int32(1)) ÷ C + Int32(1)
            inverse[i, s, batch] = solved[i, s]
        end
    end
    return
end

function delta_chunked!(owner, padded, query, key, mixed, beta, decay, cfg, tokens)
    C = DELTA_CHUNK
    dk, dv, heads = cfg.key_dim, cfg.value_dim, cfg.value_heads
    chunks = cld(tokens, C)
    batches = heads * chunks
    width = dk + dv
    keys_queries = scratch(Float32, (dk, 2C, batches))
    scaled = scratch(Float32, (width, C, batches))
    decayed_queries = scratch(Float32, (dk, C, batches))
    ending_keys = scratch(Float32, (dk, C, batches))
    cumulative = scratch(Float32, (C, batches))
    gates = scratch(Float32, (C, batches))
    growths = scratch(Float32, (batches,))
    CUDA.@cuda threads=128 blocks=(batches, DELTA_PARTS) delta_chunk_prepare_kernel!(
        keys_queries,
        scaled,
        decayed_queries,
        ending_keys,
        cumulative,
        gates,
        growths,
        query,
        key,
        mixed,
        beta,
        decay,
        Int32(dk),
        Int32(dv),
        Int32(2 * dk * cfg.key_heads),
        Int32(heads),
        Int32(heads ÷ cfg.key_heads),
        Int32(tokens),
    )
    # products[:, 1:C] = k'k, products[:, C+1:2C] = k'q
    products = scratch(Float32, (C, 2C, batches))
    chunk_gemm!(
        'T',
        'N',
        C,
        2C,
        dk,
        owner.one,
        keys_queries,
        0,
        dk,
        dk * 2C,
        keys_queries,
        0,
        dk,
        dk * 2C,
        owner.zero,
        products,
        0,
        C,
        C * 2C,
        batches,
    )
    inverse = scratch(Float32, (C, C, batches))
    intra = scratch(Float32, (C, C, batches))
    CUDA.@cuda threads=4C blocks=batches delta_chunk_inverse_kernel!(
        inverse,
        intra,
        products,
        cumulative,
        gates,
    )
    # solved[1:dk] = W' (reading keys), solved[dk+1:end] = U' (new values)
    solved = scratch(Float32, (width, C, batches))
    chunk_gemm!(
        'N',
        'T',
        width,
        C,
        C,
        owner.one,
        scaled,
        0,
        width,
        width * C,
        inverse,
        0,
        C,
        C * C,
        owner.zero,
        solved,
        0,
        width,
        width * C,
        batches,
    )
    state = scratch(Float32, (dv, dk, heads))
    for chunk = 1:chunks
        base = (chunk - 1) * heads
        values = (base * width * C) + dk          # U' rows of this chunk
        keys = base * width * C                   # W' rows
        out = (chunk - 1) * C * dv * heads        # padded[:, :, first token]
        if chunk > 1
            # corrections = U' - S W'
            chunk_gemm!(
                'N',
                'N',
                dv,
                C,
                dk,
                owner.minus_one,
                state,
                0,
                dv,
                dv * dk,
                solved,
                keys,
                width,
                width * C,
                owner.one,
                solved,
                values,
                width,
                width * C,
                heads,
            )
            chunk_gemm!(
                'N',
                'N',
                dv,
                C,
                dk,
                owner.one,
                state,
                0,
                dv,
                dv * dk,
                decayed_queries,
                base * dk * C,
                dk,
                dk * C,
                owner.zero,
                padded,
                out,
                dv * heads,
                dv,
                heads,
            )
        end
        chunk_gemm!(
            'N',
            'N',
            dv,
            C,
            C,
            owner.one,
            solved,
            values,
            width,
            width * C,
            intra,
            base * C * C,
            C,
            C * C,
            chunk > 1 ? owner.one : owner.zero,
            padded,
            out,
            dv * heads,
            dv,
            heads,
        )
        chunk == chunks && break
        if chunk > 1
            state .*= reshape(view(growths, (base+1):(base+heads)), 1, 1, heads)
        end
        chunk_gemm!(
            'N',
            'T',
            dv,
            dk,
            C,
            owner.one,
            solved,
            values,
            width,
            width * C,
            ending_keys,
            base * dk * C,
            dk,
            dk * C,
            chunk > 1 ? owner.one : owner.zero,
            state,
            0,
            dv,
            dv * dk,
            heads,
        )
    end
    return padded
end
