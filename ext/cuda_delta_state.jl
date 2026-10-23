# Combine the two state reads into one wider GEMM:
#   [U, 0] - S * [W, -exp(G)Q] = [corrections, incoming output].
# The following intra-chunk product adds to the second column block. Packing
# all chunks once replaces one small batched GEMM per recurrent hand-off.
function delta_state_pack_kernel!(rhs, values, solved, queries, dk, dv, batches)
    index = (Int32(CUDA.blockIdx().x)-Int32(1))*Int32(CUDA.blockDim().x) + Int32(CUDA.threadIdx().x)
    C = Int32(DELTA_CHUNK)
    @inbounds begin
        if index <= dk*Int32(2)*C*batches
            row = rem(index-Int32(1), dk)+Int32(1)
            column = (index-Int32(1))÷dk
            token = rem(column, Int32(2)*C)+Int32(1)
            batch = column÷(Int32(2)*C)+Int32(1)
            rhs[index] = token <= C ? solved[row, token, batch] : -queries[row, token-C, batch]
        end
        if index <= dv*Int32(2)*C*batches
            row = rem(index-Int32(1), dv)+Int32(1)
            column = (index-Int32(1))÷dv
            token = rem(column, Int32(2)*C)+Int32(1)
            batch = column÷(Int32(2)*C)+Int32(1)
            values[index] = token <= C ? solved[dk+row, token, batch] : 0f0
        end
    end
    return
end

function delta_state_unpack_kernel!(output, values, dv, heads, elements)
    index = (Int32(CUDA.blockIdx().x)-Int32(1))*Int32(CUDA.blockDim().x) + Int32(CUDA.threadIdx().x)
    if index <= elements
        C = Int32(DELTA_CHUNK)
        row = rem(index-Int32(1), dv)+Int32(1)
        column = (index-Int32(1))÷dv
        head = rem(column, heads)+Int32(1)
        token = column÷heads
        batch = head+(token÷C)*heads
        @inbounds output[index] = values[row, C+rem(token, C)+Int32(1), batch]
    end
    return
end

function delta_chunked_state_packed!(owner, padded, solved, queries, intra, ending_keys, growths, cfg, chunks)
    C = DELTA_CHUNK
    dk, dv, heads = cfg.key_dim, cfg.value_dim, cfg.value_heads
    batches = heads*chunks
    rhs = scratch(Float32, (dk, 2C, batches))
    values = scratch(Float32, (dv, 2C, batches))
    CUDA.@cuda threads=256 blocks=cld(max(length(rhs), length(values)), 256) delta_state_pack_kernel!(
        rhs, values, solved, queries, Int32(dk), Int32(dv), Int32(batches))
    state = scratch(Float32, (dv, dk, heads))
    for chunk in 1:chunks
        base = (chunk-1)*heads
        offset = base*dv*2C
        if chunk > 1
            chunk_gemm!('N', 'N', dv, 2C, dk, owner.minus_one,
                state, 0, dv, dv*dk,
                rhs, base*dk*2C, dk, dk*2C,
                owner.one, values, offset, dv, dv*2C, heads)
        end
        chunk_gemm!('N', 'N', dv, C, C, owner.one,
            values, offset, dv, dv*2C,
            intra, base*C*C, C, C*C,
            owner.one, values, offset+dv*C, dv, dv*2C, heads)
        chunk == chunks && break
        if chunk > 1
            state .*= reshape(view(growths, (base+1):(base+heads)), 1, 1, heads)
        end
        chunk_gemm!('N', 'T', dv, dk, C, owner.one,
            values, offset, dv, dv*2C,
            ending_keys, base*dk*C, dk, dk*C,
            chunk > 1 ? owner.one : owner.zero,
            state, 0, dv, dv*dk, heads)
    end
    CUDA.@cuda threads=256 blocks=cld(length(padded), 256) delta_state_unpack_kernel!(
        padded, values, Int32(dv), Int32(heads), Int32(length(padded)))
    return padded
end
