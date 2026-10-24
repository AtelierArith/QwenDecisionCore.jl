# Load an operand transposed into a padded shared tile, coalescing global reads.
@inline function load_state_operand!(rhs, source, rows, batch, thread)
    @inbounds for first in thread:Int32(512):(rows*Int32(DELTA_CHUNK))
        values = ntuple(Val(4)) do j
            source_index = first+Int32(128*(j-1))
            source_row = rem(source_index-Int32(1),rows)+Int32(1)
            source_col = (source_index-Int32(1))÷rows+Int32(1)
            @inbounds source[source_row,source_col,batch]
        end
        Base.Cartesian.@nexprs 4 j -> begin
            e = first+Int32(128*(j-1))
            rr = rem(e-Int32(1),rows)+Int32(1)
            cc = (e-Int32(1))÷rows+Int32(1)
            rhs[cc,rr] = values[j]
        end
    end
    return nothing
end

# Keep only the recurrent hand-off in the persistent kernel. Readout is a
# single batched GEMM over all chunks, with enough batches to fill the device.
function delta_state_history_kernel!(combined, solved, ending, growths,
    heads, chunks, ::Val{K}) where {K}
    row, group = Int32(CUDA.threadIdx().x), Int32(CUDA.threadIdx().y)
    rowbase = (Int32(CUDA.blockIdx().x)-Int32(1))*Int32(16)
    head = Int32(CUDA.blockIdx().y)
    thread = row+(group-Int32(1))*Int32(4)
    C = Int32(DELTA_CHUNK)
    state = CUDA.CuStaticSharedArray(Float32,(17,K))
    corrected = CUDA.CuStaticSharedArray(Float32,(17,DELTA_CHUNK))
    rhs = CUDA.CuStaticSharedArray(Float32,(DELTA_CHUNK+1,K))
    @inbounds begin
        Base.Cartesian.@nexprs 16 p -> begin
            if 8p <= K
                rr = row+Int32(4*((p-1)÷(K÷32)))
                key = group+Int32(32*rem(p-1,K÷32))
                combined[rowbase+rr,key,head] = 0f0
            end
        end
        for chunk in Int32(1):chunks
            batch = head+(chunk-Int32(1))*heads
            predicted = ntuple(_ -> 0f0,Val(8))
            if chunk > Int32(1)
                load_state_operand!(rhs,solved,Int32(K),batch,thread)
                CUDA.sync_threads()
                for key in Int32(1):Int32(K)
                    av = ntuple(r -> state[row+Int32(4*(r-1)),key],Val(4))
                    bv = ntuple(j -> rhs[group+Int32(32*(j-1)),key],Val(2))
                    predicted = let previous=predicted, av=av, bv=bv
                        ntuple(p -> muladd(av[(p-1)÷2+1],bv[rem(p-1,2)+1],previous[p]),Val(8))
                    end
                end
            end
            Base.Cartesian.@nexprs 8 p -> begin
                rr = row+Int32(4*((p-1)÷2))
                token = group+Int32(32*rem(p-1,2))
                value = solved[Int32(K)+rowbase+rr,token,batch]-predicted[p]
                corrected[rr,token] = value
                combined[rowbase+rr,Int32(K)+token,batch] = value
            end
            chunk == chunks && break
            CUDA.sync_threads()
            load_state_operand!(rhs,ending,Int32(K),batch,thread)
            CUDA.sync_threads()
            updated = ntuple(_ -> 0f0,Val(K÷8))
            for token in Int32(1):C
                av = ntuple(r -> corrected[row+Int32(4*(r-1)),token],Val(4))
                bv = ntuple(j -> rhs[token,group+Int32(32*(j-1))],Val(K÷32))
                updated = let previous=updated, av=av, bv=bv
                    ntuple(p -> muladd(av[(p-1)÷(K÷32)+1],bv[rem(p-1,K÷32)+1],previous[p]),Val(K÷8))
                end
            end
            factor = growths[batch]
            Base.Cartesian.@nexprs 16 p -> begin
                if 8p <= K
                    rr = row+Int32(4*((p-1)÷(K÷32)))
                    key = group+Int32(32*rem(p-1,K÷32))
                    old = chunk == Int32(1) ? 0f0 : state[rr,key]*factor
                    value = old+updated[p]
                    state[rr,key] = value
                    combined[rowbase+rr,key,batch+heads] = value
                end
            end
            CUDA.sync_threads()
        end
    end
    return
end

function delta_readout_unpack_kernel!(output, values, dv, heads, elements)
    index = (Int32(CUDA.blockIdx().x)-Int32(1))*Int32(CUDA.blockDim().x)+Int32(CUDA.threadIdx().x)
    if index <= elements
        C = Int32(DELTA_CHUNK)
        row = rem(index-Int32(1),dv)+Int32(1)
        col = (index-Int32(1))÷dv
        head, token = rem(col,heads)+Int32(1), col÷heads
        @inbounds output[index] = values[row,rem(token,C)+Int32(1),head+(token÷C)*heads]
    end
    return
end

function delta_chunked_hybrid!(owner, padded, solved, rhs, ending, growths, cfg, chunks)
    chunks >= 2 && cfg.key_dim in (64,128) && cfg.value_dim % 16 == 0 || return false
    dk,dv,heads = cfg.key_dim,cfg.value_dim,cfg.value_heads
    C = DELTA_CHUNK
    batches = heads*chunks
    combined = scratch(Float32,(dv,dk+C,batches))
    values = scratch(Float32,(dv,C,batches))
    CUDA.@cuda threads=(4,32) maxthreads=128 always_inline=true blocks=(dv÷16,heads) delta_state_history_kernel!(
        combined,solved,ending,growths,Int32(heads),Int32(chunks),Val(dk))
    chunk_gemm!('N','N',dv,C,dk+C,owner.one,
        combined,0,dv,dv*(dk+C),rhs,0,dk+C,(dk+C)*C,
        owner.zero,values,0,dv,dv*C,batches)
    CUDA.@cuda threads=256 blocks=cld(length(padded),256) delta_readout_unpack_kernel!(
        padded,values,Int32(dv),Int32(heads),Int32(length(padded)))
    return true
end
