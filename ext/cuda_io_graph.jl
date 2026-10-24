function gather_input_kernel!(hidden, mask, embedding, input, width, tokens, elements)
    index = (Int32(CUDA.blockIdx().x)-Int32(1))*Int32(CUDA.blockDim().x)+Int32(CUDA.threadIdx().x)
    if index <= elements
        row = rem(index-Int32(1),width)+Int32(1)
        token = (index-Int32(1))÷width+Int32(1)
        @inbounds begin
            hidden[index] = embedding[row,input[token]+Int32(1)]
            if row == Int32(1)
                mask[token] = input[tokens+token]
            end
        end
    end
    return
end

function cuda_io_forward!(b, input, output, device_input, hidden, mask)
    n = size(hidden,2)
    result = GC.@preserve input output begin
        unsafe_copyto!(pointer(device_input),pointer(input),length(input);async=true)
        CUDA.@cuda threads=256 blocks=cld(length(hidden),256) gather_input_kernel!(
            hidden,mask,b.embedding,device_input,Int32(size(hidden,1)),Int32(n),Int32(length(hidden)))
        result = cuda_hidden_forward(hidden,b.layers,mask,b.final_norm,b.config)
        unsafe_copyto!(pointer(output),pointer(result),length(output);async=true)
        result
    end
    return result
end

# Pinned staging buffers let a single graph include both transfers and the
# gather. Copy the completed host output while still holding the model lock.
function QwenDecisionCore.backbone_last_hidden(
    b::QwenDecisionCore.QwenBackbone{<:CUDA.CuMatrix{Float32}},
    ids::AbstractVector{<:Integer}, mask::AbstractVector{<:Integer},
)
    if get(ENV,"QDC_CUDA_GRAPH","1") == "0"
        return invoke(QwenDecisionCore.backbone_last_hidden,
            Tuple{QwenDecisionCore.QwenBackbone,AbstractVector{<:Integer},AbstractVector{<:Integer}},b,ids,mask)
    end
    QwenDecisionCore.check_sequence(b,ids,mask)
    return QwenDecisionCore.native_forward_scope(b.embedding,length(ids)) do
        owner = get(task_local_storage(),WORKSPACE_KEY,nothing)::ForwardWorkspace
        n, h = length(ids),size(b.embedding,1)
        device_input = scratch(Int32,(2n,))
        hidden = scratch(Float32,(h,n))
        device_mask = scratch(Int32,(n,))
        key = (:io,Tuple(objectid(layer) for layer in b.layers),objectid(device_input),
            objectid(hidden),objectid(device_mask),objectid(b.final_norm),b.config,
            CUDA.math_mode(),CUDA.math_precision(),get(ENV,"QDC_CUDA_DELTA","chunked"))
        cached = owner.forward_graph
        if cached !== nothing && hasproperty(cached,:input) && length(cached.input) == 2n && length(cached.output) == h
            input, output = cached.input,cached.output
        else
            input, output = Vector{Int32}(undef,2n),Vector{Float32}(undef,h)
            CUDA.pin(input)
            CUDA.pin(output)
        end
        @inbounds for i in 1:n
            input[i],input[n+i] = Int32(ids[i]),Int32(mask[i])
        end
        if cached === nothing || cached.key != key
            result = cuda_io_forward!(b,input,output,device_input,hidden,device_mask)
            cached = (;key,graph=nothing,result,input,output,cursor=owner.cursor,handle=CUDA.CUBLAS.handle())
            owner.forward_graph = cached
        else
            if cached.graph === nothing
                start = owner.cursor
                result = Ref{Any}()
                graph = CUDA.capture(;throw_error=false) do
                    result[] = cuda_io_forward!(b,input,output,device_input,hidden,device_mask)
                end
                if graph === nothing
                    owner.cursor = start
                    result[] = cuda_io_forward!(b,input,output,device_input,hidden,device_mask)
                else
                    cached = (;key,graph=CUDA.instantiate(graph),result=result[],input,output,
                        cursor=owner.cursor,handle=CUDA.CUBLAS.handle())
                    owner.forward_graph = cached
                    CUDA.launch(cached.graph)
                end
            else
                CUDA.launch(cached.graph)
                owner.cursor = cached.cursor
            end
        end
        CUDA.synchronize(owner.stream)
        return copy(output)
    end
end
