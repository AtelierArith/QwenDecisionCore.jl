# AMDGPU counterpart of cuda_workspace.jl: a model's embedding identifies its
# private scratch, same-backend forwards are serialized, and scratch is never
# handed to another forward before its stream has completed. Unlike CUDA,
# rocBLAS alpha/beta coefficients are host scalars, so the workspace carries no
# device one/zero vectors.

const ROCBuffer = AMDGPU.Runtime.Mem.HIPBuffer
const ROCMatrixF32 = AMDGPU.ROCArray{Float32,2,ROCBuffer}

mutable struct ForwardWorkspace
    lock::ReentrantLock
    stream::AMDGPU.HIP.HIPStream
    slots::Vector{Any}
    layer_slots::Dict{Symbol,Vector{Any}}
    active_slots::Vector{Any}
    cursor::Int
end

const WORKSPACES = Dict{UInt,Tuple{WeakRef,ForwardWorkspace}}()
const WORKSPACE_LOCK = ReentrantLock()
const WORKSPACE_KEY = :qdc_native_amdgpu_workspace

function workspace(reference)
    @lock WORKSPACE_LOCK begin
        for key in collect(keys(WORKSPACES))
            WORKSPACES[key][1].value === nothing && delete!(WORKSPACES, key)
        end
        key = objectid(reference)
        entry = get(WORKSPACES, key, nothing)
        if entry === nothing || entry[1].value !== reference
            slots = Any[]
            owner = ForwardWorkspace(
                ReentrantLock(),
                AMDGPU.stream(),
                slots,
                Dict{Symbol,Vector{Any}}(),
                slots,
                0,
            )
            WORKSPACES[key] = (WeakRef(reference), owner)
            return owner
        end
        return entry[2]
    end
end

function QwenDecisionCore.native_forward_scope(f::F, reference::AMDGPU.ROCArray) where {F}
    if AMDGPU.device() != AMDGPU.device(reference)
        return AMDGPU.device!(AMDGPU.device(reference)) do
            QwenDecisionCore.native_forward_scope(f, reference)
        end
    end
    owner = workspace(reference)
    lock(owner.lock) do
        AMDGPU.synchronize(owner.stream)
        owner.stream = AMDGPU.stream()
        owner.cursor = 0
        owner.active_slots = owner.slots
        task_local_storage(WORKSPACE_KEY, owner) do
            try
                return f()
            finally
                # Covers failed forwards as well as ordinary host readback.
                AMDGPU.synchronize(owner.stream)
            end
        end
    end
end

function scratch(::Type{T}, dims::NTuple{N,Int}) where {T,N}
    owner = get(task_local_storage(), WORKSPACE_KEY, nothing)
    owner === nothing && return AMDGPU.ROCArray{T,N,ROCBuffer}(undef, dims)
    owner = owner::ForwardWorkspace
    owner.cursor += 1
    slot = owner.cursor
    slots = owner.active_slots
    if slot > length(slots)
        push!(slots, AMDGPU.ROCArray{T,N,ROCBuffer}(undef, dims))
    else
        array = slots[slot]
        if !(array isa AMDGPU.ROCArray{T,N,ROCBuffer}) || size(array) != dims
            slots[slot] = AMDGPU.ROCArray{T,N,ROCBuffer}(undef, dims)
        end
    end
    return slots[slot]::AMDGPU.ROCArray{T,N,ROCBuffer}
end

const DeviceMatrix = Union{
    AMDGPU.StridedROCMatrix{Float32},
    Transpose{Float32,<:AMDGPU.StridedROCMatrix{Float32}},
}

function QwenDecisionCore.native_matmul(a::DeviceMatrix, b::DeviceMatrix)
    size(a, 2) == size(b, 1) ||
        throw(DimensionMismatch("Matrix inner dimensions must match."))
    output = scratch(Float32, (size(a, 1), size(b, 2)))
    # rocBLAS reads the destination only when beta != 0, which the 3-argument
    # mul! selects, so stale scratch contents never leak into the product.
    return mul!(output, a, b)
end

function batched_matmul!(output, a, b, ta, tb)
    return AMDGPU.rocBLAS.gemm_strided_batched!(ta, tb, 1.0f0, a, b, 0.0f0, output)
end

@inline function warp_sum(value::Float32)
    for offset in (16, 8, 4, 2, 1)
        value += AMDGPU.Device.shfl_xor_sync(UInt64(0xffffffff), value, Cint(offset))
    end
    return value
end

function rms_kernel!(output, input, weight, width, columns, eps, centered)
    lane = Int32(AMDGPU.threadIdx().x)
    column = Int32(AMDGPU.blockIdx().x)
    total = 0.0f0
    for row = lane:Int32(32):width
        value = input[row+(column-Int32(1))*width]
        total += value*value
    end
    scale = inv(sqrt(warp_sum(total)/Float32(width)+eps))
    for row = lane:Int32(32):width
        offset = row+(column-Int32(1))*width
        gamma = centered ? 1.0f0+weight[row] : weight[row]
        output[offset] = input[offset]*scale*gamma
    end
    return
end

function QwenDecisionCore.native_rms(
    input::AMDGPU.ROCArray{Float32},
    weight,
    eps;
    centered = true,
)
    output = scratch(Float32, size(input))
    width = size(input, 1)
    columns = length(input)÷width
    AMDGPU.@roc groupsize = 32 gridsize = columns rms_kernel!(
        output,
        input,
        weight,
        Int32(width),
        Int32(columns),
        eps,
        centered,
    )
    return output
end

function QwenDecisionCore.native_prepare_mask(reference::AMDGPU.ROCArray, mask)
    output = scratch(Int32, (length(mask),))
    copyto!(output, Int32.(mask))
    return output
end

function QwenDecisionCore.native_residual_rms(
    x::AMDGPU.ROCArray{Float32,2,ROCBuffer},
    mixed,
    weight,
    eps,
)
    residual = scratch(Float32, size(x))
    residual .= x .+ mixed
    return residual, QwenDecisionCore.native_rms(residual, weight, eps)
end

function QwenDecisionCore.native_hidden_forward(
    hidden::AMDGPU.ROCArray{Float32},
    layers,
    mask,
    final_norm,
    cfg,
)
    owner = get(task_local_storage(), WORKSPACE_KEY, nothing)
    if owner === nothing
        for layer in layers
            hidden = QwenDecisionCore.native_layer(layer, hidden, mask, cfg)
        end
    else
        owner = owner::ForwardWorkspace
        buffers = (scratch(Float32, size(hidden)), scratch(Float32, size(hidden)))
        root_slots, root_cursor = owner.active_slots, owner.cursor
        for (index, layer) in enumerate(layers)
            owner.active_slots = get!(()->Any[], owner.layer_slots, layer.attention.kind)
            owner.cursor = 0
            destination = buffers[isodd(index) ? 1 : 2]
            try
                result = QwenDecisionCore.native_layer(layer, hidden, mask, cfg)
                # Preserve the next layer's input outside the reused layer
                # scratch. The copy and subsequent writes share one stream.
                copyto!(destination, result)
            finally
                owner.active_slots, owner.cursor = root_slots, root_cursor
            end
            hidden = destination
        end
    end
    return QwenDecisionCore.native_rms(
        view(hidden, :, size(hidden, 2):size(hidden, 2)),
        final_norm,
        cfg.eps,
    )
end

function gather_kernel!(output, embedding, ids, width, elements)
    index =
        (Int32(AMDGPU.blockIdx().x)-Int32(1))*Int32(AMDGPU.blockDim().x)+Int32(
            AMDGPU.threadIdx().x,
        )
    if index <= elements
        row = rem(index-Int32(1), width)+Int32(1)
        token = (index-Int32(1))÷width+Int32(1)
        @inbounds output[index] = embedding[row, ids[token]+Int32(1)]
    end
    return
end

function QwenDecisionCore.native_gather(
    embedding::AMDGPU.ROCArray{Float32},
    ids::Vector{<:Integer},
)
    all(id -> 0 <= id < size(embedding, 2), ids) ||
        throw(ArgumentError("Token ID is outside the vocabulary."))
    device_ids = scratch(Int32, (length(ids),))
    copyto!(device_ids, Int32.(ids))
    output = scratch(Float32, (size(embedding, 1), length(ids)))
    AMDGPU.@roc groupsize = 256 gridsize = cld(length(output), 256) gather_kernel!(
        output,
        embedding,
        device_ids,
        Int32(size(embedding, 1)),
        Int32(length(output)),
    )
    return output
end
