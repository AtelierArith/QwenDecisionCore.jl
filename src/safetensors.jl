# Safetensors tensors use row-major storage. Reverse axes when loading so
# Julia's column-major arrays share the same feature/token ordering.
function read_native_weights(path; select = Returns(true), convert_array = identity)
    tensors = Dict{String,Any}()
    open(path, "r") do io
        header_length = Int(ltoh(read(io, UInt64)))
        0 < header_length <= filesize(path) - 8 ||
            throw(ArgumentError("Invalid safetensors header."))
        header = JSON.parse(String(read(io, header_length)))
        base = 8 + header_length
        for (name, spec) in header
            name == "__metadata__" && continue
            select(name) || continue
            shape = Tuple(reverse(Int.(spec["shape"])))
            start, stop = Int.(spec["data_offsets"])
            0 <= start <= stop <= filesize(path) - base ||
                throw(ArgumentError("Invalid offsets for tensor $name."))
            dtype = spec["dtype"]
            T =
                dtype == "BF16" ? UInt16 :
                dtype == "F16" ? Float16 :
                dtype == "F32" ? Float32 :
                throw(ArgumentError("Native backend does not support tensor dtype $dtype."))
            stop - start == prod(shape) * sizeof(T) ||
                throw(ArgumentError("Invalid size for tensor $name."))
            seek(io, base + start)
            raw = read!(io, Vector{T}(undef, prod(shape)))
            values =
                dtype == "BF16" ? reinterpret(Float32, UInt32.(ltoh.(raw)) .<< 16) :
                Float32.(ltoh.(raw))
            tensors[String(name)] = convert_array(reshape(values, shape))
        end
    end
    return tensors
end

# Round a Float32 to bfloat16 (round-to-nearest-even on the 16 dropped bits).
function float32_to_bf16(value::Float32)
    bits = reinterpret(UInt32, value)
    rounded = bits + 0x00007fff + ((bits >> 16) & 0x1)
    return UInt16(rounded >> 16)
end

safetensors_bytes(A::Array{Float32}) = (collect(reinterpret(UInt8, vec(A))), "F32")
safetensors_bytes(A::Array{Float16}) = (collect(reinterpret(UInt8, vec(A))), "F16")
safetensors_bytes(A::Array{UInt16}) = (collect(reinterpret(UInt8, vec(A))), "BF16")
safetensors_bytes(A) = throw(ArgumentError("Unsupported tensor eltype $(eltype(A))"))

"""
    write_native_weights(path, tensors; metadata=nothing)

Write `tensors` (a name => array map) as safetensors. Arrays are stored the
same way `read_native_weights` reads them, so a round trip is exact; a PyTorch
`Linear` weight written from Julia's (in, out) layout is stored with the
reversed shape and read back by torch as (out, in).
"""
function write_native_weights(path::AbstractString, tensors::AbstractDict; metadata = nothing)
    header = Dict{String,Any}()
    data = IOBuffer()
    offset = 0
    for name in sort!(collect(keys(tensors)))
        bytes, dtype = safetensors_bytes(tensors[name])
        header[String(name)] = Dict(
            "dtype" => dtype,
            "shape" => collect(reverse(Int.(size(tensors[name])))),
            "data_offsets" => [offset, offset + length(bytes)],
        )
        write(data, bytes)
        offset += length(bytes)
    end
    metadata === nothing || (header["__metadata__"] = metadata)
    payload = take!(data)
    header_bytes = Vector{UInt8}(codeunits(JSON.json(header)))
    open(path, "w") do io
        write(io, htol(UInt64(length(header_bytes))))
        write(io, header_bytes)
        write(io, payload)
    end
    return path
end
