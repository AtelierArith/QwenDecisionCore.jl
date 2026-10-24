# A private, lazily loaded SGEMM does not redirect libblastrampoline or change
# another package's BLAS. The explicit _64 entry point has a stable integer ABI
# independently of MKL_INTERFACE_LAYER. Its thread-local limit is restored.
const CPU_MKL_GEMM = Ref{Ptr{Cvoid}}(C_NULL)
const CPU_MKL_THREADS = Ref{Ptr{Cvoid}}(C_NULL)
const CPU_MKL_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)
const CPU_MKL_ERROR = Ref{Any}(nothing)
const CPU_MKL_LOCK = ReentrantLock()

function reset_cpu_mkl!()
    CPU_MKL_GEMM[] = C_NULL
    CPU_MKL_THREADS[] = C_NULL
    CPU_MKL_HANDLE[] = C_NULL
    CPU_MKL_ERROR[] = nothing
    return nothing
end

function load_cpu_mkl!()
    CPU_MKL_GEMM[] != C_NULL && return true
    CPU_MKL_ERROR[] === nothing || return false
    lock(CPU_MKL_LOCK) do
        CPU_MKL_GEMM[] != C_NULL && return true
        CPU_MKL_ERROR[] === nothing || return false
        try
            library = Base.require(Base.PkgId(Base.UUID("856f044c-d86e-5d09-b602-aeab76dc8ba7"), "MKL_jll"))
            Base.invokelatest(getproperty(library, :is_available)) || error("MKL is unavailable on this platform.")
            handle = Libdl.dlopen(getproperty(library, :libmkl_rt))
            gemm = Libdl.dlsym(handle, :cblas_sgemm_64)
            threads = Libdl.dlsym(handle, :MKL_Set_Num_Threads_Local)
            CPU_MKL_HANDLE[] = handle
            CPU_MKL_THREADS[] = threads
            CPU_MKL_GEMM[] = gemm
            return true
        catch failure
            failure isa InterruptException && rethrow()
            CPU_MKL_ERROR[] = failure
            return false
        end
    end
end

cpu_mkl_projection!(output, weight, input, beta) = false
function cpu_mkl_projection!(output::Matrix{Float32}, weight::AbstractMatrix{Float32}, input::AbstractMatrix{Float32}, beta)
    cpu_setting(:mkl_serial) && Threads.nthreads(:default) == 1 &&
        cpu_projection_blas_threads() == 1 || return false
    source = weight isa Transpose ? parent(weight) : weight
    source isa StridedMatrix{Float32} && input isa StridedMatrix{Float32} &&
        stride(source,1) == 1 && stride(input,1) == 1 &&
        stride(source,2) > 0 && stride(input,2) > 0 &&
        min(size(output,1),size(output,2),size(input,1)) > 0 || return false
    load_cpu_mkl!() || return false
    previous = ccall(CPU_MKL_THREADS[], Cint, (Cint,), 1)
    try
        GC.@preserve output weight input source begin
            ccall(CPU_MKL_GEMM[], Cvoid,
                (Cint,Cint,Cint,Int64,Int64,Int64,Cfloat,Ptr{Float32},Int64,Ptr{Float32},Int64,Cfloat,Ptr{Float32},Int64),
                102, weight isa Transpose ? 111 : 112, 111,
                size(output,1),size(output,2),size(input,1),1f0,
                pointer(source),stride(source,2),pointer(input),stride(input,2),
                Float32(beta),pointer(output),stride(output,2))
        end
    finally
        ccall(CPU_MKL_THREADS[], Cint, (Cint,), previous)
    end
    return true
end
