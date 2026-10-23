# Compare the same public API across two revisions. See benchmark/README.md.
using QwenDecisionCore, Random, Statistics, LinearAlgebra, JSON
const QDC = QwenDecisionCore
include("synthetic.jl")
option(name, default) = (i = findfirst(==(name), ARGS); i === nothing ? default : ARGS[i+1])
device = Symbol(option("--device", "cpu"))
samples = parse(Int, option("--samples", "21"))
samples > 0 || error("--samples must be positive")
Threads.nthreads(:default) == 1 || error("Run with julia -t 1")
BLAS.set_num_threads(1)
if device == :cuda
    @eval using CUDA
    CUDA.allowscalar(false)
elseif device != :cpu
    error("Expected --device cpu or cuda")
end
checkpoint = option("--checkpoint", "")
b = isempty(checkpoint) ? synthetic_backbone() : QwenBackbone(checkpoint; device)
isempty(checkpoint) && device != :cpu && (b = device_backbone(b, device))
operation = "--all" in ARGS ? backbone_hidden : backbone_last_hidden
baseline_path = option("--baseline", "")
baseline = isempty(baseline_path) ? nothing : JSON.parsefile(baseline_path)
"--require-speedup" in ARGS && baseline === nothing && error("--require-speedup needs --baseline")
results = []
println("Julia=$VERSION; Julia threads=1; BLAS threads=$(BLAS.get_num_threads()); device=$device; checkpoint=$checkpoint")
flush(stdout)
for n in (64, 128, 256, 512)
    ids = mod.(collect(0:(n-1)), min(256, size(b.embedding, 2)))
    mask = ones(Int, n)
    f() = operation(b, ids, mask)
    result = f()
    for _ in 1:3; f(); end
    if baseline !== nothing
        expected = Float32.(baseline["results"][length(results)+1]["output"])
        @assert baseline["results"][length(results)+1]["tokens"] == n
        @assert isapprox(vec(result), expected; atol=2e-5, rtol=2e-5)
        println((; tokens=n, max_error=maximum(abs.(vec(result).-expected))))
    end
    times = Float64[]
    for _ in 1:samples
        GC.gc()
        push!(times, @elapsed(f()))
    end
    row = (; tokens=n, median_ms=1000median(times), output=vec(result))
    push!(results, row)
    println((; tokens=n, median_ms=row.median_ms)); flush(stdout)
end
aggregate = sum(r.median_ms for r in results)
report = (; device, checkpoint, julia=string(VERSION), samples, all_tokens="--all" in ARGS,
           aggregate_ms=aggregate, results)
output = option("--output", "")
isempty(output) || open(io -> JSON.print(io, report), output, "w")
println((; aggregate_ms=aggregate))
if baseline !== nothing
    @assert baseline["device"] == string(device) && baseline["checkpoint"] == checkpoint
    @assert baseline["all_tokens"] == ("--all" in ARGS)
    speedup = 100(baseline["aggregate_ms"]/aggregate - 1)
    println((; speedup_percent=speedup))
    @assert speedup >= parse(Float64, option("--require-speedup", "0"))
end
