using QwenDecisionCore, Random, Statistics, LinearAlgebra
const QDC = QwenDecisionCore
Random.seed!(42)
# A deterministic hybrid backbone with production-sized Delta head dimensions.
function synthetic_backbone(; hidden=256, intermediate=512, depth=4)
    cfg = (hidden=hidden, heads=4, kv_heads=2, head_dim=64, key_heads=2,
           value_heads=4, key_dim=128, value_dim=128, eps=1f-6,
           rope_theta=10000f0, rotary_dim=16)
    weight(a,b) = randn(Float32,a,b) ./ sqrt(Float32(a))
    layers = map(1:depth) do i
        attention = if i % 4 == 0
            (kind=:full, q=weight(hidden,512), k=weight(hidden,128),
             v=weight(hidden,128), out=weight(256,hidden),
             q_norm=zeros(Float32,64), k_norm=zeros(Float32,64))
        else
            (kind=:delta, qkv=weight(hidden,1024), z=weight(hidden,512),
             a=weight(hidden,4), b=weight(hidden,4),
             conv=transpose(randn(Float32,1024,4) .* 0.1f0),
             a_decay=-ones(Float32,4), dt_bias=zeros(Float32,4),
             norm=ones(Float32,128), out=weight(512,hidden))
        end
        mlp=(gate=weight(hidden,intermediate), up=weight(hidden,intermediate),
             down=weight(intermediate,hidden))
        QDC.QwenLayer(attention,mlp,zeros(Float32,hidden),zeros(Float32,hidden))
    end
    concrete = Vector{Union{unique(typeof.(layers))...}}(layers)
    QwenBackbone(weight(hidden,256),concrete,zeros(Float32,hidden),cfg)
end
# Preserve the original all-token implementation for an end-to-end comparison.
function original_hidden(b, ids, mask)
    QDC.check_sequence(b, ids, mask)
    return QDC.native_forward_scope(b.embedding, length(ids)) do
        hidden = QDC.native_gather(b.embedding, collect(Int, ids))
        prepared = QDC.native_prepare_mask(hidden, collect(Int, mask))
        QDC.native_host(invoke(QDC.native_hidden_states,
            Tuple{Any,Any,Any,Any,Any}, hidden, b.layers, prepared, b.final_norm, b.config))
    end
end
checkpoint = findfirst(==("--checkpoint"), ARGS)
b = checkpoint === nothing ? synthetic_backbone() : QwenBackbone(ARGS[checkpoint+1])
defaults = QDC.cpu_settings()
optimized_threads = BLAS.get_num_threads()
baseline_threads = defaults.accelerate || Threads.nthreads() == 1 ? min(8, Sys.CPU_THREADS) : 1
println("Julia ", VERSION, "; Julia threads=", Threads.nthreads(),
        "; baseline BLAS threads=", baseline_threads,
        "; optimized BLAS threads=", optimized_threads)
flush(stdout)
results = NamedTuple[]
for n in (64,128,256,512)
    ids=rand(0:255,n); mask=ones(Int,n)
    function f(optimized)
        # Each preceding forward has joined all workers before changing BLAS.
        BLAS.set_num_threads(optimized ? optimized_threads : baseline_threads)
        return QDC.with_cpu_settings(
            :recurrent_vector_math => optimized && defaults.recurrent_vector_math,
            :fused_convolution => optimized && defaults.fused_convolution,
        ) do
            if "--last" in ARGS
                backbone_last_hidden(b,ids,mask)
            else
                optimized ? backbone_hidden(b,ids,mask) : original_hidden(b,ids,mask)
            end
        end
    end
    reference=f(false); result=f(true)
    @assert isapprox(result, reference; atol=2e-5, rtol=2e-5)
    for _ in 1:3; f(false); f(true); end
    baseline, optimized = Float64[], Float64[]
    # Alternate ordering to reduce effects from CPU frequency and machine load.
    for sample in 1:21
        for vectorized in (isodd(sample) ? (false,true) : (true,false))
            GC.gc()
            elapsed = @elapsed f(vectorized)
            push!(vectorized ? optimized : baseline, elapsed*1000)
        end
    end
    old, new = median(baseline), median(optimized)
    stats = (tokens=n, baseline_ms=old, optimized_ms=new,
             speedup_percent=100*(old/new-1), time_reduction_percent=100*(1-new/old),
             max_error=maximum(abs.(result-reference)))
    push!(results, stats)
    println(stats)
    flush(stdout)
end

old = sum(r.baseline_ms for r in results)
new = sum(r.optimized_ms for r in results)
println((aggregate_baseline_ms=old, aggregate_optimized_ms=new,
         speedup_percent=100*(old/new-1), time_reduction_percent=100*(1-new/old)))
BLAS.set_num_threads(optimized_threads)
gate = findfirst(==("--require-speedup"), ARGS)
if gate !== nothing
    requested = parse(Float64, ARGS[gate+1])
    @assert 100*(old/new-1) >= requested "Aggregate speedup is below the requested target."
end
