# Optional hardware suite: julia --project=<CUDA test environment> -t 1 test/cuda.jl
using Test, QwenDecisionCore, CUDA, Random, LinearAlgebra
const QDC = QwenDecisionCore
include("../benchmark/synthetic.jl")
CUDA.allowscalar(false)
BLAS.set_num_threads(1)
@testset "CUDA inference agrees with CPU" begin
    for cpu in (QwenBackbone(joinpath(@__DIR__, "fixtures", "native")), synthetic_backbone())
        gpu = device_backbone(cpu, :cuda)
        for n in (1, 9, 63, 64, 65, 129, 256, 257), holes in (false, true)
            ids = mod.(collect(0:(n-1)), size(cpu.embedding, 2))
            mask = ones(Int, n)
            holes && n > 1 && (mask[1:3:(n-1)] .= 0)
            expected = backbone_hidden(cpu, ids, mask)
            actual = backbone_hidden(gpu, ids, mask)
            @test actual ≈ expected atol=2e-5 rtol=2e-5
            last = backbone_last_hidden(gpu, ids, mask)
            @test last ≈ expected[:, end] atol=2e-5 rtol=2e-5
            saved = copy(last)
            @test backbone_last_hidden(gpu, ids, mask) ≈ expected[:, end] atol=2e-5 rtol=2e-5
            extension = Base.get_extension(QDC, :QwenDecisionCoreCUDAExt)
            @test extension.workspace(gpu.embedding).forward_graph.graph !== nothing
            # Replay on changed values at fixed shapes, then invalidate by
            # switching to the all-token scratch layout.
            changed = backbone_last_hidden(gpu, reverse(ids), mask)
            @test changed ≈ backbone_last_hidden(cpu, reverse(ids), mask) atol=2e-5 rtol=2e-5
            @test backbone_hidden(gpu, ids, mask) ≈ expected atol=2e-5 rtol=2e-5
            @test last == saved
        end
    end
end

@testset "CUDA graph reuse across tasks and streams" begin
    path = joinpath(@__DIR__, "fixtures", "native")
    cpu, gpu = QwenBackbone(path), QwenBackbone(path; device=:cuda)
    ids = mod.(collect(0:8), size(cpu.embedding, 2))
    mask = ones(Int, length(ids))
    for _ in 1:3; backbone_last_hidden(gpu, ids, mask); end
    inputs = (ids, reverse(ids), ids, reverse(ids))
    tasks = [Threads.@spawn(backbone_last_hidden(gpu, input, mask)) for input in inputs]
    for (task, input) in zip(tasks, inputs)
        @test fetch(task) ≈ backbone_last_hidden(cpu, input, mask) atol=2e-5 rtol=2e-5
    end
    extension = Base.get_extension(QDC, :QwenDecisionCoreCUDAExt)
    @test extension.workspace(gpu.embedding).forward_graph.graph !== nothing
end
