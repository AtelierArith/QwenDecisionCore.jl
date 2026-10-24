# Run in a fresh process with MKL_INTERFACE_LAYER=ILP64 and julia -t 1.
using Test, QwenDecisionCore, LinearAlgebra
const QDC = QwenDecisionCore
Threads.nthreads(:default) == 1 || error("Run this ABI check with julia -t 1.")
BLAS.set_num_threads(1)
@testset "Private SGEMM uses explicit 64-bit ABI" begin
    weight = reshape(sin.(Float32.(1:2048)),64,32)
    input = reshape(cos.(Float32.(1:448)),64,7)
    expected = transpose(weight)*input
    actual = QDC.with_cpu_settings(:mkl_serial=>true) do
        QDC.cpu_projection!(zeros(Float32,32,7),weight,input)
    end
    @test QDC.CPU_MKL_GEMM[] != C_NULL
    @test actual ≈ expected atol=2e-5 rtol=2e-5
    @test BLAS.get_num_threads() == 1
end
