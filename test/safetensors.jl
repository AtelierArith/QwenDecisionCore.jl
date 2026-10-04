@testset "Safetensors round trip" begin
    mktempdir() do dir
        path = joinpath(dir, "weights.safetensors")
        f32 = randn(Float32, 3, 4)
        f16 = randn(Float16, 2, 5)
        bf16 = rand(UInt16, 2, 2)
        write_native_weights(
            path,
            Dict("f32" => f32, "f16" => f16, "bf16" => bf16);
            metadata = Dict("format" => "pt"),
        )
        got = read_native_weights(path)
        @test got["f32"] == f32
        @test got["f16"] == Float32.(f16)
        @test got["bf16"] == reinterpret(Float32, UInt32.(bf16) .<< 16)
        @test Set(keys(read_native_weights(path; select = name -> name == "f32"))) ==
              Set(["f32"])
        @test_throws ArgumentError write_native_weights(
            joinpath(dir, "bad.safetensors"),
            Dict("x" => [1, 2, 3]),
        )
    end
    @test QwenDecisionCore.float32_to_bf16(1.0f0) == 0x3f80
    @test QwenDecisionCore.float32_to_bf16(-2.0f0) == 0xc000
    @test QwenDecisionCore.float32_to_bf16(0.0f0) == 0x0000
end
