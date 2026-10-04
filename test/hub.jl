@testset "Julia checkpoint cache" begin
    mktempdir() do root
        commit = repeat("a", 40)
        repo = "test/qdc-cache"
        endpoint = joinpath(root, "endpoint")
        cache = joinpath(root, "cache")
        shared = joinpath(root, "shared")
        files = [
            "config.json",
            "decision_config.json",
            "readout.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors",
        ]
        # A client checkpoint can require adapter/head files and additional
        # weight patterns; the resolver keywords must honor them.
        required = ("config.json", "decision_config.json", "readout.safetensors")
        auxiliary = ("tokenizer.json", "tokenizer_config.json")
        patterns = (r"^model(?:-[0-9]+-of-[0-9]+)?\.safetensors$",)
        api = joinpath(endpoint, "api", "models", repo, "revision", "main")
        mkpath(dirname(api))
        write(
            api,
            JSON.json(
                Dict(
                    "sha" => commit,
                    "siblings" =>
                        [Dict("rfilename" => name) for name in [files; "videos/demo.mp4"]],
                ),
            ),
        )
        source = joinpath(endpoint, repo, "resolve", commit)
        mkpath(source)
        for file in files
            write(joinpath(source, file), "fixture")
        end
        resolve(args...; kwargs...) = resolve_checkpoint(
            args...;
            required = required,
            auxiliary = auxiliary,
            patterns = patterns,
            kwargs...,
        )
        withenv(
            "HF_ENDPOINT" => "file://$endpoint",
            "HF_HUB_CACHE" => shared,
            "HF_HUB_OFFLINE" => "0",
            "HF_TOKEN" => nothing,
        ) do
            dir = resolve(repo; cache_dir = cache)
            @test dir == joinpath(cache, "models--test--qdc-cache", "snapshots", commit)
            @test Set(readdir(dir)) == Set(files)
            @test read(
                joinpath(cache, "models--test--qdc-cache", "refs", "main"),
                String,
            ) == commit
            @test resolve_checkpoint(dir) == dir
            @test resolve(repo; cache_dir = cache, offline = true) == dir
            @test resolve(repo; revision = commit, cache_dir = cache, offline = true) ==
                  dir
            @test_throws ArgumentError resolve("test/missing"; cache_dir = cache, offline = true)
            @test_throws ArgumentError resolve(repo; revision = "../outside", cache_dir = cache)
            @test_throws ArgumentError resolve_checkpoint("./missing")
            @test_throws ArgumentError resolve_checkpoint(mktempdir(root))

            # Existing HF snapshots take precedence over the package cache.
            shared_dir =
                joinpath(shared, "models--test--qdc-cache", "snapshots", commit)
            mkpath(dirname(shared_dir))
            cp(dir, shared_dir)
            @test resolve(repo; revision = commit, cache_dir = cache, offline = true) ==
                  shared_dir

            # Exercise default Scratch storage, using a unique fixture repo.
            scratch_repo = "fixture/qdc-" * basename(root)
            scratch_api =
                joinpath(endpoint, "api", "models", scratch_repo, "revision", "main")
            mkpath(dirname(scratch_api))
            cp(api, scratch_api)
            scratch_source = joinpath(endpoint, scratch_repo, "resolve", commit)
            mkpath(dirname(scratch_source))
            cp(source, scratch_source)
            scratch_dir = resolve(scratch_repo)
            @test startswith(scratch_dir, joinpath(first(DEPOT_PATH), "scratchspaces"))
            @test resolve(scratch_repo; offline = true) == scratch_dir
        end
    end
end

@testset "Interrupted download and resume" begin
    mktempdir() do root
        commit = repeat("b", 40)
        repo = "test/qdc-resume"
        endpoint = joinpath(root, "endpoint")
        cache = joinpath(root, "cache")
        files = ["config.json", "decision_config.json", "model.safetensors"]
        required = ("config.json", "decision_config.json")
        patterns = (r"^model(?:-[0-9]+-of-[0-9]+)?\.safetensors$",)
        api = joinpath(endpoint, "api", "models", repo, "revision", "main")
        mkpath(dirname(api))
        write(
            api,
            JSON.json(
                Dict(
                    "sha" => commit,
                    "siblings" => [Dict("rfilename" => name) for name in files],
                ),
            ),
        )
        source = joinpath(endpoint, repo, "resolve", commit)
        mkpath(source)
        for file in files[1:(end-1)]
            write(joinpath(source, file), "fixture")
        end
        withenv(
            "HF_ENDPOINT" => "file://$endpoint",
            "HF_HUB_CACHE" => joinpath(root, "shared"),
            "HF_HUB_OFFLINE" => "0",
            "HF_TOKEN" => nothing,
        ) do
            @test_throws Exception resolve_checkpoint(
                repo;
                cache_dir = cache,
                required = required,
                patterns = patterns,
            )
            cache_root = joinpath(cache, "models--test--qdc-resume")
            @test !isfile(joinpath(cache_root, "refs", "main"))
            @test all(!startswith(name, "download-") for name in readdir(cache_root))
            @test_throws ArgumentError resolve_checkpoint(
                repo;
                cache_dir = cache,
                offline = true,
                required = required,
                patterns = patterns,
            )
            # Resuming preserves files already fully downloaded.
            completed = joinpath(cache_root, "snapshots", commit, "config.json")
            write(completed, "keep this cached file")
            write(joinpath(source, "model.safetensors"), "fixture")
            dir = resolve_checkpoint(
                repo;
                cache_dir = cache,
                required = required,
                patterns = patterns,
            )
            @test read(joinpath(dir, "config.json"), String) == "keep this cached file"
            @test resolve_checkpoint(
                repo;
                cache_dir = cache,
                offline = true,
                required = required,
                patterns = patterns,
            ) == dir
        end
    end
end
