# Hugging Face checkpoint resolution and caching. The cache strategy follows
# Laya.jl/src/agent.jl: consult the existing HF cache, then keep complete
# downloads in package-specific Scratch space. Required files and weight
# patterns are keywords so the backbone loader and a client's decision
# checkpoint can share the same resolver.

const DEFAULT_REQUIRED = ("config.json",)
const DEFAULT_AUXILIARY = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "processor_config.json",
    "model.safetensors.index.json",
    "LICENSE",
    "NOTICE",
)
const DEFAULT_PATTERNS = (r"^model(?:-[0-9]+-of-[0-9]+)?\.safetensors$",)

function safe_relative_path(path::AbstractString)
    isempty(path) && throw(ArgumentError("Path must not be empty."))
    (
        isabspath(path) ||
        occursin('\\', path) ||
        occursin('\0', path) ||
        any(part -> part in ("", ".", ".."), split(path, '/'))
    ) && throw(ArgumentError("Expected a safe relative path: $path"))
    return path
end

checkpoint_file(path, required, auxiliary, patterns) =
    path in required ||
    path in auxiliary ||
    any(pattern -> occursin(pattern, path), patterns)

function complete_checkpoint(dir, required)
    all(file -> isfile(joinpath(dir, file)), required) || return false
    index = joinpath(dir, "model.safetensors.index.json")
    isfile(index) || return true
    mapping = JSON.parsefile(index)["weight_map"]
    isempty(mapping) && return false
    return all(values(mapping)) do file
        safe_relative_path(file)
        isfile(joinpath(dir, file))
    end
end

has_weight_files(files) =
    any(file -> endswith(file, ".safetensors") || endswith(file, ".index.json"), files)

function huggingface_cache()
    haskey(ENV, "HF_HUB_CACHE") && return ENV["HF_HUB_CACHE"]
    haskey(ENV, "HF_HOME") && return joinpath(ENV["HF_HOME"], "hub")
    return joinpath(homedir(), ".cache", "huggingface", "hub")
end

hub_offline() = lowercase(get(ENV, "HF_HUB_OFFLINE", "0")) in ("1", "true", "yes", "on")
hub_root(cache, repo) = joinpath(cache, "models--" * replace(repo, "/" => "--"))

function cached_checkpoint(cache, repo, revision, required)
    root = hub_root(cache, repo)
    ref = joinpath(root, "refs", revision)
    commit = isfile(ref) ? strip(read(ref, String)) : revision
    occursin(r"^[0-9a-f]{40}$", commit) || return nothing
    dir = joinpath(root, "snapshots", commit)
    return complete_checkpoint(dir, required) ? dir : nothing
end

function escape_hub_path(path)
    return join(
        split(path, '/') .|>
        part -> join(
            (
                byte in UInt8('a'):UInt8('z') ||
                byte in UInt8('A'):UInt8('Z') ||
                byte in UInt8('0'):UInt8('9') ||
                byte in codeunits("-._~")
            ) ? string(Char(byte)) : "%" * uppercase(string(byte; base = 16, pad = 2))
            for byte in codeunits(part)
        ),
        '/',
    )
end

function publish_file(source, destination)
    mkpath(dirname(destination))
    # rename within the cache filesystem publishes complete files atomically.
    Base.Filesystem.rename(source, destination)
end

"""
    resolve_checkpoint(model_id_or_path; revision="main", cache_dir=nothing,
                       offline=hub_offline(), required=DEFAULT_REQUIRED,
                       auxiliary=DEFAULT_AUXILIARY, patterns=DEFAULT_PATTERNS)

Return a complete local checkpoint directory. Look up repository IDs in the
existing Hugging Face cache, then in this package's Scratch cache; download
only missing files if needed. A local directory is validated and returned
without downloading.

Pass an immutable 40-character commit as `revision` for reproducibility.
Branch names resolve to immutable snapshots on first download; subsequent
calls reuse the cached reference. `cache_dir` overrides Scratch storage for
this call. `HF_HUB_CACHE`, `HF_HOME`, `HF_ENDPOINT`, `HF_TOKEN`, and
`HF_HUB_OFFLINE` are respected. Partial downloads are kept outside completed
snapshot files and are removed on failure.

`required` lists files that must be present, `auxiliary` optional files that
are still downloaded, and `patterns` regexes for weight shards (a client
checkpoint can add its adapter and head files).
"""
function resolve_checkpoint(
    model_id_or_path::AbstractString;
    revision::AbstractString = "main",
    cache_dir::Union{Nothing,AbstractString} = nothing,
    offline::Bool = hub_offline(),
    required = DEFAULT_REQUIRED,
    auxiliary = DEFAULT_AUXILIARY,
    patterns = DEFAULT_PATTERNS,
    require_weights::Bool = true,
)
    local_path = expanduser(model_id_or_path)
    if isdir(local_path)
        complete_checkpoint(local_path, required) ||
            throw(ArgumentError("Incomplete checkpoint: $local_path"))
        return abspath(local_path)
    end
    occursin(
        r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$",
        model_id_or_path,
    ) || throw(
        ArgumentError(
            "Use an existing checkpoint directory or an org/model repository ID.",
        ),
    )
    safe_relative_path(revision)
    repo = String(model_id_or_path)
    shared = cached_checkpoint(huggingface_cache(), repo, revision, required)
    shared === nothing || return shared
    cache =
        cache_dir === nothing ? Scratch.@get_scratch!("hub") :
        abspath(expanduser(cache_dir))
    cached = cached_checkpoint(cache, repo, revision, required)
    cached === nothing || return cached
    offline && throw(
        ArgumentError("$repo@$revision is not cached; offline mode prevents downloading."),
    )

    endpoint = rstrip(get(ENV, "HF_ENDPOINT", "https://huggingface.co"), '/')
    headers =
        haskey(ENV, "HF_TOKEN") ? ["Authorization" => "Bearer $(ENV["HF_TOKEN"])"] :
        Pair{String,String}[]
    url = "$endpoint/api/models/$repo/revision/$(escape_hub_path(revision))"
    info = JSON.parse(String(take!(Downloads.download(url, IOBuffer(); headers))))
    commit = String(info["sha"])
    occursin(r"^[0-9a-f]{40}$", commit) ||
        throw(ArgumentError("Hub returned an invalid commit ID."))
    occursin(r"^[0-9a-f]{40}$", revision) &&
        revision != commit &&
        throw(ArgumentError("Hub returned a different commit than the requested revision."))
    files = String[
        item["rfilename"] for
        item in info["siblings"] if checkpoint_file(item["rfilename"], required, auxiliary, patterns)
    ]
    all(file -> file in files, required) ||
        throw(ArgumentError("$repo@$commit is missing required checkpoint files."))
    require_weights && !has_weight_files(files) &&
        throw(ArgumentError("$repo@$commit has no model weights."))
    root = hub_root(cache, repo)
    snapshot = joinpath(root, "snapshots", commit)
    mkpath(root)
    mktempdir(root; prefix = "download-") do temporary
        for file in files
            safe_relative_path(file)
            destination = joinpath(snapshot, file)
            isfile(destination) && continue
            staged = joinpath(temporary, file)
            @info "Downloading checkpoint file" repo commit file
            Downloads.download(
                "$endpoint/$repo/resolve/$commit/$(escape_hub_path(file))",
                staged;
                headers,
            )
            publish_file(staged, destination)
        end
        complete_checkpoint(snapshot, required) ||
            throw(ArgumentError("Downloaded checkpoint is incomplete: $snapshot"))
        staged_ref = joinpath(temporary, "revision")
        write(staged_ref, commit)
        publish_file(staged_ref, joinpath(root, "refs", revision))
    end
    return snapshot
end
