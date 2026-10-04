using Documenter
using QwenDecisionCore

makedocs(;
    sitename = "QwenDecisionCore.jl",
    authors = "Satoshi Terasaki",
    modules = [QwenDecisionCore],
    remotes = nothing,
    doctest = false,
    checkdocs = :none,
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        repolink = "https://github.com/AtelierArith/QwenDecisionCore.jl",
        edit_link = nothing,
    ),
    pages = ["Home" => "index.md", "API reference" => "api.md"],
)

if get(ENV, "CI", "false") == "true"
    deploydocs(;
        repo = "github.com/AtelierArith/QwenDecisionCore.jl.git",
        devbranch = "main",
        push_preview = false,
    )
end
