using Documenter
using TextAssociations

makedocs(
    sitename="TextAssociations.jl",
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://atantos.github.io/TextAssociations.jl/",
        assets=String[],
        sidebar_sitename=true,
        collapselevel=2,
        edit_link="main",
    ),
    modules=[TextAssociations],
    authors="Alexandros Tantos <atantos@gmail.com>",
    repo="https://github.com/atantos/TextAssociations.jl",
    pages=[
        "Getting Started" => [
            "Installation" => "getting_started/installation.md",
            "Quick Tutorial" => "getting_started/tutorial.md",
        ],
        "API Reference" => [
            "Core Types" => "api/types.md",
            "Main Functions" => "api/functions.md",
        ],
        # add more entries here only when the files actually exist under docs/src/...
    ],
    warnonly=true,
    checkdocs=:exports,
    doctestfilters=Regex[
        r"Ptr{0x[0-9a-f]+}",
        r"[0-9\.]+ seconds \(.*\)"
    ],
)

deploydocs(
    repo="github.com/atantos/TextAssociations.jl.git",
    devbranch="main",
    push_preview=true,
    deps=nothing,
    make=nothing,
)
