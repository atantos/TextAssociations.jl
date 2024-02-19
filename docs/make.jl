using TextAssociations
using Documenter

DocMeta.setdocmeta!(TextAssociations, :DocTestSetup, :(using TextAssociations); recursive=true)

makedocs(;
    modules=[TextAssociations],
    authors="atantos <atantos@gmail.com> and contributors",
    sitename="TextAssociations.jl",
    format=Documenter.HTML(;
        canonical="https://atantos.github.io/TextAssociations.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Intro" => "index.md",
        "Quick Start" => "tutorial.md",
        "Guided Examples" => "howto.md",
        "API Reference" => "reference.md",
        "Theory" => "theory.md",
    ],
)

deploydocs(;
    repo="github.com/atantos/TextAssociations.jl",
    devbranch="main",
)
