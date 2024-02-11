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
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/atantos/TextAssociations.jl",
    devbranch="main",
)
