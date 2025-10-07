using Documenter, TextAssociations, DataFrames, CSV, Dates

# using Documenter
# using TextAssociations
# using TextAnalysis
# using DataFrames
# using CSV
# using Dates


makedocs(
    sitename="TextAssociations.jl",
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://atantos.github.io/TextAssociations.jl/",
        assets=String["assets/custom.css"],
        sidebar_sitename=true,
        collapselevel=2,
        repolink="https://github.com/atantos/TextAssociations.jl",
        edit_link="main",
    ),
    modules=[TextAssociations],
    authors="Alexandros Tantos <atantos@gmail.com>",
    repo="https://github.com/atantos/TextAssociations.jl",
    pages=[
        "Home" => "index.md",
        "Contributing" => "contributing.md",
        "Getting Started" => [
            "Installation" => "getting_started/installation.md",
            "Quick Tutorial" => "getting_started/tutorial.md",
            "Basic Examples" => "getting_started/examples.md",
        ],
        "User Guide" => [
            "Core Concepts" => "guide/concepts.md",
            "Text Preprocessing" => "guide/preprocessing.md",
            "Choosing Metrics" => "guide/choosing_metrics.md",
            "Working with Corpora" => "guide/corpus_analysis.md",
        ],
        "Metrics" => [
            "Overview" => "metrics/overview.md",
            "Information Theoretic" => "metrics/information_theoretic.md",
            "Statistical" => "metrics/statistical.md",
            "Similarity" => "metrics/similarity.md",
            "Epidemiological" => "metrics/epidemiological.md",
        ],
        "Advanced Features" => [
            "Temporal Analysis" => "advanced/temporal.md",
            "Network Analysis" => "advanced/networks.md",
            "Keyword Extraction" => "advanced/keywords.md",
            # "Concordance" => "advanced/concordance.md",
        ],
        "API Reference" => [
            "Overview" => "api/index.md",
            "Core Types" => "api/types.md",
            "Main Functions" => "api/functions.md",
            # "Corpus Functions" => "api/corpus.md",
            "Metric Functions" => "api/metrics.md",
        ],
        #     "Theory" => "theory.md",
        #     # "How-To Guides" => "howto.md",
        #     "Performance" => "performance.md",
        #     "Troubleshooting" => "troubleshooting.md",
        #     "Contributing" => "contributing.md",
    ],
    pagesonly=true,
    warnonly=true,
    checkdocs=:exports,
    doctestfilters=Regex[
        r"Ptr{0x[0-9a-f]+}",
        r"[0-9\.]+ seconds \(.*\)"
    ],
)

# deploydocs(
#     repo="github.com/atantos/TextAssociations.jl.git",
#     devbranch="main",
#     push_preview=true,
#     deps=nothing,
#     make=nothing,
# )