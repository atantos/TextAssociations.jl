# using TextAssociations
# using Documenter

# DocMeta.setdocmeta!(TextAssociations, :DocTestSetup, :(using TextAssociations); recursive=true)

# makedocs(;
#     modules=[TextAssociations],
#     authors="atantos <atantos@gmail.com> and contributors",
#     sitename="TextAssociations.jl",
#     format=Documenter.HTML(
#         prettyurls=false,  # Disable pretty URLs and generate flat files without subdirectories that create problems in the linking
#         canonical="https://atantos.github.io/TextAssociations.jl",
#         edit_link="main",
#         assets=String[]
#     ),
#     pages=[
#         "Intro" => "index.md",
#         "Quick Start" => "tutorial.md",
#         "Guided Examples" => "howto.md",
#         "API Reference" => "reference.md",
#         "Theory" => "theory.md",
#     ],
# )

# deploydocs(;
#     repo="github.com/atantos/TextAssociations.jl",
#     devbranch="main",
#     # Add the following if you want to generate 'stable' docs:
#     # stablebranch="main",
#     # push_preview=true, # You can set this to `false` after you've confirmed it works.
# )
using Documenter
using TextAssociations

# Configure Documenter
makedocs(
    sitename="TextAssociations.jl",
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", nothing) == "true",
        canonical="https://atantos.github.io/TextAssociations.jl/stable/",
        assets=String[],
        sidebar_sitename=true,
        collapselevel=2,
        # Add custom CSS if needed
        # assets = ["assets/custom.css"],
        # Add analytics if desired
        # analytics = "UA-XXXXXXXXX-X",
    ),
    modules=[TextAssociations],
    authors="Alexandros Tantos <atantos@gmail.com>",
    repo="https://github.com/yourusername/TextAssociations.jl/blob/{commit}{path}#{line}",
    pages=[
        "Home" => "index.md",
        "Getting Started" => [
            "Installation" => "getting_started/installation.md",
            "Quick Tutorial" => "getting_started/tutorial.md",
            "Basic Examples" => "getting_started/basic_examples.md",
        ],
        "User Guide" => [
            "Core Concepts" => "guide/concepts.md",
            "Text Preprocessing" => "guide/preprocessing.md",
            "Corpus Analysis" => "guide/corpus_analysis.md",
            "Choosing Metrics" => "guide/choosing_metrics.md",
            "Best Practices" => "guide/best_practices.md",
        ],
        "Metrics" => [
            "Overview" => "metrics/overview.md",
            "Information-Theoretic" => "metrics/information_theoretic.md",
            "Statistical" => "metrics/statistical.md",
            "Similarity" => "metrics/similarity.md",
            "Effect Size" => "metrics/effect_size.md",
            "Specialized" => "metrics/specialized.md",
        ],
        "Advanced Features" => [
            "Temporal Analysis" => "advanced/temporal.md",
            "Comparative Analysis" => "advanced/comparative.md",
            "Keyword Extraction" => "advanced/keywords.md",
            "Network Analysis" => "advanced/networks.md",
            "Concordance" => "advanced/concordance.md",
        ],
        "Examples" => [
            "Simple Analysis" => "examples/simple.md",
            "Corpus Examples" => "examples/corpus.md",
            "Real-World Cases" => "examples/real_world.md",
            "Multilingual" => "examples/multilingual.md",
            "Performance" => "examples/performance.md",
        ],
        "API Reference" => [
            "Core Types" => "api/types.md",
            "Main Functions" => "api/functions.md",
            "Corpus Functions" => "api/corpus.md",
            "Metric Functions" => "api/metrics.md",
            "Utilities" => "api/utilities.md",
        ],
        "Development" => [
            "Contributing" => "dev/contributing.md",
            "Adding Metrics" => "dev/adding_metrics.md",
            "Testing" => "dev/testing.md",
        ],
    ],
    strict=false,  # Set to true in production
    checkdocs=:exports,
    doctestfilters=Regex[
        r"Ptr{0x[0-9a-f]+}",
        r"[0-9\.]+ seconds \(.*\)"
    ],
)

# Deploy documentation
deploydocs(
    repo="github.com/atantos/TextAssociations.jl.git",
    devbranch="main",
    push_preview=true,
    deps=nothing,
    make=nothing,
)