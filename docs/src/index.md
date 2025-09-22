<!-- ```@meta
CurrentModule = TextAssociations
```

# TextAssociations

Documentation for [TextAssociations](https://github.com/atantos/TextAssociations.jl).

## Install

```julia-repl
julia> add https://github.com/atantos/TextAssociations.jl
``` -->

# TextAssociations.jl

```@meta
CurrentModule = TextAssociations
```

_A comprehensive Julia package for word association analysis and collocation extraction_

## Overview

TextAssociations.jl provides a powerful and efficient framework for computing word association metrics and performing corpus-level collocation analysis. With over 50 implemented association measures, this package serves researchers in computational linguistics, corpus linguistics, natural language processing, and digital humanities.

!!! note "Package Highlights" - **50+ association metrics** including PMI, LogDice, LLR, and many more - **Efficient processing** with lazy evaluation and caching - **Corpus analysis** at scale with streaming and parallel processing - **Multilingual support** including proper Unicode and diacritic handling - **Advanced features** like temporal analysis and collocation networks

## Quick Start

```@example quickstart
using TextAssociations

# Analyze collocations in text
text = """
Machine learning algorithms learn patterns from data.
Deep learning is a subset of machine learning.
Neural networks power deep learning systems.
"""

# Find collocations of "learning"
ct = ContingencyTable(text, "learning", windowsize=3, minfreq=1)

# Calculate multiple metrics
results = evalassoc([PMI, LogDice, LLR], ct)

# Display top collocations
using DataFrames
sort!(results, :PMI, rev=true)
first(results, 5)
```

## Key Features

### 📊 Comprehensive Metric Collection

The package implements metrics from various theoretical frameworks:

- **Information-theoretic**: PMI, PPMI, Mutual Information variants
- **Statistical**: Log-likelihood ratio, Chi-square, T-score, Z-score
- **Similarity-based**: Dice, Jaccard, Cosine similarity
- **Effect size**: Odds ratio, Relative risk, Cohen's d
- **Specialized**: Lexical Gravity, Delta P, Minimum Sensitivity

### 🚀 Performance and Scalability

- **Lazy evaluation**: Computations are deferred and cached
- **Memory efficient**: Stream processing for large corpora
- **Parallel processing**: Built-in support for distributed computing
- **Optimized algorithms**: Efficient implementations for all metrics

### 🔧 Flexible and Extensible

- **Multiple input formats**: Raw text, files, directories, CSV, JSON
- **Customizable preprocessing**: Full control over text normalization
- **Extensible design**: Easy to add new metrics or modify existing ones
- **Rich output options**: DataFrames, CSV, JSON, Excel export

## Installation

```julia
using Pkg
Pkg.add("TextAssociations")
```

For the development version:

```julia
Pkg.add(url="https://github.com/yourusername/TextAssociations.jl")
```

See [Installation](@ref) for detailed instructions and troubleshooting.

## Basic Usage

### Single Document Analysis

```@example basic
using TextAssociations

# Load and preprocess text
doc = prepstring("Your text here...",
    strip_punctuation=true,
    strip_case=true
)

# Create contingency table
ct = ContingencyTable(text(doc), "word", windowsize=5, minfreq=3)

# Evaluate metrics
pmi_scores = evalassoc(PMI, ct)
```

### Corpus-Level Analysis

```@example corpus
using TextAssociations

# Load corpus from directory
corpus = load_corpus("path/to/documents/")

# Analyze across entire corpus
results = analyze_corpus(corpus, "innovation", PMI,
    windowsize=5,
    minfreq=10
)

# Get corpus statistics
stats = corpus_statistics(corpus)
println("Documents: $(stats[:num_documents])")
println("Vocabulary: $(stats[:vocabulary_size])")
```

## Advanced Features

### Temporal Analysis

Track how word associations change over time:

```julia
temporal_analysis = temporal_corpus_analysis(
    corpus, ["digital", "transformation"], :year, PMI
)
```

### Collocation Networks

Build networks of related terms:

```julia
network = build_collocation_network(
    corpus, ["artificial", "intelligence"],
    metric=PMI, depth=2
)
```

### Comparative Analysis

Compare associations across subcorpora:

```julia
comparison = compare_subcorpora(
    corpus, :category, "technology", PMI
)
```

## Package Architecture

```
TextAssociations.jl
│
├── Core Components
│   ├── ContingencyTable     # Single document analysis
│   ├── Corpus               # Document collection
│   └── Metrics              # Association measures
│
├── Analysis Functions
│   ├── evalassoc()          # Metric evaluation
│   ├── analyze_corpus()     # Corpus analysis
│   └── extract_keywords()   # Keyword extraction
│
└── Advanced Features
    ├── Temporal analysis
    ├── Network building
    └── Concordance generation
```

## Documentation Guide

```@raw html
<div class="card-container">
    <div class="card">
        <h3>🚀 Getting Started</h3>
        <ul>
            <li><a href="getting_started/installation.html">Installation Guide</a></li>
            <li><a href="getting_started/tutorial.html">Quick Tutorial</a></li>
            <li><a href="getting_started/basic_examples.html">Basic Examples</a></li>
        </ul>
    </div>
    <div class="card">
        <h3>📖 User Guide</h3>
        <ul>
            <li><a href="guide/concepts.html">Core Concepts</a></li>
            <li><a href="guide/preprocessing.html">Text Preprocessing</a></li>
            <li><a href="guide/choosing_metrics.html">Choosing Metrics</a></li>
        </ul>
    </div>
    <div class="card">
        <h3>📊 Metrics</h3>
        <ul>
            <li><a href="metrics/overview.html">Metrics Overview</a></li>
            <li><a href="metrics/information_theoretic.html">Information-theoretic</a></li>
            <li><a href="metrics/statistical.html">Statistical Tests</a></li>
        </ul>
    </div>
    <div class="card">
        <h3>🔬 Advanced</h3>
        <ul>
            <li><a href="advanced/temporal.html">Temporal Analysis</a></li>
            <li><a href="advanced/networks.html">Network Analysis</a></li>
            <li><a href="advanced/keywords.html">Keyword Extraction</a></li>
        </ul>
    </div>
</div>
```

## Performance Benchmarks

| Task            | Size      | Time   | Memory   |
| --------------- | --------- | ------ | -------- |
| Single document | 10K words | ~50ms  | 10MB     |
| Small corpus    | 100 docs  | ~2s    | 50MB     |
| Large corpus    | 10K docs  | ~30s   | 500MB    |
| Streaming       | Unlimited | Linear | Constant |

## Community and Support

- 📚 [Complete API Reference](@ref api_types)
- 💬 [GitHub Discussions](https://github.com/yourusername/TextAssociations.jl/discussions)
- 🐛 [Issue Tracker](https://github.com/yourusername/TextAssociations.jl/issues)
- 📧 Contact: your.email@example.com

## Citation

If you use TextAssociations.jl in your research, please cite:

```bibtex
@software{textassociations2024,
    title = {TextAssociations.jl: A Julia Package for Word Association Analysis},
    author = {Your Name},
    year = {2024},
    url = {https://github.com/yourusername/TextAssociations.jl},
    version = {0.1.0}
}
```

## Contributing

We welcome contributions! See our [Contributing Guide](@ref) for:

- Bug reports and feature requests
- Code contributions
- Documentation improvements
- Adding new metrics

## License

TextAssociations.jl is licensed under the [MIT License](https://github.com/yourusername/TextAssociations.jl/blob/main/LICENSE).

## Acknowledgments

This package builds upon decades of research in computational linguistics:

- Evert, S. (2008). "Corpora and collocations." _Corpus Linguistics: An International Handbook_
- Church, K. W., & Hanks, P. (1990). "Word association norms, mutual information, and lexicography." _Computational Linguistics_
- Pecina, P. (2010). "Lexical association measures and collocation extraction." _Language Resources and Evaluation_

## Index

```@index

```

## Functions

```@autodocs
Modules = [TextAssociations]
Order   = [:function]
```
