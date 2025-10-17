```@meta
CurrentModule = TextAssociations
```

# TextAssociations

Documentation for [TextAssociations](https://github.com/atantos/TextAssociations.jl).

## Install

You can install `TextAssociations.jl` directly from its `GitHub` repository using `Julia`’s package manager. In the `Julia REPL`, press `]` to enter `Pkg` mode and run:

```julia-repl
pkg> add https://github.com/atantos/TextAssociations.jl
```

Once the package is registered in the `Julia` _General registry_, you will be able to install it more simply with:

```julia-repl
pkg> add TextAssociations
```

See [Installation](@ref) for detailed instructions and troubleshooting.

# TextAssociations.jl

```@meta
CurrentModule = TextAssociations
```

_A Julia package for word association measures, collocation analysis and descriptive statistics across text and corpus levels._

## Overview

`TextAssociations.jl` is a comprehensive framework for computing word association metrics, performing collocation analysis, and producing a wide range of descriptive statistical indices at both the text and corpus levels. With 49 implemented association measures, it is designed to support research in computational linguistics, corpus linguistics, natural language processing, and the digital humanities.

!!! tip **Package Highlights**

- 49 association metrics across statistical, information-theoretic, similarity and epidemiological families — including PMI, LogDice, LLR, Chi-square, Odds Ratio, Lexical Gravity, and many more
- **Efficient processing** of large corpora with lazy evaluation and caching
- **Corpus analysis** at scale via batch and streaming modes, with optional parallelism
- **Multilingual support** with proper Unicode handling and diacritic normalization

- **Advanced features**:
  - Temporal association analysis and trend detection
  - Subcorpus comparisons with effect sizes and statistical testing
  - Collocation network construction and export (e.g., to Gephi)
  - KWIC concordances for contextual exploration
  - Keyword extraction (currently TF-IDF, with RAKE and TextRank planned)

## Quick Start

After installation, you can immediately begin analyzing text and exploring collocations with just a few lines of code. The example below demonstrates how to create a contingency table for a target word, compute multiple association measures, and display the top collocates.

For a step-by-step explanation of what happens in each stage and detailed guidance on how to use the package effectively, see the Tutorial section of this documentation.

```@example quickstart
using TextAssociations
using DataFrames

# Analyze collocations in text
text = """
Machine learning algorithms learn patterns from data, allowing computers to make predictions, classifications, and decisions without being explicitly programmed for every possible scenario. Instead of following hard-coded instructions, these algorithms identify statistical regularities in large datasets and use them to generalize from past examples to new situations. Within the broader field of machine learning, deep learning represents a particularly powerful and transformative subset. Deep learning methods are built around neural networks—computational architectures inspired by the structure and function of the human brain—that consist of layers of interconnected nodes or “neurons.” Each layer processes input data and passes its transformed representation to the next, enabling the system to detect increasingly abstract and complex features. Through this hierarchical representation learning, deep learning models can automatically extract meaning from raw data such as images, audio, or text, achieving remarkable performance across domains that once required handcrafted rules and expert knowledge. Neural networks, therefore, form the backbone of deep learning systems, powering technologies like speech recognition, image analysis, and large language models. The success of these systems illustrates how learning from data, rather than from explicit programming, has become a defining paradigm of modern artificial intelligence.
"""

# Find collocations of "learning"
ct = ContingencyTable(text, "learning", windowsize=3, minfreq=1)

# Calculate multiple metrics
results = assoc_score([PMI, LogDice, LLR], ct)

# Display top 5 collocations
sort!(results, :PMI, rev=true)
first(results, 5)
```

## Key Features

### 📊 Comprehensive Metric Collection

The package provides metrics from several families of measures. The examples below are representative; the full list of implemented measures, along with their formulae, is provided in the **Measures**
section.

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

## Basic Usage

### Single Document Analysis

In the code cell below, contingency tables are created for the word "innovation" within the `text_sample`. These tables record all words that occur within a 5-word window around innovation and appear at least once. Based on these counts, `PMI` (Pointwise Mutual Information) scores are computed to measure the strength of association between innovation and each of its collocates. The top five collocates with the highest PMI scores are then displayed in a `DataFrame`.

```@example basic
using TextAssociations
using DataFrames

text_sample =  """
    Computational linguistics increasingly intersects with innovation practice.
    Teams use data to evaluate hypotheses, prototype ideas quickly, and measure impact with reproducible pipelines.
    In modern research workflows, small models are validated against well-defined tasks before scaling, ensuring that innovation is more than a buzzword—it is a methodical, testable process.
    When AI systems are involved, documentation and transparent governance help peers replicate results and trust conclusions.
    """

# Create the contingency tables of "innovation".
ct = ContingencyTable(text_sample, "innovation";
                                    windowsize=5,
                                    minfreq=1,
                                    norm_config=TextNorm(strip_punctuation=true, strip_case=true))

# Calculate PMI for ct, sort and return the top 5 collocates with their scores
pmi_scores = assoc_score(PMI, ct)
sort!(pmi_scores, :PMI, rev=true)
first(pmi_scores, 5)
```

### Corpus-Level Analysis

When moving from a single string or file to a corpus of strings or files, you can create a `Corpus` instance using the `read_corpus` function.
You can then apply the `analyze_node` function, which returns a `DataFrame` containing the same type of information as shown above.
An alternative approach allows you to work at a lower level, examining in more detail how the contingency table is constructed. In this workflow, you first create a `Corpus` instance, then build the contingency table separately, and finally compute association scores using `assoc_score`. For a step-by-step explanation, see the Tutorial section.

```@example corpus
using TextAssociations

# Create a temporary directory that will store a mini-corpus
dir = mktempdir()

files = Dict(
    "doc1.txt" => """
    Computational linguistics increasingly intersects with innovation practice.
    Teams use data to evaluate hypotheses, prototype ideas quickly, and measure impact with reproducible pipelines.
    In modern research workflows, small models are validated against well-defined tasks before scaling, ensuring that innovation is more than a buzzword—it is a methodical, testable process.
    When AI systems are involved, documentation and transparent governance help peers replicate results and trust conclusions.
    """,

    "doc2.txt" => """
    Successful innovation rarely happens in isolation.
    It emerges from an ecosystem of universities, startups, industry labs, and public institutions that collaborate and share partial results early.
    Well-run projects cultivate collaboration rituals—design reviews, error analyses, and postmortems—so ideas move from promising theory to usable tools.
    Open exchange reduces duplication and accelerates learning across the ecosystem.
    """,

    "doc3.txt" => """
    Prototyping is the bridge between research and deployment.
    A minimal prototype clarifies the problem, surfaces risks, and reveals unknown edge cases.
    From there, teams harden the system for scalability, add observability, and evaluate ethical trade-offs such as bias, privacy, and safety.
    A principled evaluation plan is part of the prototype, not an afterthought.
    """,

    "doc4.txt" => """
    Education benefits when innovation is human-centered.
    Instructors can combine classic readings with hands-on labs that trace data through each step of the pipeline.
    Open-source examples and clear rubrics help students reason about uncertainty, interpret model behavior, and articulate the limits of automation.
    The goal is durable understanding and real-world impact, not just higher benchmark scores.
    """
)

# Write files to the temporary directory
for (name, content) in files
    open(joinpath(dir, name), "w") do io
        write(io, strip(content))
    end
end

# Load the corpus from the real path we just created
corpus = read_corpus(dir)

# Analyze across entire corpus
results = analyze_node(corpus, "innovation", PMI,
    windowsize=5,
    minfreq=1
)

# Return the top 5 collocates with the higher PMI score.
first(results, 5)
```

There are 31 descriptive statistical indices available out of the box through the corpus_stats function when applied to a corpus.
Below, only a few representative examples are shown. You can find the complete list and explanations in the Tutorial section.

```@example corpus
# Get corpus statistics
stats = corpus_stats(corpus)
println("Documents: $(stats[:num_documents])")
println("Vocabulary: $(stats[:vocabulary_size])")
println("Type-Token Ratio: $(stats[:type_token_ratio])")
println("Hapax Ratio: $(stats[:hapax_ratio])")
println("Median Type Frequency: $(stats[:median_type_frequency])")
println("Mean Type Frequency: $(stats[:mean_type_frequency ])")
```

## Advanced Features

### Collocation Networks

The code cell below builds a one-layer collocation network centered on the seed word “innovation”.
It scans the corpus using a sliding window (default windowsize=5) and counts the co-occurrences of “innovation” with all neighboring words (only pairs with a frequency ≥ minfreq=1 are considered). Each pair is then scored using LLR (Log-Likelihood Ratio), a robust association measure that performs well even with small samples.

For small corpora, you may need to relax the thresholds (e.g., increase max_neighbors) to prevent the network from being empty. For more details on collocation networks, see the relevant section of the Tutorial.

```@example corpus
network = colloc_graph(
    corpus, ["innovation"],
    metric=LLR,
    depth=1,
    minfreq=1,
    include_frequency=true,
    weight_normalization=:rank,
    compute_centrality=true
)

first(network.edges, 5)
```

The returned `node_metrics` table now includes degree/strength totals and optional
centrality scores, providing a quick overview of the structural role of each term.

### Comparative Analysis

Compare associations across subcorpora:

```julia
comparison = compare_subcorpora(
    corpus, :category, "technology", PMI
)


```

### Temporal Analysis

Track how word associations change over time:

```julia
temporal_analysis = analyze_temporal(
    corpus, ["digital", "transformation"], :year, PMI
)
```

## Package Architecture

```
TextAssociations.jl
│
├─ Types & Basics
│  ├─ AssociationMetric / AssociationDataFormat
│  ├─ TextNorm (single source of truth for normalization)
│  └─ LazyProcess / LazyInput (lazy evaluation & caching)
│
├─ Utils
│  ├─ I/O & encoding (read_text_smart)
│  ├─ Text processing (normalize_node, prep_string, strip_diacritics)
│  ├─ Statistical helpers (available_metrics, log_safe)
│  └─ Text analysis helpers (token find/count utilities)
│
├─ Core Data Structures
│  ├─ ContingencyTable          # per-document co-occurrence table
│  ├─ Corpus                    # collection + vocabulary/DTM
│  └─ CorpusContingencyTable    # corpus-level aggregation (lazy)
│
├─ API (Unified)
│  └─ assoc_score(metric(s), x::AssociationDataFormat; …)
│
├─ Metrics
│  ├─ Interface + dispatch
│  └─ 49 measures across families (PMI, LLR, LogDice, χ², OR, etc.)
│
├─ Analysis Functions
│  ├─ analyze_node / analyze_nodes
│  ├─ corpus_stats, token_distribution, vocab_coverage
│  ├─ write_results, export/load with metadata
│  ├─ batch_process_corpus, stream_corpus_analysis
│  └─ keyterms (TF-IDF; RAKE/TextRank placeholders)
│
└─ Advanced Features
   ├─ analyze_temporal, compare_subcorpora
   ├─ colloc_graph → gephi_graph (network export)
   └─ kwic (concordance)

```

## Documentation Guide

```@raw html
<div class="card-container">
    <div class="card">
        <h3>🚀 Getting Started</h3>
        <ul>
            <li><a href="getting_started/installation/">Installation Guide</a></li>
            <li><a href="getting_started/tutorial/">Quick Tutorial</a></li>
            <li><a href="getting_started/examples/">Basic Examples</a></li>
        </ul>
    </div>
    <div class="card">
        <h3>📖 User Guide</h3>
        <ul>
            <li><a href="guide/concepts">Core Concepts</a></li>
            <li><a href="guide/preprocessing">Text Preprocessing</a></li>
            <li><a href="guide/choosing_metrics">Choosing Metrics</a></li>
        </ul>
    </div>
    <div class="card">
        <h3>📊 Metrics</h3>
        <ul>
            <li><a href="metrics/overview">Metrics Overview</a></li>
            <li><a href="metrics/information_theoretic">Information-theoretic</a></li>
            <li><a href="metrics/statistical">Statistical Tests</a></li>
        </ul>
    </div>
    <div class="card">
        <h3>🔬 Advanced</h3>
        <ul>
            <li><a href="advanced/temporal">Temporal Analysis</a></li>
            <li><a href="advanced/networks">Network Analysis</a></li>
            <li><a href="advanced/keywords">Keyword Extraction</a></li>
        </ul>
    </div>
</div>
```

## Community and Support

- 📚 [Complete API Reference](@ref api_reference)
- 💬 [GitHub Discussions](https://github.com/atantos/TextAssociations.jl/discussions)
- 🐛 [Issue Tracker](https://github.com/atantos/TextAssociations.jl/issues)

## Contributing

We welcome contributions! See our [Contributing Guide](@ref contributing) for:

- Bug reports and feature requests
- Code contributions
- Documentation improvements
- Adding new metrics

## License

TextAssociations.jl is licensed under the [MIT License](https://github.com/yourusername/TextAssociations.jl/blob/main/LICENSE).

## Acknowledgments

This package builds upon decades of research in corpus and computational linguistics. The references that follow are illustrative rather than exhaustive, highlighting some of the key contributions that have shaped the development of association measures and corpus analysis methods.

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
