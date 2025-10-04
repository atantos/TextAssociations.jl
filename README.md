<p align="center">
  <img src="https://github.com/atantos/TextAssociations.jl/raw/main/assets/TextAssociations_logo.png" alt="`TextAssociations.jl`" width="1100" height="400"/>
</p>

# TextAssociations.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://atantos.github.io/TextAssociations.jl/)
[![Build Status](https://github.com/atantos/TextAssociations.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/atantos/TextAssociations.jl/actions/workflows/CI.yml?query=branch%3Amain)

## 🎯 Introduction

`TextAssociations.jl` is a comprehensive `Julia` package for word association analysis and corpus linguistics. It currently includes 47 association measures, enabling researchers to quantify lexical relationships within texts and corpora in a transparent, data-driven way.

⚠️ Early Release Notice
This is an early, pre-registration release of `TextAssociations.jl`.
The package is fully functional but still evolving — documentation, tutorials, and examples are actively being expanded.

Even at this stage, it already offers functionality comparable to established corpus analysis tools:

- **AntConc** (but more programmable)
- **SketchEngine** (but open source)
- **WordSmith Tools** (but with more metrics)
<!-- - **Corpus Workbench** (but easier to use) -->

With added advantages of:

- Being fully programmable and extensible
- Integration with `Julia`'s ecosystem
- Support for custom metrics
- Ability to process streaming data
- Modern parallel computing capabilities

This makes `TextAssociations.jl` a powerful tool for computational linguistics, digital humanities, and any field requiring sophisticated text analysis!

### Why Word Association Metrics Still Matter

Even in the era of transformer models and word embeddings, association metrics remain valuable because they:

- 📊 **Are interpretable**: Provide transparent, statistical insights into word relationships
- 🔄 **Complement neural models**: Can be used alongside embeddings to enhance performance and also enhance RAG pipelines.
- 📏 **Serve as benchmarks**: Provide baselines for evaluating complex models
- 💾 **Work with limited data**: Perform well even with small corpora

## ✨ Core Features

### 📈 **47 Association Metrics**

Comprehensive suite including `PMI`, `Log-likelihood`, `Dice`, `Jaccard`, `Lexical Gravity` and many more specialized measures from corpus linguistics, information theory and even some association metrics inspired from epidemiology.

### 📚 **Corpus-Level Analysis**

Process entire document collections with built-in support for:

- Large-scale corpus processing
- Temporal analysis (track changes over time)
- Subcorpus comparison with statistical tests
- Keyword extraction (TF-IDF and other methods soon to come)

### 🚀 **Performance Optimized**

- Lazy evaluation for memory efficiency
- Parallel processing support
- Streaming for massive corpora
- Caching system for repeated analyses

### 🔧 **Flexible and Extensible**

- Multiple input formats (text files, CSV, JSON, DataFrames)
- Easy to add custom metrics
- Comprehensive API for programmatic access

## 📦 Installation

You can install `TextAssociations.jl` directly from its `GitHub` repository using `Julia`’s package manager. In the `Julia REPL`, press `]` to enter `Pkg` mode and run:

```julia
using Pkg
Pkg.add("https://github.com/atantos/TextAssociations.jl")
```

## 🚀 Quick Start

### Basic Usage

```julia
using TextAssociations

# Simple analysis with a single text
text = "The cat sat on the mat. The cat played with the ball."
ct = ContingencyTable(text, "cat", windowsize=3, minfreq=1)

# Calculate PMI scores
pmi_scores = assoc_score(PMI, ct)

# Multiple metrics at once
results = assoc_score([PMI, LogDice, LLR], ct)
```

### Corpus Analysis

```julia
# Load a corpus from a directory
corpus = read_corpus("path/to/texts/", preprocess=true)

# Analyze word associations across the entire corpus
results = analyze_corpus(corpus, "innovation", PMI, windowsize=5, minfreq=10)

# Analyze multiple words with multiple metrics
nodes = ["technology", "innovation", "research"]
metrics = [PMI, LogDice, LLR, ChiSquare]
analysis = analyze_nodes(corpus, nodes, metrics, top_n=100)

# Export results
write_results(analysis, "results/", format=:csv)
```

## 📊 Supported Metrics

`TextAssociations.jl` supports 47 metrics organized by category:

### Information-Theoretic Metrics

- **PMI** (Pointwise Mutual Information): $\log \frac{P(x,y)}{P(x)P(y)}$
- **PMI²**, **PMI³**: Squared and cubed variants
- **PPMI**: Positive PMI (negative values set to 0)
- **LLR**: Log-likelihood ratio
- **LexicalGravity**: Asymmetric association measure

### Statistical Metrics

- **ChiSquare**: Pearson's χ² test
- **Tscore**, **Zscore**: Statistical significance tests
- **PhiCoef**: Phi coefficient (φ)
- **CramersV**: Cramér's V
- **YuleQ**, **YuleOmega**: Yule's measures

### Similarity Coefficients

- **Dice**: $\frac{2a}{2a + b + c}$
- **LogDice**: Logarithmic Dice (more stable)
- **JaccardIdx**: Jaccard similarity
- **CosineSim**: Cosine similarity
- **OverlapCoef**: Overlap coefficient

### Epidemiological Metrics

- **RelRisk**, **LogRelRisk**: Relative risk measures
- **OddsRatio**, **LogOddsRatio**: Odds ratios
- **RiskDiff**: Risk difference
- **AttrRisk**: Attributable risk

### Complete Metric List

<details>
<summary>Click to see all 47 metrics with formulas</summary>

| Metric            | Type          | Formula                                              |
| ----------------- | ------------- | ---------------------------------------------------- |
| PMI               | `PMI`         | $\log \frac{P(x,y)}{P(x)P(y)}$                       |
| PMI²              | `PMI²`        | $(\log \frac{P(x,y)}{P(x)P(y)})^2$                   |
| PMI³              | `PMI³`        | $(\log \frac{P(x,y)}{P(x)P(y)})^3$                   |
| PPMI              | `PPMI`        | $\max(0, \log \frac{P(x,y)}{P(x)P(y)})$              |
| LLR               | `LLR`         | $2 \sum_{i,j} O_{ij} \ln \frac{O_{ij}}{E_{ij}}$      |
| LLR²              | `LLR²`        | $\sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$      |
| Dice              | `Dice`        | $\frac{2a}{2a + b + c}$                              |
| LogDice           | `LogDice`     | $14 + \log_2(\frac{2a}{2a + b + c})$                 |
| Jaccard           | `JaccardIdx`  | $\frac{a}{a + b + c}$                                |
| Cosine            | `CosineSim`   | $\frac{a}{\sqrt{(a + b)(a + c)}}$                    |
| Overlap           | `OverlapCoef` | $\frac{a}{\min(a + b, a + c)}$                       |
| Relative Risk     | `RelRisk`     | $\frac{a/(a+b)}{c/(c+d)}$                            |
| Odds Ratio        | `OddsRatio`   | $\frac{ad}{bc}$                                      |
| Chi-square        | `ChiSquare`   | $\sum_{i,j}\frac{(f_{ij}-\hat{f_ij})^2}{\hat{f_ij}}$ |
| Phi               | `PhiCoef`     | $\frac{ad - bc}{\sqrt{(a+b)(c+d)(a+c)(b+d)}}$        |
| Cramér's V        | `CramersV`    | $\sqrt{\frac{\chi^2}{n \cdot \min(r-1, c-1)}}$       |
| _...and 35+ more_ |               |                                                      |

</details>

## 🎯 Advanced Features

### Temporal Analysis

Track how word associations change over time:

```julia
temporal_analysis = analyze_temporal(
    corpus, ["pandemic", "vaccine"], :year, PMI, time_bins=5
)
```

### Subcorpus Comparison

Compare associations across document groups with statistical tests:

```julia
comparison = compare_subcorpora(
    corpus, :category, "innovation", PMI
)
# Access statistical tests and effect sizes
tests = comparison.statistical_tests
```

### Collocation Networks

Build and export word association networks:

```julia
network = colloc_graph(
    corpus, ["climate", "change"],
    metric=PMI, depth=2, min_score=3.0
)
gephi_graph(network, "nodes.csv", "edges.csv")
```

### Keyword Extraction

```julia
keywords = keyterms(corpus, method=:tfidf, num_keywords=50)
```

### Concordance (KWIC)

```julia
concordance = kwic(corpus, "innovation", context_size=50)
for line in concordance.lines
    println("...$(line.LeftContext) [$(line.Node)] $(line.RightContext)...")
end
```

## ⚡ Performance Features

### Parallel Processing

```julia
# Use multiple cores
using Distributed
addprocs(4)

analysis = analyze_nodes(
    corpus, nodes, metrics, parallel=true
)
```

### Streaming for Large Corpora

```julia
# Process files without loading everything into memory
results = stream_corpus_analysis(
    "texts/*.txt", "word", PMI, chunk_size=1000
)
```

### Batch Processing

```julia
# Process hundreds of node words efficiently
batch_process_corpus(
    corpus, "nodelist.txt", "output/",
    batch_size=100
)
```

## 🔬 Use Cases

`TextAssociations.jl` is ideal for:

- **Corpus Linguistics**: Collocation analysis, lexical patterns, semantic prosody
- **Digital Humanities**: Literary analysis, historical text mining, stylometry
- **NLP Research**: Feature extraction, baseline models, evaluation metrics
- **Social Media Analysis**: Trend detection, sentiment associations, hashtag networks
- **Information Retrieval**: Query expansion, document similarity, term weighting

## 📖 Documentation

- [Getting Started Guide](https://atantos.github.io/TextAssociations.jl/)
- [API Reference](https://atantos.github.io/TextAssociations.jl/api/)
<!-- - [Metric Formulas](https://atantos.github.io/TextAssociations.jl/dev/metrics) -->
- [Examples](https://atantos.github.io/TextAssociations.jl/api/functions/#Examples)

## 💻 Example Workflows

### Research Paper Analysis

```julia
# Load abstracts from CSV
corpus = read_corpus("papers.csv",
    text_column=:abstract,
    metadata_columns=[:year, :journal])

# Extract domain-specific keywords
keywords = keyterms(corpus, method=:tfidf, num_keywords=100)

# Analyze key terms over time
temporal = analyze_temporal(
    corpus, keywords[1:10], :year, PMI
)

# Compare across journals
comparison = compare_subcorpora(corpus, :journal, "methodology", LogDice)
```

### Literary Text Analysis

```julia
# Load novels
corpus = read_corpus("novels/", preprocess=true)

# Character co-occurrence network
characters = ["Elizabeth", "Darcy", "Jane", "Bingley"]
network = colloc_graph(
    corpus, characters, windowsize=20
)

# Export for visualization
gephi_graph(network, "characters.csv", "relations.csv")
```

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](https://github.com/atantos/TextAssociations.jl/discussions) for details on:

- Adding new metrics
- Improving performance
- Extending functionality
- Reporting issues

### Development Setup

```julia
# Clone repository
git clone https://github.com/atantos/TextAssociations.jl
cd TextAssociations.jl

# Activate environment
julia --project=.

# Run tests
using Pkg; Pkg.test()
```

## 🙏 Acknowledgments

`TextAssociations.jl` builds on established methods from computational linguistics and is inspired by:

- **AntConc** (Anthony, 2022)
- **SketchEngine** (Kilgarriff et al., 2014)
- **WordSmith Tools** (Scott, 2020)

While offering the performance and flexibility of the Julia ecosystem.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🗺️ Roadmap

- [ ] GPU acceleration for large-scale processing
- [ ] Additional keyword extraction methods (TextRank, RAKE)
- [ ] Integration with word embeddings
- [ ] Indexing & Search Engine (à la _Corpus Workbench_)
- [ ] Support for more file formats (XML, CONLL)

## ❓ FAQ

<details>
<summary><b>How does this compare to other tools?</b></summary>

| Feature             | TextAssociations.jl     | AntConc           | SketchEngine      | WordSmith         |
| ------------------- | ----------------------- | ----------------- | ----------------- | ----------------- |
| Open Source         | ✅                      | ✅                | ❌                | ❌                |
| Metrics             | 47                      | ~10\*<sup>2</sup> | ~20\*<sup>2</sup> | ~15\*<sup>2</sup> |
| Corpus Size         | Unlimited\*<sup>1</sup> | Limited           | Large             | Medium            |
| Parallel Processing | ✅                      | ❌                | ✅                | ❌                |
| API Access          | ✅                      | ❌                | ✅                | ❌                |
| Programmable        | ✅                      | ❌                | Limited           | ❌                |

\*<sup>1</sup> With streaming and memory-mapped files

\*<sup>2</sup> This is a rough estimate including both association measures and keyness tests. A more precise count from users of these tools is welcome.

</details>

<details>
<summary><b>What file formats are supported?</b></summary>

- Plain text files (.txt)
- CSV files with text columns
- JSON files
- Julia DataFrames
- Directory of text files
<!-- - Compressed archives (.gz) -->

</details>

<details>
<summary><b>Can it handle non-English text?</b></summary>

Yes! `TextAssociations.jl` works with any Unicode text. The preprocessing steps (lowercasing, punctuation removal) are Unicode-aware.

</details>

---

**📬 Contact**: For questions and support, please open an issue on [GitHub](https://github.com/atantos/TextAssociations.jl/issues).

**🌟 Star us on GitHub**: If you find this package useful, please consider giving it a star!
