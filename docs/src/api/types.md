# Core Types

```@meta
CurrentModule = TextAssociations
```

This section documents all core types in TextAssociations.jl. These types form the foundation of the package's functionality.

## Overview

The type system is organized into several categories:

```@raw html
<div class="type-hierarchy">
<pre>
TextAssociations Types
├── Data Structures
│   ├── ContingencyTable       # Word co-occurrence data
│   ├── Corpus                 # Document collection
│   └── CorpusContingencyTable # Aggregated corpus data
├── Analysis Results
│   ├── MultiNodeAnalysis      # Multiple word analysis
│   ├── TemporalCorpusAnalysis # Time-based analysis
│   ├── SubcorpusComparison    # Comparative analysis
│   ├── CollocationNetwork     # Network representation
│   └── Concordance            # KWIC concordance
├── Abstract Types
│   ├── AssociationMetric      # Base for all metrics
│   └── AssociationDataFormat  # Base for data formats
└── Utility Types
    ├── LazyProcess            # Lazy evaluation wrapper
    └── LazyInput              # Lazy input wrapper
</pre>
</div>
```

## Primary Data Structures

### ContingencyTable

```@docs
ContingencyTable
```

The `ContingencyTable` is the fundamental data structure for word co-occurrence analysis. It stores information about how often words co-occur within a specified window.

#### Constructor

```julia
ContingencyTable(inputstring::AbstractString,
                 node::AbstractString,
                 windowsize::Int,
                 minfreq::Int64=5;
                 auto_prep::Bool=true,
                 strip_accents::Bool=false)
```

#### Parameters

- `inputstring`: Text to analyze (can be raw text, file path, or directory)
- `node`: Target word to analyze
- `windowsize`: Number of words to consider on each side
- `minfreq`: Minimum frequency threshold (default: 5)
- `auto_prep`: Automatically preprocess text (default: true)
- `strip_accents`: Remove diacritical marks (default: false)

#### Fields

- `con_tbl::LazyProcess{T,DataFrame}`: Lazy-loaded contingency data
- `node::AbstractString`: The target word
- `windowsize::Int`: Context window size
- `minfreq::Int64`: Minimum frequency threshold
- `input_ref::LazyInput`: Reference to processed input

#### Example Usage

```@example ct
using TextAssociations

text = """
The field of data science combines statistical analysis with machine learning.
Data scientists use various tools for data visualization and data mining.
Modern data science relies heavily on big data technologies.
"""

# Create contingency table for "data" (use positional args)
ct = ContingencyTable(text, "data", 3, 1)

# The table is computed lazily when first accessed
results = assoc_score(PMI, ct)
println("Found $(nrow(results)) collocates for 'data'")
```

#### Contingency Table Structure

The internal contingency table contains the following values for each word pair:

| Cell | Description             | Formula            |
| ---- | ----------------------- | ------------------ |
| a    | Co-occurrence frequency | f(node, collocate) |
| b    | Node without collocate  | f(node) - a        |
| c    | Collocate without node  | f(collocate) - a   |
| d    | Neither occurs          | N - a - b - c      |
| N    | Total observations      | Total positions    |

---

### Corpus

```@docs
Corpus
```

Represents a collection of documents for corpus-level analysis.

#### Constructor

```julia
Corpus(documents::Vector{StringDocument};
       build_dtm::Bool=false,
       metadata::Dict{String,Any}=Dict())
```

#### Fields

- `documents::Vector{StringDocument}`: Collection of documents
- `metadata::Dict{String,Any}`: Document metadata
- `vocabulary::OrderedDict{String,Int}`: Word-to-index mapping
- `doc_term_matrix::Union{Nothing,SparseMatrixCSC}`: Optional document-term matrix

#### Example Usage

```@example corpus
using TextAssociations
using TextAnalysis: StringDocument  # avoid bringing TextAnalysis.Corpus into scope

# Create corpus from documents
docs = [
    StringDocument("Artificial intelligence is transforming technology."),
    StringDocument("Machine learning is a subset of artificial intelligence."),
    StringDocument("Deep learning uses neural networks.")
]

corpus = TextAssociations.Corpus(docs, metadata=Dict("source" => "AI texts"))

println("Corpus Statistics:")
println("  Documents: ", length(corpus.documents))
println("  Vocabulary size: ", length(corpus.vocabulary))
println("  Metadata: ", collect(keys(corpus.metadata)))
```

---

### CorpusContingencyTable

```@docs
CorpusContingencyTable
```

Aggregates contingency tables across an entire corpus for comprehensive analysis.

#### Constructor

```julia
CorpusContingencyTable(corpus::Corpus,
                       node::AbstractString;
                       windowsize::Int,
                       minfreq::Int64=5;
                       strip_accents::Bool=false)
```

#### Fields

- `tables::Vector{ContingencyTable}`: Individual document tables
- `aggregated_table::LazyProcess`: Lazily computed aggregate
- `node::AbstractString`: Target word
- `windowsize::Int`: Context window
- `minfreq::Int64`: Minimum frequency threshold
- `corpus_ref::Corpus`: Reference to source corpus

## Analysis Result Types

### MultiNodeAnalysis

```@docs
MultiNodeAnalysis
```

Stores results from analyzing multiple node words across a corpus.

#### Fields

- `nodes::Vector{String}`: Analyzed words
- `results::Dict{String,DataFrame}`: Results per node
- `corpus_ref::Corpus`: Source corpus
- `parameters::Dict{Symbol,Any}`: Analysis parameters

#### Example

```julia
# Example (illustrative)
# analysis = MultiNodeAnalysis(
#     ["learning", "intelligence"],
#     Dict("learning" => DataFrame(), "intelligence" => DataFrame()),
#     corpus,
#     Dict(:windowsize => 5, :metric => PMI)
# )
```

---

### TemporalCorpusAnalysis

```@docs
TemporalCorpusAnalysis
```

Results from analyzing word associations over time periods.

#### Fields

- `time_periods::Vector{String}`: Period labels
- `results_by_period::Dict{String,MultiNodeAnalysis}`: Period-specific results
- `trend_analysis::DataFrame`: Trend statistics
- `corpus_ref::Corpus`: Source corpus

---

### SubcorpusComparison

```@docs
SubcorpusComparison
```

Results from comparing word associations between different subcorpora.

#### Fields

- `subcorpora::Dict{String,Corpus}`: Subcorpus divisions
- `node::String`: Analyzed word
- `results::Dict{String,DataFrame}`: Results per subcorpus
- `statistical_tests::DataFrame`: Statistical comparisons
- `effect_sizes::DataFrame`: Effect size calculations

---

### CollocationNetwork

```@docs
CollocationNetwork
```

Network representation of word collocations for visualization and analysis.

#### Fields

- `nodes::Vector{String}`: Network nodes (words)
- `edges::DataFrame`: Edge data with columns [Source, Target, Weight, Metric]
- `node_metrics::DataFrame`: Per-node metrics
- `parameters::Dict{Symbol,Any}`: Network construction parameters

#### Example

```julia
# Example (illustrative)
# using DataFrames
# nodes = ["machine", "learning", "deep", "neural"]
# edges = DataFrame(
#     Source = ["machine", "machine", "deep"],
#     Target = ["learning", "deep", "neural"],
#     Weight = [8.5, 6.2, 7.8],
#     Metric = ["PMI", "PMI", "PMI"]
# )
# node_metrics = DataFrame(
#     Node = nodes,
#     Degree = [2, 1, 2, 1],
#     AvgScore = [7.35, 8.5, 7.0, 7.8]
# )
# network = CollocationNetwork(
#     nodes, edges, node_metrics,
#     Dict(:metric => PMI, :depth => 2)
# )
```

---

### Concordance

```@docs
Concordance
```

KWIC (Key Word In Context) concordance representation.

#### Fields

- `node::String`: Target word
- `lines::DataFrame`: Concordance lines with columns [LeftContext, Node, RightContext, DocId, Position]
- `statistics::Dict{Symbol,Any}`: Occurrence statistics

## Abstract Types

### AssociationMetric

```@docs
AssociationMetric
```

Abstract supertype for all association metrics. All specific metrics (PMI, Dice, LLR, etc.) inherit from this type.

#### Type Hierarchy

```julia
abstract type AssociationMetric <: SemiMetric end

# Concrete subtypes (examples)
abstract type PMI <: AssociationMetric end
abstract type Dice <: AssociationMetric end
abstract type LLR <: AssociationMetric end
# ... more metrics
```

---

### AssociationDataFormat

```@docs
AssociationDataFormat
```

Abstract supertype for data formats used in association computations.

#### Subtypes

- `ContingencyTable`: Single document analysis
- `CorpusContingencyTable`: Corpus-level analysis

## Utility Types

### LazyProcess

```@docs
LazyProcess
```

Enables lazy evaluation with caching for expensive computations.

#### Type Parameters

- `T`: Function type
- `R`: Result type

#### Fields

- `f::T`: Function to compute result
- `cached_result::Union{Nothing,R}`: Cached result
- `cached_process::Bool`: Whether result is cached

#### Example

```@example lazy
using TextAssociations
using DataFrames

# Return a DataFrame so it matches LazyProcess{..., DataFrame}
expensive_df() = DataFrame(x = 1:3, y = [10, 20, 30])

lp = LazyProcess(expensive_df)   # default R = DataFrame

# First call computes the result
result1 = cached_data(lp)

# Second call uses cache
result2 = cached_data(lp)

println("Results equal: ", result1 == result2)
```

---

### LazyInput

```@docs
LazyInput
```

Wrapper for lazily storing and accessing processed input documents.

#### Fields

- `loader::LazyProcess{F,StringDocument}`: Lazy document loader

## Type Traits and Extensions

### Metric Traits

Some metrics require additional information beyond the contingency table:

```julia
# Trait to indicate token requirement
NeedsTokens(::Type{LexicalGravity}) = Val(true)
NeedsTokens(::Type{PMI}) = Val(false)
```

### Custom Types

You can extend the type system with custom metrics:

```julia
# Define custom metric
abstract type MyCustomMetric <: AssociationMetric end

# Implement evaluation function
function eval_mycustommetric(data::AssociationDataFormat)
    # Your implementation
end
```

## Type Conversions and Utilities

### Common Conversions

```julia
# Convert string to StringDocument
doc = StringDocument("your text")

# Convert to ContingencyTable
ct = ContingencyTable(text(doc), "word", 5)

# Extract DataFrame from results
df = assoc_score(PMI, ct)
```

### Type Checking

```julia
# Check if type is an association metric
isa(PMI, Type{<:AssociationMetric})  # true

# Check if data format is valid
isa(ct, AssociationDataFormat)  # true
```

## Performance Considerations

### Memory Usage

| Type                   | Typical Memory Usage   | Notes                               |
| ---------------------- | ---------------------- | ----------------------------------- |
| ContingencyTable       | O(vocab_size)          | Lazy loading reduces initial memory |
| Corpus                 | O(n_docs × avg_length) | Use streaming for large corpora     |
| CorpusContingencyTable | O(vocab_size × n_docs) | Aggregated lazily                   |
| CollocationNetwork     | O(nodes + edges)       | Scales with network size            |

### Optimization Tips

1. **Use lazy evaluation**: Data is computed only when needed.
2. **Reuse contingency tables**: Avoid recreating for multiple metrics.
3. **Stream large corpora**: Use `stream_corpus_analysis()` for memory efficiency.
4. **Cache results**: `LazyProcess` automatically caches computations.

## See Also

- Main Functions — coming soon
- Corpus Functions — coming soon
- Metric Functions — coming soon
- Examples — coming soon
