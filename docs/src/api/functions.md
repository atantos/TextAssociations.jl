# Main Functions

```@meta
CurrentModule = TextAssociations
```

This section provides comprehensive documentation for all main API functions in TextAssociations.jl.

## Function Categories

```@contents
Pages = ["functions.md"]
Depth = 2
```

## Core Evaluation Functions

### assoc_score - Primary Evaluation Function

```@docs
assoc_score
```

The `assoc_score` function is the primary interface for computing association metrics. It supports multiple signatures for different use cases.

#### Method Signatures

```julia
# Single metric on prepared data
assoc_score(metric::Type{<:AssociationMetric},
          data::AssociationDataFormat;
          scores_only::Bool=false,
          tokens::Union{Nothing,Vector{String}}=nothing) -> Union{DataFrame, Vector}

# Multiple metrics on prepared data
assoc_score(metrics::Vector{DataType},
          data::AssociationDataFormat;
          scores_only::Bool=false) -> Union{DataFrame, Dict}

# Direct from text (single metric)
assoc_score(metric::Type{<:AssociationMetric},
          text::AbstractString,
          node::AbstractString,
          windowsize::Int,
          minfreq::Int=5;
          scores_only::Bool=false) -> Union{DataFrame, Vector}
```

#### Parameters

| Parameter        | Type                     | Description                                | Default   |
| ---------------- | ------------------------ | ------------------------------------------ | --------- |
| `metric/metrics` | Type or Vector{DataType} | Association metric(s) to compute           | Required  |
| `data`           | AssociationDataFormat    | ContingencyTable or CorpusContingencyTable | Required  |
| `scores_only`    | Bool                     | Return only numeric scores                 | `false`   |
| `tokens`         | Vector{String}           | Token list for metrics that need it        | `nothing` |
| `text`           | AbstractString           | Raw text for direct evaluation             | -         |
| `node`           | AbstractString           | Target word                                | -         |
| `windowsize`     | Int                      | Context window size                        | -         |
| `minfreq`        | Int                      | Minimum frequency threshold                | `5`       |

#### Return Values

- **Default** (`scores_only=false`): Returns `DataFrame` with columns:

  - `Node`: Target word
  - `Collocate`: Co-occurring word
  - `Frequency`: Co-occurrence frequency
  - `[MetricName]`: Score column(s) named after metric(s)

- **Performance mode** (`scores_only=true`):
  - Single metric: `Vector{Float64}` of scores
  - Multiple metrics: `Dict{String, Vector{Float64}}`

#### Examples

##### Basic Usage

```@example assoc_score
using TextAssociations

text = """
Data science combines mathematics, statistics, and computer science.
Machine learning is a crucial part of data science.
Data analysis helps extract insights from data.
"""

# Create contingency table
ct = ContingencyTable(text, "data"; windowsize=3, minfreq=1)

# Single metric evaluation
pmi_results = assoc_score(PMI, ct)
println("PMI Results:")
println(pmi_results)
```

##### Multiple Metrics

```@example assoc_score_multi
# Evaluate multiple metrics simultaneously
metrics = [PMI, LogDice, LLR, Dice]
multi_results = assoc_score(metrics, ct)

println("\nColumns in results: ", names(multi_results))
println("Top result by PMI:")
println(first(sort(multi_results, :PMI, rev=true), 1))
```

##### Direct from Text

```@example assoc_score_direct
# Skip contingency table creation
results = assoc_score(PMI, text, "science", windowsize=4, minfreq=1)
println("\nDirect evaluation results:")
println(results)
```

##### Performance Mode

```@example assoc_score_perf
# Get only scores for better performance
scores = assoc_score(PMI, ct, scores_only=true)
println("\nScore vector: ", scores)
println("Length: ", length(scores))

# Multiple metrics with scores_only
score_dict = assoc_score([PMI, LogDice], ct, scores_only=true)
println("\nScore dictionary keys: ", keys(score_dict))
```

#### Advanced Usage

##### Custom Filtering Pipeline

```@example assoc_score_advanced
# Evaluate and filter in one pipeline
function analyze_with_thresholds(text, word, thresholds)
    ct = ContingencyTable(text, word; windowsize=5, minfreq=2)
    results = assoc_score([PMI, LogDice, LLR], ct)

    # Apply multiple thresholds
    filtered = filter(row ->
        row.PMI >= thresholds[:pmi] &&
        row.LogDice >= thresholds[:logdice] &&
        row.LLR >= thresholds[:llr],
        results
    )

    return sort(filtered, :PMI, rev=true)
end

thresholds = Dict(:pmi => 2.0, :logdice => 5.0, :llr => 3.84)
filtered = analyze_with_thresholds(text, "data", thresholds)
println("Filtered results: ", nrow(filtered), " collocates")
```

## Text Processing Functions

### prep_string - Text Preprocessing

```@docs
prep_string
```

Preprocesses text with extensive customization options for different languages and domains.

#### Parameters

| Parameter              | Type           | Description                       | Default  |
| ---------------------- | -------------- | --------------------------------- | -------- |
| `input_path`           | AbstractString | File path, directory, or raw text | Required |
| `strip_punctuation`    | Bool           | Remove punctuation                | `true`   |
| `punctuation_to_space` | Bool           | Replace punctuation with spaces   | `true`   |
| `strip_whitespace`     | Bool           | Remove all whitespace             | `false`  |
| `normalize_whitespace` | Bool           | Collapse multiple spaces          | `true`   |
| `strip_case`           | Bool           | Convert to lowercase              | `true`   |
| `strip_accents`        | Bool           | Remove diacritical marks          | `false`  |
| `unicode_form`         | Symbol         | Unicode normalization form        | `:NFC`   |
| `use_prepare`          | Bool           | Apply TextAnalysis pipeline       | `false`  |

#### Examples

##### Basic Preprocessing

```@example prep_string
using TextAssociations

# Default preprocessing
text = "Hello, WORLD!!! Multiple   spaces..."
doc = prep_string(text)
println("Default: '", text(doc), "'")

# Custom preprocessing
# doc_custom = prep_string(text,
#     strip_case=false,        # Keep original case
#     strip_punctuation=false, # Keep punctuation
#     normalize_whitespace=true # Fix spacing only
# )
println("Custom: '", text(doc_custom), "'")
```

##### Multilingual Text

```@example prep_string_multi
# Greek text with diacritics
greek = "Καλημέρα! Η ανάλυση κειμένου είναι σημαντική."

# Keep diacritics (default)
doc_with = prep_string(greek, strip_accents=false)
println("With accents: '", text(doc_with), "'")

# Remove diacritics
doc_without = prep_string(greek, strip_accents=true)
println("Without accents: '", text(doc_without), "'")
```

##### Processing Files and Directories

```@example prep_string_files
# From file
# doc = prep_string("document.txt")

# From directory (concatenates all .txt files)
# doc = prep_string("corpus/")

# Example with temporary file
using Mmap
temp_file = tempname() * ".txt"
write(temp_file, "Sample text from file.")
doc = prep_string(temp_file)
println("From file: '", text(doc), "'")
rm(temp_file)
```

### build_vocab - Vocabulary Creation

```@docs
build_vocab
```

Creates an ordered dictionary mapping words to indices.

#### Parameters

- `input`: Either a `StringDocument` or `Vector{String}`

#### Returns

- `OrderedDict{String,Int}`: Word-to-index mapping

#### Examples

```@example vocab
using TextAssociations

# From document
doc = prep_string("The quick brown fox jumps over the lazy dog")
vocab = build_vocab(doc)

println("Vocabulary size: ", length(vocab))
println("First 5 words:")
for (word, idx) in Iterators.take(vocab, 5)
    println("  $idx: '$word'")
end

# From word vector
words = ["apple", "banana", "cherry", "apple"]  # Duplicates removed
vocab2 = build_vocab(words)
println("\nUnique words: ", length(vocab2))
```

## Utility Functions

### available_metrics - List Available Metrics

```@docs
available_metrics
```

Returns a vector of all available association metric symbols.

#### Example

```@example available_metrics
using TextAssociations

metrics = available_metrics()
println("Total available metrics: ", length(metrics))
println("\nInformation-theoretic metrics:")
info_metrics = filter(m -> occursin("PMI", String(m)) || m == :PPMI, metrics)
println(info_metrics)

println("\nStatistical metrics:")
stat_metrics = filter(m -> m in [:LLR, :ChiSquare, :Tscore, :Zscore], metrics)
println(stat_metrics)
```

### cached_data - Access Lazy Data

```@docs
cached_data
```

Extracts data from a `LazyProcess`, computing it if necessary.

#### Parameters

- `z::LazyProcess`: Lazy process wrapper

#### Returns

- The computed/cached result

#### Example

```@example lazy
using TextAssociations

# ContingencyTable uses lazy evaluation internally
ct = ContingencyTable("sample text", "text"; windowsize=3, minfreq=1)

# First access computes the table
println("First access...")
data1 = cached_data(ct.con_tbl)

# Second access uses cache (no computation)
println("Second access...")
data2 = cached_data(ct.con_tbl)

println("Same object? ", data1 === data2)  # true - same cached object
```

### document - Access Document

```@docs
document
```

Extracts the document from a `LazyInput` wrapper.

#### Parameters

- `input::LazyInput`: Lazy input wrapper

#### Returns

- `StringDocument`: The stored document

## Batch Processing Functions

### Processing Multiple Nodes

```@example batch
using TextAssociations

text = """
Artificial intelligence and machine learning are transforming technology.
Deep learning, a subset of machine learning, uses neural networks.
Machine learning algorithms can learn from data without explicit programming.
"""

# Analyze multiple words
nodes = ["learning", "machine", "neural", "data"]
results = Dict{String, DataFrame}()

for node in nodes
    ct = ContingencyTable(text, node; windowsize=3, minfreq=1)
    results[node] = assoc_score(PMI, ct)
end

println("Results per node:")
for (node, df) in results
    println("  $node: $(nrow(df)) collocates, top PMI = $(maximum(df.PMI))")
end
```

### Comparative Analysis

```@example comparative
using TextAssociations
using DataFrames

# Compare different window sizes
function compare_parameters(text, word)
    params = [
        (window=2, minfreq=1),
        (window=5, minfreq=1),
        (window=10, minfreq=1)
    ]

    comparison = DataFrame()
    for p in params
        ct = ContingencyTable(text, word; windowsize=p.window, minfreq=p.minfreq)
        df = assoc_score(PMI, ct)
        df.WindowSize .= p.window
        append!(comparison, df)
    end

    return comparison
end

comparison = compare_parameters(text, "learning")
grouped = groupby(comparison, :WindowSize)
summary = combine(grouped,
    nrow => :NumCollocates,
    :PMI => mean => :AvgPMI,
    :PMI => maximum => :MaxPMI
)
println("\nWindow size comparison:")
println(summary)
```

## Performance Optimization

### Memory-Efficient Processing

```@example memory
using TextAssociations

# Use scores_only for large-scale processing
function process_many_nodes(text, nodes)
    scores = Dict{String, Vector{Float64}}()

    for node in nodes
        ct = ContingencyTable(text, node; windowsize=5, minfreq=1)
        # Get only scores to save memory
        scores[node] = assoc_score(PMI, ct, scores_only=true)
    end

    return scores
end

nodes = ["intelligence", "artificial", "learning"]
score_dict = process_many_nodes(text, nodes)
println("\nScore vectors per node:")
for (node, scores) in score_dict
    println("  $node: $(length(scores)) scores, max = $(maximum(scores))")
end
```

### Parallel Evaluation

```@example parallel
using TextAssociations

# Function for parallel processing (conceptual)
function parallel_evaluate(strings, word, metrics)
    results = []

    # In practice, use @distributed or Threads.@threads
    for s in strings
        ct = ContingencyTable(s, word; windowsize=5, minfreq=2)
        push!(results, assoc_score(metrics, ct))
    end

    return results
end

# Example with multiple text segments
strings = [
    "Machine learning is powerful.",
    "Deep learning uses neural networks.",
    "Artificial intelligence includes machine learning."
]

results = parallel_evaluate(strings, "learning", [PMI, LogDice])
println("\nResults from $(length(results)) text segments processed")
```

## Error Handling and Validation

### Input Validation

```@example validation
using TextAssociations

# Handle empty or invalid inputs
function safe_evaluate(s, word, metric)
    try
        # Validate inputs
        isempty(s) && throw(ArgumentError("Text cannot be empty"))
        isempty(word) && throw(ArgumentError("Word cannot be empty"))

        ct = ContingencyTable(s, word; windowsize=5, minfreq=1)
        results = assoc_score(metric, ct)

        if isempty(results)
            println("Warning: No collocates found for '$word'")
            return DataFrame()
        end

        return results
    catch e
        println("Error: ", e)
        return DataFrame()
    end
end

# Test with various inputs
println("Valid input:")
valid = safe_evaluate(s, "learning", PMI)
println("  Found $(nrow(valid)) collocates")

println("\nEmpty word:")
empty_word = safe_evaluate(s, "", PMI)

println("\nWord not in text:")
missing = safe_evaluate(s, "quantum", PMI)
```

### Parameter Validation

```@example param_validation
# Validate parameters before processing
function validated_analysis(s, word, windowsize, minfreq)
    # Check window size
    if windowsize < 1
        throw(ArgumentError("Window size must be positive"))
    elseif windowsize > 50
        @warn "Large window size may include noise" windowsize
    end

    # Check minimum frequency
    if minfreq < 1
        throw(ArgumentError("Minimum frequency must be at least 1"))
    elseif minfreq > 100
        @warn "High minimum frequency may exclude valid collocates" minfreq
    end

    ct = ContingencyTable(s, word; windowsize, minfreq)
    return assoc_score(PMI, ct)
end

# Test validation
try
    validated_analysis(s, "learning", -1, 5)
catch e
    println("Caught error: ", e)
end

results = validated_analysis(s, "learning", 3, 1)
println("Valid analysis: $(nrow(results)) results")
```

## Integration Examples

### Complete Analysis Pipeline

```@example pipeline
using TextAssociations
using DataFrames

function comprehensive_analysis(s, target_word)
    # Step 1: Preprocess
    doc = prep_string(s,
        strip_punctuation=true,
        strip_case=true,
        normalize_whitespace=true
    )

    # Step 2: Create contingency table
    ct = ContingencyTable(text(doc), target_word; windowsize=5, minfreq=1)

    # Step 3: Evaluate multiple metrics
    metrics = [PMI, LogDice, LLR, Dice, JaccardIdx]
    results = assoc_score(metrics, ct)

    # Step 4: Add composite score
    results.CompositeScore = (
        results.PMI / maximum(results.PMI) * 0.3 +
        results.LogDice / 14 * 0.3 +
        results.LLR / maximum(results.LLR) * 0.2 +
        results.Dice * 0.1 +
        results.JaccardIdx * 0.1
    )

    # Step 5: Sort by composite score
    sort!(results, :CompositeScore, rev=true)

    return results
end

analysis = comprehensive_analysis(s, "learning")
println("\nTop 3 collocates by composite score:")
for row in eachrow(first(analysis, 3))
    println("  $(row.Collocate): Score = $(round(row.CompositeScore, digits=3))")
end
```

### Export Functions

```@example export
using TextAssociations
using CSV

# Prepare results for export
ct = ContingencyTable(s, "intelligence"; windowsize=5, minfreq=1)
results = assoc_score([PMI, LogDice, LLR], ct)

# Add metadata
metadata!(results, "node", "intelligence", style=:note)
metadata!(results, "window_size", 5, style=:note)
metadata!(results, "min_freq", 1, style=:note)
metadata!(results, "timestamp", now(), style=:note)

# Export to CSV
output_file = tempname() * ".csv"
CSV.write(output_file, results)
println("Results exported to: ", output_file)

# Clean up
rm(output_file)
```

## Function Chaining and Composition

### Using Chain.jl

```@example chain
using TextAssociations
using Chain
using DataFrames

# Chain operations for cleaner code
result = @chain text begin
    prep_string(strip_accents=false)
    text
    ContingencyTable("learning"; windowsize=4, minfreq=1)
    assoc_score([PMI, LogDice], _)
    filter(row -> row.PMI > 2 && row.LogDice > 5, _)
    sort(:PMI, rev=true)
    first(5)
end

println("\nChained analysis result:")
println(result)
```

### Custom Function Composition

```@example compose
# Compose functions for reusable pipelines
preprocess = text -> prep_string(text, strip_case=true, strip_punctuation=true)
analyze = (text, word) -> ContingencyTable(text, word; windowsize=5, minfreq=2)
evaluate = ct -> assoc_score([PMI, LogDice, LLR], ct)
filter_strong = df -> filter(row -> row.PMI > 3 && row.LLR > 10.83, df)

# Use composition
pipeline = text -> begin
    doc = preprocess(text)
    ct = analyze(text(doc), "machine")
    results = evaluate(ct)
    filter_strong(results)
end

final_results = pipeline(text)
println("\nPipeline results: $(nrow(final_results)) strong collocates")
```

## Best Practices

### 1. Parameter Selection

```julia
# Recommended defaults
const DEFAULT_PARAMS = Dict(
    :windowsize => 5,      # Balanced for most applications
    :minfreq => 5,         # Filter noise in medium corpora
    :strip_case => true,   # Standard normalization
    :strip_punctuation => true,
    :normalize_whitespace => true
)
```

### 2. Metric Selection Guide

```julia
# Choose metrics based on goal
const METRIC_GUIDE = Dict(
    "discovery" => [PMI, PPMI],           # Find new associations
    "validation" => [LLR, ChiSquare],     # Test significance
    "comparison" => [LogDice, PPMI],      # Cross-corpus stable
    "similarity" => [Dice, JaccardIdx],   # Measure overlap
    "comprehensive" => [PMI, LogDice, LLR, Dice]  # Multiple perspectives
)
```

### 3. Performance Tips

```julia
# For large-scale processing
function optimized_processing(corpus, nodes, metrics)
    # 1. Reuse contingency tables
    cache = Dict{String, ContingencyTable}()

    # 2. Use scores_only when possible
    # 3. Process in batches
    # 4. Consider parallel processing

    results = Dict()
    for node in nodes
        if !haskey(cache, node)
            cache[node] = ContingencyTable(corpus, node; windowsize=5, minfreq=10)
        end
        results[node] = assoc_score(metrics, cache[node], scores_only=true)
    end

    return results
end
```

## Troubleshooting

### Common Issues and Solutions

| Issue                 | Cause                        | Solution                                     |
| --------------------- | ---------------------------- | -------------------------------------------- |
| Empty results         | Word not in text or too rare | Lower `minfreq`, check preprocessing         |
| Memory error          | Large vocabulary             | Use `scores_only=true`, stream processing    |
| Slow performance      | Large corpus or window       | Reduce window size, increase minfreq         |
| Unexpected collocates | Preprocessing issues         | Check `strip_accents`, `strip_case` settings |

### Debug Helper

```@example debug
function debug_analysis(text, word, windowsize, minfreq)
    println("Debug Analysis for '$word'")
    println("="^40)

    # Check preprocessing
    doc = prep_string(text)
    tokens = TextAnalysis.tokens(doc)
    println("Total tokens: ", length(tokens))
    println("Unique tokens: ", length(unique(tokens)))
    println("Word frequency: ", count(==(lowercase(word)), tokens))

    # Check contingency table
    ct = ContingencyTable(text(doc), word; windowsize, minfreq)
    data = cached_data(ct.con_tbl)
    println("Contingency table rows: ", nrow(data))

    if !isempty(data)
        println("Frequency range: ", minimum(data.a), " - ", maximum(data.a))
    end

    # Check results
    results = assoc_score(PMI, ct)
    println("Final results: ", nrow(results), " collocates")

    return results
end

debug_results = debug_analysis(text, "learning", 3, 1)
```

## See Also

- [Core Types](@ref): Type definitions and structures
- [Corpus Functions](@ref): Corpus-level operations
- [Metrics Guide](@ref): Detailed metric descriptions
- [Examples](@ref): More usage examples
- [API Reference](@ref): Complete API documentation
