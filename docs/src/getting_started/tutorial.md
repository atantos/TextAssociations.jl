# [Quick Tutorial](@id getting_started_tutorial)

```@meta
CurrentModule = TextAssociations
DocTestSetup = quote
    using TextAssociations
end
```

This tutorial provides a comprehensive introduction to TextAssociations.jl. By the end, you'll understand how to analyze word associations in text and interpret the results.

## Prerequisites

This tutorial assumes you have:

- Julia 1.9 or later installed
- TextAssociations.jl installed (`using Pkg; Pkg.add("TextAssociations")`)
- Basic familiarity with Julia

## Your First Analysis

Let's start with a simple example analyzing word associations in text about technology.

### Step 1: Load the Package

```@example tutorial
using TextAssociations
using DataFrames
```

### Step 2: Prepare Your Text

```@example tutorial
docs = [
    "Machine learning algorithms can learn from data without explicit programming.",
    "Deep learning is a subset of machine learning that uses neural networks.",
    "Artificial intelligence includes machine learning and deep learning techniques.",
    "Neural networks are the foundation of modern deep learning systems."
]

# Combine documents into one text
s = join(docs, " ")
println("Text length: ", length(s), " characters")
```

### Step 3: Create a Contingency Table

The contingency table captures co-occurrence patterns between your target word and its context.

```@example tutorial
# Analyze the word "learning"
ct = ContingencyTable(
    s,
    "learning";
    windowsize=3,  # Consider 3 words on each side
    minfreq=1      # Include words appearing at least once
)

println("Contingency table created for 'learning'")
```

**Parameters explained:**

- `windowsize=3`: Looks 3 words left and right of "learning"
- `minfreq=1`: Only includes words that appear at least once as collocates

### Step 4: Calculate Association Scores

Now let's calculate PMI (Pointwise Mutual Information) scores to identify strong collocates.

```@example tutorial
# Calculate PMI scores
results = assoc_score(PMI, ct)

println("\nTop 5 collocates of 'learning':")
println(first(sort(results, :PMI, rev=true), 5))
```

### Step 5: Try Multiple Metrics

Different metrics reveal different aspects of associations.

```@example tutorial
# Calculate multiple metrics at once
multi_results = assoc_score([PMI, LogDice, LLR], ct)

println("\nTop 3 collocates with multiple metrics:")
println(first(sort(multi_results, :PMI, rev=true), 3))
```

## Working with Corpora

For analyzing multiple documents, use the Corpus functionality.

```@example tutorial
using TextAnalysis: StringDocument

# Create a simple corpus directly
doc_objects = [StringDocument(d) for d in docs]
corpus = Corpus(doc_objects)

# Analyze "learning" across the corpus
corpus_results = analyze_node(
    corpus,
    "learning",     # Node word
    PMI,           # Metric
    windowsize=3,   # Context window
    minfreq=2      # Min frequency across corpus
)

println("Top collocates of 'learning' in corpus:")
println(first(corpus_results, 5))
```

## Understanding the Results

### Interpreting PMI Scores

PMI (Pointwise Mutual Information) measures how much more likely two words co-occur than by chance:

- **PMI > 0**: Words co-occur more than expected (positive association)
- **PMI = 0**: Co-occurrence matches random expectation
- **PMI < 0**: Words co-occur less than expected (negative association)

```@example tutorial
# Filter for strong associations
strong_assoc = filter(row -> row.PMI > 3.0, results)
println("\nStrong associations (PMI > 3):")
println(strong_assoc)
```

### Comparing Metrics

Different metrics highlight different aspects:

```@example tutorial
# Create comparison
comparison = assoc_score([PMI, LogDice, Dice], ct)

println("\nMetric comparison for top collocate:")
if nrow(comparison) > 0
    top_row = first(sort(comparison, :PMI, rev=true))
    println("  Collocate: ", top_row.Collocate)
    println("  PMI: ", round(top_row.PMI, digits=2))
    println("  LogDice: ", round(top_row.LogDice, digits=2))
    println("  Dice: ", round(top_row.Dice, digits=3))
end
```

## Text Preprocessing

Control how text is normalized before analysis:

```@example tutorial
using TextAnalysis: text

# Example with case-sensitive analysis
text_mixed = "Machine Learning and machine learning are related. Machine learning is powerful."

# Default: case normalization ON
config_lower = TextNorm(strip_case=true)
ct_lower = ContingencyTable(text_mixed, "learning";
    windowsize=3, minfreq=1, norm_config=config_lower)

# Case-sensitive: case normalization OFF
config_case = TextNorm(strip_case=false)
ct_case = ContingencyTable(text_mixed, "learning";
    windowsize=3, minfreq=1, norm_config=config_case)

println("Lowercase normalization: ", nrow(assoc_score(PMI, ct_lower)), " collocates")
println("Case-sensitive: ", nrow(assoc_score(PMI, ct_case)), " collocates")
```

### Preprocessing Options

```@example tutorial
# Full preprocessing configuration
full_config = TextNorm(
    strip_case=true,              # Convert to lowercase
    strip_punctuation=true,        # Remove punctuation
    strip_accents=false,           # Keep diacritics
    normalize_whitespace=true,     # Collapse multiple spaces
    unicode_form=:NFC              # Unicode normalization
)

# Apply preprocessing
preprocessed = prep_string(s, full_config)
println("Preprocessed text (first 100 chars):")
println(first(text(preprocessed), 100), "...")
```

## Common Workflows

### 1. Find Strong Collocations

```@example tutorial
function find_strong_collocations(text, word, threshold=3.0)
    ct = ContingencyTable(text, word; windowsize=5, minfreq=2)
    results = assoc_score([PMI, LogDice], ct)

    # Filter for strong associations
    strong = filter(row -> row.PMI > threshold, results)
    sort!(strong, :PMI, rev=true)

    return strong
end

collocations = find_strong_collocations(s, "learning")
println("\nStrong collocations found: ", nrow(collocations))
```

### 2. Compare Multiple Words

```@example tutorial
function compare_words(text, words, metric=PMI)
    all_results = DataFrame()

    for word in words
        ct = ContingencyTable(text, word; windowsize=5, minfreq=1)
        word_results = assoc_score(metric, ct)
        word_results.Node .= word
        append!(all_results, word_results)
    end

    return all_results
end

comparison_results = compare_words(s, ["learning", "neural", "deep"])
println("\nComparison across words:")
println(first(sort(comparison_results, :PMI, rev=true), 10))
```

### 3. Parameter Tuning

```@example tutorial
function tune_parameters(text, word)
    configs = [
        (ws=2, mf=1, name="Narrow"),
        (ws=5, mf=2, name="Balanced"),
        (ws=10, mf=3, name="Wide")
    ]

    for config in configs
        ct = ContingencyTable(text, word;
            windowsize=config.ws, minfreq=config.mf)
        tune_results = assoc_score(PMI, ct)
        println("$(config.name): $(nrow(tune_results)) collocates")
    end
end

println("\nParameter tuning for 'learning':")
tune_parameters(s, "learning")
```

## Next Steps

Now that you understand the basics, explore:

- **[Metrics Guide](../metrics/overview.md)**: Learn about all available metrics
- **[Corpus Analysis](../guide/corpus_analysis.md)**: Advanced corpus techniques
- **[Preprocessing](../guide/preprocessing.md)**: Detailed text normalization
- **[API Reference](../api/functions.md)**: Complete function documentation

## Quick Reference

### Basic Analysis Pattern

```julia
# 1. Load package
using TextAssociations

# 2. Create contingency table
ct = ContingencyTable(s, "word"; windowsize=5, minfreq=2)

# 3. Calculate scores
results = assoc_score(PMI, ct)

# 4. Examine results
sort!(results, :PMI, rev=true)
```

### Common Parameters

| Parameter           | Typical Range | Description                     |
| ------------------- | ------------- | ------------------------------- |
| `windowsize`        | 2-10          | Context window size             |
| `minfreq`           | 1-5           | Minimum co-occurrence frequency |
| `strip_case`        | true/false    | Convert to lowercase            |
| `strip_punctuation` | true/false    | Remove punctuation              |

### Recommended Metrics

- **PMI**: General-purpose, interpretable
- **LogDice**: Balanced, less affected by frequency
- **LLR**: Statistical significance testing
- **Dice**: Simple similarity measure

## Troubleshooting

### No Results Found

```julia
# Check if word exists in text
doc = prep_string(s, TextNorm(strip_case=true))
tokens = TextAnalysis.tokens(doc)
word_count = count(==("yourword"), tokens)
println("Word appears $word_count times")
```

### Empty DataFrame

Possible causes:

1. `minfreq` too high - try `minfreq=1`
2. `windowsize` too small - try `windowsize=10`
3. Word not in text - check spelling and case
4. Text too short - need more context

### Memory Issues

```julia
# Use scores_only for large analyses
scores = assoc_score(PMI, ct, scores_only=true)  # Returns Vector{Float64}
```

## Practice Exercises

1. Analyze your own text data
2. Compare different window sizes
3. Try all available metrics
4. Build a collocate extraction pipeline
5. Analyze a corpus of documents

Happy analyzing!
