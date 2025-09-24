# Quick Tutorial

```@meta
CurrentModule = TextAssociations
DocTestSetup = quote
    using TextAssociations
end
```

This tutorial will walk you through the basic workflow of TextAssociations.jl in about 10 minutes.

## Step 1: Basic Collocation Analysis

Let's start with a simple example to find collocations of a word:

```@example tutorial
using TextAssociations

# Sample text (you can also load from a file)
text = """
Data science is an interdisciplinary field that uses scientific methods,
processes, algorithms and systems to extract knowledge from data.
Machine learning is a key component of data science.
Data scientists use various tools for data analysis and data visualization.
"""

# Create a contingency table for the word "data"
ct = ContingencyTable(
    text,           # Input text
    "data",         # Node word (target)
    windowsize=3,   # Look 3 words left/right
    minfreq=1       # Minimum frequency
)

# Calculate PMI (Pointwise Mutual Information)
results = assoc_score(PMI, ct)
println(results)
```

## Step 2: Understanding the Results

The results DataFrame contains:

- **Node**: The target word we're analyzing
- **Collocate**: Words that co-occur with the node
- **Frequency**: How often they co-occur
- **PMI**: The association score

Let's interpret the scores:

```@example tutorial
# Sort by PMI score
using DataFrames
sorted_results = sort(results, :PMI, rev=true)
println("Top collocations by PMI:")
for row in eachrow(first(sorted_results, 5))
    println("  $(row.Collocate): PMI = $(round(row.PMI, digits=2))")
end
```

## Step 3: Comparing Multiple Metrics

Different metrics capture different aspects of word association:

```@example tutorial
# Evaluate multiple metrics at once
metrics = [PMI, LogDice, LLR, Dice]
multi_results = assoc_score(metrics, ct)

# View the first few rows
println("Multiple metrics comparison:")
println(first(multi_results, 5))
```

### Understanding Different Metrics

- **PMI**: Measures surprise - high when words occur together more than expected
- **LogDice**: Stable across corpus sizes - good for comparison
- **LLR**: Statistical significance - tests if association is real
- **Dice**: Overlap measure - symmetric similarity

## Step 4: Loading and Analyzing a Corpus

For larger analyses, work with a corpus:

```@example tutorial
# Create a small corpus for demonstration
docs = [
    "Machine learning algorithms learn from data patterns.",
    "Deep learning is a subset of machine learning.",
    "Data science combines statistics and machine learning.",
    "Neural networks power deep learning systems.",
    "Big data requires efficient processing algorithms."
]

# Save as files (in practice, you'd have existing files)
temp_dir = mktempdir()
for (i, doc) in enumerate(docs)
    write(joinpath(temp_dir, "doc$i.txt"), doc)
end

# Load corpus from directory
corpus = read_corpus(temp_dir, preprocess=true, min_doc_length=5)

println("Corpus loaded:")
println("  Documents: ", length(corpus.documents))
println("  Vocabulary size: ", length(corpus.vocabulary))

# Clean up temp directory
rm(temp_dir, recursive=true)
```

## Step 5: Corpus-Level Analysis

Analyze collocations across the entire corpus:

```@example tutorial
# Create a simple corpus directly
using TextAnalysis
doc_objects = [StringDocument(d) for d in docs]
corpus = Corpus(doc_objects)

# Analyze "learning" across the corpus
results = analyze_corpus(
    corpus,
    "learning",     # Node word
    PMI,           # Metric
    windowsize=3,   # Context window
    minfreq=2      # Min frequency across corpus
)

println("Top collocates of 'learning' in corpus:")
println(first(results, 5))
```

## Step 6: Filtering and Interpreting Results

Apply filters to find the most relevant collocations:

```@example tutorial
# Recreate results for filtering example
ct = ContingencyTable(text, "science", 4, 1)
results = assoc_score([PMI, LogDice, LLR], ct)

# Filter for strong collocations
strong_collocations = filter(row ->
    row.PMI > 2.0 &&           # Moderate PMI
    row.LogDice > 5.0 &&        # Reliable collocation
    row.LLR > 3.84,            # Statistically significant (p < 0.05)
    results
)

println("Strong collocations of 'science':")
for row in eachrow(strong_collocations)
    println("  $(row.Collocate): PMI=$(round(row.PMI, digits=2)), ",
            "LogDice=$(round(row.LogDice, digits=2))")
end
```

## Step 7: Exporting Results

Save your results for further analysis or publication:

```@example tutorial
using CSV

# Create results
results = assoc_score([PMI, LogDice], ct)

# Save to CSV
output_file = "collocations.csv"
CSV.write(output_file, results)
println("Results saved to $output_file")

# Clean up
rm(output_file)
```

## Step 8: Visualization Preparation

Prepare data for visualization:

```@example tutorial
# Get top collocations for plotting
top_n = 10
sorted_results = sort(results, :PMI, rev=true)
top_results = first(sorted_results, min(top_n, nrow(sorted_results)))

# Extract data for plotting
words = String.(top_results.Collocate)
scores = top_results.PMI

println("Top $(length(words)) collocations ready for plotting:")
for (word, score) in zip(words, scores)
    println("  $word: $(round(score, digits=2))")
end

# In practice, you would plot with:
# using Plots
# bar(words, scores, xlabel="Collocate", ylabel="PMI Score", rotation=45)
```

## Complete Workflow Example

Here's everything together in a typical workflow:

```@example tutorial
function analyze_text(text::String, target_word::String)
    # 1. Preprocess text
    doc = prep_string(text, strip_case=true, strip_punctuation=true)

    # 2. Create contingency table
    ct = ContingencyTable(text(doc), target_word, windowsize=5, minfreq=2)

    # 3. Calculate multiple metrics
    results = assoc_score([PMI, LogDice, LLR], ct)

    # 4. Filter significant results
    significant = filter(row -> row.LLR > 10.83, results)  # p < 0.001

    # 5. Sort by PMI
    sorted_results = sort(significant, :PMI, rev=true)

    return sorted_results
end

# Use the function
sample_text = """
Natural language processing enables computers to understand human language.
Language models are fundamental to natural language processing.
Modern language models use transformer architectures.
"""

results = analyze_text(sample_text, "language")
println("\nAnalysis complete. Top results:")
println(first(results, 3))
```

## What's Next?

Now that you understand the basics:

1. **Explore different metrics**: See [Metrics Guide](@ref metrics_overview)
2. **Work with larger corpora**: See [Corpus Analysis](@ref corpus_analysis_guide)
3. **Try advanced features**:
   - [Temporal Analysis](@ref advanced_temporal)
   - [Network Building](@ref advanced_networks)
   - [Keyword Extraction](@ref advanced_keywords)

## Practice Exercises

1. **Exercise 1**: Find collocations of "research" with window size 2 vs 5
2. **Exercise 2**: Compare PMI vs LogDice for the same word
3. **Exercise 3**: Find words that collocate with multiple related terms
4. **Exercise 4**: Identify domain-specific terminology using high PMI threshold

## Tips for Beginners

- Start with **LogDice** for reliable results
- Use **window size 5** as a good default
- Set **minfreq** based on corpus size (5 for small, 10+ for large)
- Always compare multiple metrics
- Check concordance lines to verify collocations

## Getting Help

```julia
# Get help on any function
?assoc_score
?ContingencyTable
?analyze_corpus

# List all available metrics
listmetrics()

# Check package version
using Pkg
Pkg.status("TextAssociations")
```

Ready to dive deeper? Continue to [Basic Examples](@ref getting_started_examples) for more use cases.
