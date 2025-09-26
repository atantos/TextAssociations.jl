# Choosing Metrics

```@meta
CurrentModule = TextAssociations
```

Selecting the right association metric is crucial for meaningful results. This guide helps you choose metrics based on your research goals and data characteristics.

## Quick Selection Guide

```@example quick_guide
using TextAssociations, DataFrames

# Quick reference for metric selection
metric_guide = DataFrame(
    Goal = [
        "Find rare but meaningful associations",
        "Validate known collocations",
        "Compare across different corpora",
        "Identify fixed expressions",
        "Statistical significance testing",
        "Symmetric word similarity"
    ],
    RecommendedMetrics = [
        "PMI, PPMI",
        "LLR, Chi-square",
        "LogDice, PPMI",
        "Dice, MI",
        "LLR, Chi-square, T-score",
        "Dice, Jaccard, Cosine"
    ],
    Reason = [
        "High PMI for rare co-occurrences",
        "Statistical tests for reliability",
        "Stable across corpus sizes",
        "High scores for fixed phrases",
        "P-values for hypothesis testing",
        "Symmetric similarity measures"
    ]
)

println("Metric Selection Guide:")
for row in eachrow(metric_guide)
    println("\n$(row.Goal):")
    println("  Use: $(row.RecommendedMetrics)")
    println("  Why: $(row.Reason)")
end
```

## Metric Properties Comparison

### Information-Theoretic Metrics

```@example info_metrics
using TextAssociations, DataFrames

text = """
Quantum computing revolutionizes computational power.
Classical computing cannot match quantum supremacy.
Quantum algorithms solve complex problems efficiently.
"""

ct = ContingencyTable(text, "quantum"; windowsize=3, minfreq=1)
results = assoc_score([PMI, PMI², PMI³, PPMI], ct)

println("PMI Family Comparison:")
for row in eachrow(results)
    println("\n$(row.Collocate):")
    println("  PMI:  $(round(row.PMI, digits=2)) - Standard measure")
    println("  PMI²: $(round(row.PMI², digits=2)) - Emphasizes frequency")
    println("  PMI³: $(round(row.PMI³, digits=2)) - Strong frequency bias")
    println("  PPMI: $(round(row.PPMI, digits=2)) - No negative values")
end
```

### Statistical Significance Metrics

```@example stat_metrics
using TextAssociations, DataFrames

# Text with clear patterns
text = """
Statistical analysis requires statistical methods and statistical tools.
Random words appear randomly without random patterns.
Data analysis needs careful analysis of data patterns.
"""

ct = ContingencyTable(text, "statistical"; windowsize=4, minfreq=1)
results = assoc_score([LLR, ChiSquare, Tscore, Zscore], ct)

println("Statistical Tests Comparison:")
for row in eachrow(results)
    llr_sig = row.LLR > 10.83 ? "p<0.001" : row.LLR > 6.63 ? "p<0.01" : row.LLR > 3.84 ? "p<0.05" : "n.s."
    chi_sig = row.ChiSquare > 10.83 ? "p<0.001" : row.ChiSquare > 6.63 ? "p<0.01" : row.ChiSquare > 3.84 ? "p<0.05" : "n.s."

    println("\n$(row.Collocate):")
    println("  LLR: $(round(row.LLR, digits=2)) ($llr_sig)")
    println("  χ²:  $(round(row.ChiSquare, digits=2)) ($chi_sig)")
    println("  t:   $(round(row.Tscore, digits=2))")
    println("  z:   $(round(row.Zscore, digits=2))")
end
```

### Similarity Metrics

```@example sim_metrics
using TextAssociations, DataFrames

text = """
Machine learning and deep learning share similar foundations.
Neural networks enable deep learning applications.
Learning algorithms power machine learning systems.
"""

ct = ContingencyTable(text, "learning"; windowsize=3, minfreq=1)
results = assoc_score([Dice, LogDice, JaccardIdx, CosineSim], ct)

println("Similarity Metrics Comparison:")
for row in eachrow(results)
    println("\n$(row.Collocate):")
    println("  Dice:    $(round(row.Dice, digits=3)) ∈ [0,1]")
    println("  LogDice: $(round(row.LogDice, digits=2)) ∈ [0,14]")
    println("  Jaccard: $(round(row.JaccardIdx, digits=3)) ∈ [0,1]")
    println("  Cosine:  $(round(row.CosineSim, digits=3)) ∈ [0,1]")
end
```

## Metric Behavior Analysis

### Frequency Sensitivity

```@example freq_sensitivity
using TextAssociations, DataFrames

# Create texts with different frequency patterns
high_freq = "the the the word the the the"
low_freq = "rare unique word special unusual"

function analyze_frequency_sensitivity(text::String, node::String)
    ct = ContingencyTable(text, node; windowsize=2, minfreq=1)

    metrics = [PMI, LogDice, LLR, Dice]
    results = assoc_score(metrics, ct)

    return results
end

println("High frequency context:")
high_results = analyze_frequency_sensitivity(high_freq, "the")
for row in eachrow(high_results)
    println("  $(row.Collocate): PMI=$(round(row.PMI, digits=2)), LogDice=$(round(row.LogDice, digits=2))")
end

println("\nLow frequency context:")
low_results = analyze_frequency_sensitivity(low_freq, "word")
for row in eachrow(low_results)
    println("  $(row.Collocate): PMI=$(round(row.PMI, digits=2)), LogDice=$(round(row.LogDice, digits=2))")
end
```

### Corpus Size Stability

```@example corpus_stability
using TextAssociations, DataFrames

# Simulate different corpus sizes
small_corpus = "machine learning uses algorithms"
medium_corpus = repeat("machine learning uses algorithms and data ", 10)
large_corpus = repeat("machine learning uses algorithms and data for predictions ", 100)

function compare_corpus_sizes(node::String)
    sizes = [
        ("Small", small_corpus),
        ("Medium", medium_corpus),
        ("Large", large_corpus)
    ]

    println("\nAnalyzing '$node' across corpus sizes:")

    for (size_name, corpus) in sizes
        ct = ContingencyTable(corpus, node; windowsize=3, minfreq=1)
        results = assoc_score([PMI, LogDice, LLR], ct)

        if nrow(results) > 0
            row = first(results)  # Look at first collocate
            println("  $size_name: PMI=$(round(row.PMI, digits=2)), " *
                   "LogDice=$(round(row.LogDice, digits=2)), " *
                   "LLR=$(round(row.LLR, digits=2))")
        end
    end
end

compare_corpus_sizes("learning")
```

## Decision Trees

### For Research Goals

```julia
# Decision tree for metric selection
function recommend_metrics(goal::Symbol)
    recommendations = Dict(
        :discovery => ["Use PMI/PPMI for finding new, surprising associations",
                      "High PMI (>5) indicates strong association",
                      "PPMI removes negative associations"],

        :validation => ["Use LLR for statistical significance",
                       "LLR > 10.83 means p < 0.001",
                       "Combine with effect size (PMI) for importance"],

        :comparison => ["Use LogDice for cross-corpus stability",
                       "LogDice range [0,14] is interpretable",
                       "Less affected by corpus size than PMI"],

        :similarity => ["Use Dice/Jaccard for word similarity",
                       "Both are symmetric measures",
                       "Dice gives more weight to co-occurrences"]
    )

    return get(recommendations, goal, ["Unknown goal"])
end
```

### For Data Characteristics

```@example data_characteristics
using TextAssociations, DataFrames

function recommend_by_data(corpus_size::Symbol, frequency::Symbol, goal::Symbol)
    # Rule-based recommendations
    recommendations = []

    # Corpus size considerations
    if corpus_size == :small
        push!(recommendations, "LogDice (stable for small corpora)")
        push!(recommendations, "Dice (less affected by sparse data)")
    elseif corpus_size == :large
        push!(recommendations, "LLR (better with more data)")
        push!(recommendations, "PMI (meaningful with sufficient data)")
    end

    # Frequency considerations
    if frequency == :rare
        push!(recommendations, "PMI/PPMI (highlights rare associations)")
    elseif frequency == :common
        push!(recommendations, "LogDice (handles high frequency well)")
        push!(recommendations, "LLR (good for common words)")
    end

    # Goal considerations
    if goal == :exploratory
        push!(recommendations, "Multiple metrics for validation")
    elseif goal == :confirmatory
        push!(recommendations, "LLR with significance threshold")
    end

    return unique(recommendations)
end

# Example recommendation
recs = recommend_by_data(:small, :rare, :exploratory)
println("Recommendations for small corpus with rare words (exploratory):")
for rec in recs
    println("  • $rec")
end
```

## Metric Interpretation Guide

### Score Ranges and Thresholds

```@example thresholds
using TextAssociations, DataFrames

# Interpretation thresholds
thresholds = DataFrame(
    Metric = ["PMI", "LogDice", "LLR", "Dice", "Jaccard"],
    WeakAssociation = ["< 2", "< 5", "< 3.84", "< 0.1", "< 0.05"],
    ModerateAssociation = ["2-5", "5-8", "3.84-10.83", "0.1-0.3", "0.05-0.2"],
    StrongAssociation = ["> 5", "> 8", "> 10.83", "> 0.3", "> 0.2"],
    Interpretation = [
        "Higher = stronger",
        "Max 14, stable",
        "Statistical significance",
        "0-1 scale",
        "0-1 scale, stricter"
    ]
)

println("Metric Interpretation Thresholds:")
for row in eachrow(thresholds)
    println("\n$(row.Metric):")
    println("  Weak: $(row.WeakAssociation)")
    println("  Moderate: $(row.ModerateAssociation)")
    println("  Strong: $(row.StrongAssociation)")
    println("  Note: $(row.Interpretation)")
end
```

### Practical Examples

```@example practical
using TextAssociations, DataFrames

# Different types of word relationships
texts = Dict(
    "Fixed expression" => "by and large the results were positive",
    "Technical term" => "machine learning algorithm performs classification",
    "Semantic relation" => "doctor treats patient in hospital",
    "Syntactic relation" => "very important extremely significant quite notable"
)

function analyze_relationship_type(text::String, node::String, collocate::String)
    ct = ContingencyTable(text, node; windowsize=3, minfreq=1)
    results = assoc_score([PMI, LogDice, Dice, LLR], ct)

    # Find specific collocate
    row = filter(r -> String(r.Collocate) == collocate, results)

    if !isempty(row)
        r = first(row)
        println("$node + $collocate:")
        println("  PMI: $(round(r.PMI, digits=2))")
        println("  LogDice: $(round(r.LogDice, digits=2))")
        println("  Dice: $(round(r.Dice, digits=3))")
        println("  LLR: $(round(r.LLR, digits=2))")
    end
end

# Analyze different relationship types
println("Fixed Expression:")
analyze_relationship_type(texts["Fixed expression"], "by", "and")

println("\nTechnical Term:")
analyze_relationship_type(texts["Technical term"], "machine", "learning")

println("\nSemantic Relation:")
analyze_relationship_type(texts["Semantic relation"], "doctor", "patient")
```

## Advanced Metric Selection

### Combining Multiple Metrics

```@example combining
using TextAssociations, DataFrames, Statistics

function combined_score_analysis(ct::ContingencyTable)
    # Calculate multiple metrics
    results = assoc_score([PMI, LogDice, LLR, Dice], ct)

    # Normalize scores (0-1 range)
    for col in [:PMI, :LogDice, :LLR, :Dice]
        if hasproperty(results, col)
            values = results[!, col]
            min_val, max_val = extrema(values)
            if max_val > min_val
                results[!, Symbol(col, :_norm)] = (values .- min_val) ./ (max_val - min_val)
            else
                results[!, Symbol(col, :_norm)] = zeros(length(values))
            end
        end
    end

    # Combined score (weighted average)
    results.CombinedScore = (
        0.3 * results.PMI_norm +
        0.3 * results.LogDice_norm +
        0.2 * results.LLR_norm +
        0.2 * results.Dice_norm
    )

    # Rank by combined score
    sort!(results, :CombinedScore, rev=true)

    return results
end

text = "Data science requires data analysis and data visualization"
ct = ContingencyTable(text, "data"; windowsize=3, minfreq=1)
combined = combined_score_analysis(ct)

println("Combined Metric Analysis:")
for row in eachrow(first(combined, 3))
    println("$(row.Collocate): Combined=$(round(row.CombinedScore, digits=3))")
end
```

### Metric Stability Analysis

```@example stability
using TextAssociations, DataFrames, Statistics

function metric_stability_test(base_text::String, node::String, iterations::Int=10)
    metric_scores = Dict{Symbol,Vector{Float64}}()

    for i in 1:iterations
        # Add noise to simulate variation
        noisy_text = base_text * " " * join(rand(split(base_text), 5), " ")

        ct = ContingencyTable(noisy_text, node; windowsize=3, minfreq=1)
        results = assoc_score([PMI, LogDice, LLR], ct)

        if nrow(results) > 0
            for metric in [:PMI, :LogDice, :LLR]
                push!(get!(metric_scores, metric, Float64[]), results[1, metric])
            end
        end
    end

    # Calculate stability (lower std = more stable)
    println("Metric Stability Analysis ($iterations iterations):")
    for (metric, scores) in metric_scores
        stability = std(scores) / mean(scores)  # Coefficient of variation
        println("  $metric: CV = $(round(stability, digits=3)) ($(stability < 0.1 ? "stable" : "unstable"))")
    end
end

base = "artificial intelligence and machine learning are related fields"
metric_stability_test(base, "intelligence", 10)
```

## Best Practices

### 1. Use Multiple Metrics

```julia
# Always compare multiple perspectives
const COMPREHENSIVE_METRICS = [
    PMI,      # Informativeness
    LogDice,  # Stability
    LLR,      # Significance
    Dice      # Similarity
]
```

### 2. Set Appropriate Thresholds

```julia
# Domain-specific thresholds
const THRESHOLDS = Dict(
    :academic => (pmi=3.0, logdice=7.0, llr=10.83),
    :social_media => (pmi=2.0, logdice=5.0, llr=6.63),
    :technical => (pmi=4.0, logdice=8.0, llr=15.13)
)
```

### 3. Validate with Domain Knowledge

Always verify that high-scoring collocations make sense in your domain:

```julia
function validate_results(results::DataFrame, known_good::Vector{String})
    found = intersect(String.(results.Collocate), known_good)
    coverage = length(found) / length(known_good)

    println("Validation: Found $(length(found))/$(length(known_good)) known collocations")
    println("Coverage: $(round(coverage * 100, digits=1))%")

    return coverage > 0.7  # 70% coverage threshold
end
```

## Next Steps

- Apply metrics in [Working with Corpora](corpus_analysis.md)
- See [Advanced Features](../advanced/temporal.md) for specialized analyses
- Review [API Reference](../api/metrics.md) for all available metrics
