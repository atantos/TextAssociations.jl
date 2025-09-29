# Metrics Overview

```@meta
CurrentModule = TextAssociations
```

TextAssociations.jl implements over 50 association metrics from various theoretical frameworks. This overview helps you understand the metrics landscape and choose appropriate measures for your analysis.

```@example assoc_ct
using TextAssociations, DataFrames

text = """
Machine learning uses algorithms to find patterns.
Deep learning is a subset of machine learning.
Algorithms process data to extract patterns.
"""
ct = ContingencyTable(text, "statistics"; windowsize=3, minfreq=1)

nothing
```

## Metric Categories

```@example categories
using TextAssociations, DataFrames

# Categorize all available metrics
metrics = available_metrics()

categories = Dict(
    "Information Theoretic" => [:PMI, :PMI², :PMI³, :PPMI],
    "Statistical Tests" => [:LLR, :LLR², :ChiSquare, :Tscore, :Zscore],
    "Similarity Measures" => [:Dice, :LogDice, :JaccardIdx, :CosineSim, :OchiaiIdx],
    "Effect Size" => [:OddsRatio, :LogOddsRatio, :RelRisk, :LogRelRisk],
    "Correlation" => [:PhiCoef, :CramersV, :TschuprowT, :ContCoef],
    "Special Purpose" => [:LexicalGravity, :DeltaPi, :MinSens]
)

println("Metrics by Category:")
for (category, metric_list) in categories
    println("\n$category ($(length(metric_list)) metrics):")
    for metric in metric_list
        if metric in metrics
            println("  ✓ $metric")
        end
    end
end

println("\nTotal implemented: $(length(metrics)) metrics")
```

## Quick Reference Table

| Metric    | Range    | Best For                        | Interpretation                         |
| --------- | -------- | ------------------------------- | -------------------------------------- |
| PMI       | (-∞, +∞) | Finding surprising associations | Higher = stronger association          |
| LogDice   | [0, 14]  | Cross-corpus comparison         | Higher = stronger, stable across sizes |
| LLR       | [0, +∞)  | Statistical significance        | > 10.83 means p < 0.001                |
| Dice      | [0, 1]   | Symmetric similarity            | 1 = perfect overlap                    |
| OddsRatio | [0, +∞)  | Risk assessment                 | > 1 = positive association             |

## Metric Families

### PMI Family

```@example pmi_family
using TextAssociations

text = "Data science uses data analysis and data visualization techniques"
ct = ContingencyTable(text, "data"; windowsize=3, minfreq=1)

# Compare PMI variants
pmi_variants = assoc_score([PMI, PMI², PMI³, PPMI], ct)

println("PMI Family Comparison:")
for row in eachrow(pmi_variants)
    println("\n$(row.Collocate):")
    println("  PMI:  $(round(row.PMI, digits=2)) - Standard")
    println("  PMI²: $(round(row.PMI², digits=2)) - Frequency emphasized")
    println("  PMI³: $(round(row.PMI³, digits=2)) - Strongly frequency-weighted")
    println("  PPMI: $(round(row.PPMI, digits=2)) - Positive only")
end
```

### Statistical Significance Family

```@example assoc_ct
using TextAssociations

# Statistical metrics for hypothesis testing
stat_metrics = assoc_score([LLR, ChiSquare, Tscore, Zscore], ct)

println("\nStatistical Tests:")
for row in eachrow(stat_metrics)
    println("\n$(row.Collocate):")

    # Interpret LLR
    llr_p = row.LLR > 10.83 ? "p < 0.001" :
            row.LLR > 6.63 ? "p < 0.01" :
            row.LLR > 3.84 ? "p < 0.05" : "n.s."
    println("  LLR: $(round(row.LLR, digits=2)) ($llr_p)")

    # Chi-square uses same critical values
    chi_p = row.ChiSquare > 10.83 ? "p < 0.001" :
            row.ChiSquare > 6.63 ? "p < 0.01" :
            row.ChiSquare > 3.84 ? "p < 0.05" : "n.s."
    println("  χ²: $(round(row.ChiSquare, digits=2)) ($chi_p)")

    println("  t-score: $(round(row.Tscore, digits=2))")
    println("  z-score: $(round(row.Zscore, digits=2))")
end
```

## Metric Properties

### Scale and Interpretation

```@example properties
using TextAssociations, DataFrames

# Document metric properties
properties = DataFrame(
    Metric = ["PMI", "LogDice", "Dice", "LLR", "OddsRatio", "Jaccard"],
    Scale = ["(-∞,+∞)", "[0,14]", "[0,1]", "[0,+∞)", "[0,+∞)", "[0,1]"],
    Symmetric = ["No", "No", "Yes", "No", "No", "Yes"],
    FrequencyBias = ["Low freq", "Balanced", "High freq", "Balanced", "Low freq", "Balanced"],
    Stability = ["Low", "High", "Medium", "High", "Low", "Medium"]
)

println("Metric Properties:")
for row in eachrow(properties)
    println("\n$(row.Metric):")
    println("  Scale: $(row.Scale)")
    println("  Symmetric: $(row.Symmetric)")
    println("  Frequency bias: $(row.FrequencyBias)")
    println("  Corpus stability: $(row.Stability)")
end
```

### Computational Complexity

```@example complexity
using TextAssociations

# Measure computation time for different metrics
function benchmark_metrics(ct::ContingencyTable)
    metrics = [PMI, LogDice, LLR, Dice, OddsRatio, JaccardIdx]
    times = Dict{Symbol, Float64}()

    for metric in metrics
        time = @elapsed assoc_score(metric, ct; scores_only=true)
        times[Symbol(metric)] = time * 1000  # Convert to ms
    end

    return times
end

# Create test data
text = repeat("test word pattern ", 100)
ct = ContingencyTable(text, "test"; windowsize=5, minfreq=1)

times = benchmark_metrics(ct)
println("\nComputation times:")
for (metric, time) in sort(collect(times), by=x->x[2])
    println("  $(metric): $(round(time, digits=3))ms")
end
```

## Choosing Metrics by Task

### Discovery vs Validation

```@example task_based
using TextAssociations

# For discovering new associations
discovery_metrics = [PMI, PPMI, OddsRatio]

# For validating known associations
validation_metrics = [LLR, ChiSquare, Zscore]

# For comparing across corpora
comparison_metrics = [LogDice, PPMI, TschuprowT]

println("Task-based metric selection:")
println("\nDiscovery (find new patterns):")
for m in discovery_metrics
    println("  • $m")
end

println("\nValidation (test significance):")
for m in validation_metrics
    println("  • $m")
end

println("\nComparison (across corpora):")
for m in comparison_metrics
    println("  • $m")
end
```

## Metric Relationships

### Correlation Analysis

```@example correlation
using TextAssociations, Statistics

# Calculate multiple metrics and analyze correlations
text = """
Machine learning uses algorithms to find patterns.
Deep learning is a subset of machine learning.
Algorithms process data to extract patterns.
"""

ct = ContingencyTable(text, "learning"; windowsize=4, minfreq=1)
results = assoc_score([PMI, LogDice, LLR, Dice, OddsRatio], ct)

# Calculate pairwise correlations
metrics_to_compare = [:PMI, :LogDice, :LLR, :Dice]
println("Metric Correlations:")

for i in 1:length(metrics_to_compare)-1
    for j in i+1:length(metrics_to_compare)
        m1, m2 = metrics_to_compare[i], metrics_to_compare[j]
        if hasproperty(results, m1) && hasproperty(results, m2)
            correlation = cor(results[!, m1], results[!, m2])
            println("  $m1 vs $m2: $(round(correlation, digits=3))")
        end
    end
end
```

### Metric Equivalences

Some metrics are mathematically related:

```julia
# Mathematical relationships
# LogDice = 14 + log₂(Dice)
# LogOddsRatio = log(OddsRatio)
# CosineSim = Ochiai (for binary data)
# PMI = log(OddsRatio) when d >> a,b,c
```

## Special-Purpose Metrics

### Lexical Gravity

```@example lexical_gravity
using TextAssociations

# Lexical Gravity requires token information
text = "The conference on machine learning featured talks on deep learning and neural networks"

# This metric needs the full token sequence
ct = ContingencyTable(text, "learning"; windowsize=5, minfreq=1)
tokens = String.(split(lowercase(text)))

# Calculate with different formulas
lg_original = assoc_score(LexicalGravity, ct;
    tokens=tokens, formula=:original)
lg_simplified = assoc_score(LexicalGravity, ct;
    tokens=tokens, formula=:simplified)

println("Lexical Gravity variants:")
if !isempty(lg_original)
    for (i, row) in enumerate(eachrow(lg_original))
        simp_score = lg_simplified[i, :LexicalGravity]
        println("  $(row.Collocate): Original=$(round(row.LexicalGravity, digits=2)), " *
               "Simplified=$(round(simp_score, digits=2))")
    end
end
```

### Delta P (Directional Association)

```@example deltap
using TextAssociations

# Delta P measures directional association
text = "cause leads to effect, but effect rarely leads to cause"
ct = ContingencyTable(text, "cause"; windowsize=3, minfreq=1)

deltap = assoc_score(DeltaPi, ct)
println("\nDelta P (directional association):")
for row in eachrow(deltap)
    direction = row.DeltaPi > 0 ? "forward" : "backward"
    println("  $(row.Collocate): $(round(row.DeltaPi, digits=3)) ($direction)")
end
```

## Performance Considerations

### Memory vs Speed Tradeoffs

```julia
# Fast, low memory (single metric)
fast_results = assoc_score(PMI, ct; scores_only=true)

# Slower, more memory (multiple metrics with DataFrame)
full_results = assoc_score([PMI, LogDice, LLR], ct; scores_only=false)

# Batch processing for very large analyses
function batch_metrics(ct, metrics, batch_size=10)
    results = DataFrame()
    for i in 1:batch_size:length(metrics)
        batch = metrics[i:min(i+batch_size-1, length(metrics))]
        batch_results = assoc_score(batch, ct)
        results = hcat(results, batch_results, makeunique=true)
    end
    return results
end
```

## Validation Strategies

### Cross-validation with Multiple Metrics

```@example assoc_ct
using TextAssociations, DataFrames

function validate_associations(ct::ContingencyTable, threshold_dict::Dict)
    # Use multiple metrics for validation
    validation_metrics = [PMI, LogDice, LLR]
    results = assoc_score(validation_metrics, ct)

    # Apply thresholds
    validated = filter(row ->
        row.PMI >= threshold_dict[:PMI] &&
        row.LogDice >= threshold_dict[:LogDice] &&
        row.LLR >= threshold_dict[:LLR],
        results
    )

    println("Validation results:")
    println("  Total collocates: $(nrow(results))")
    println("  After validation: $(nrow(validated))")
    println("  Retention rate: $(round(nrow(validated)/nrow(results)*100, digits=1))%")

    return validated
end

# Example thresholds
thresholds = Dict(:PMI => 3.0, :LogDice => 7.0, :LLR => 10.83)
validated = validate_associations(ct, thresholds)
```

## Next Steps

- Deep dive into [Information Theoretic](information_theoretic.md) metrics
- Explore [Statistical](statistical.md) significance tests
- Learn about [Similarity](similarity.md) measures
- Understand [Epidemiological](epidemiological.md) metrics
- See [API Reference](../api/metrics.md) for complete metric list
