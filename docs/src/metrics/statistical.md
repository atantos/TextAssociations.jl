# Statistical Metrics

```@meta
CurrentModule = TextAssociations
```

Statistical metrics provide hypothesis testing and significance assessment for word associations.

## Chi-Square Test

### Theory

The chi-square test measures the difference between observed and expected frequencies:

```math
\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
```

### Implementation

```@example chisquare
using TextAssociations

text = """
Statistical analysis requires careful statistical methods.
Statistical significance indicates statistical relationships.
Random patterns show no statistical correlation.
"""

ct = ContingencyTable(text, "statistical", 4, 1)
results = assoc_score([ChiSquare, Tscore, Zscore], ct)

println("Statistical Tests for 'statistical':")
for row in eachrow(results)
    # Chi-square critical values (df=1)
    chi_sig = if row.ChiSquare > 10.83
        "p < 0.001"
    elseif row.ChiSquare > 6.63
        "p < 0.01"
    elseif row.ChiSquare > 3.84
        "p < 0.05"
    else
        "not significant"
    end

    println("\n$(row.Collocate):")
    println("  χ² = $(round(row.ChiSquare, digits=2)) ($chi_sig)")
    println("  t-score = $(round(row.Tscore, digits=2))")
    println("  z-score = $(round(row.Zscore, digits=2))")
end
```

## T-Score

### Theory

The t-score measures the confidence that the observed frequency differs from expected:

```math
t = \frac{O - E}{\sqrt{O}}
```

### Application

```@example tscore
using TextAssociations

# T-score for collocation strength
t_results = assoc_score(Tscore, ct)

println("\nT-score interpretation:")
for row in eachrow(t_results)
    confidence = if abs(row.Tscore) > 2.576
        "99% confidence"
    elseif abs(row.Tscore) > 1.96
        "95% confidence"
    elseif abs(row.Tscore) > 1.645
        "90% confidence"
    else
        "not confident"
    end

    println("  $(row.Collocate): t=$(round(row.Tscore, digits=2)) ($confidence)")
end
```

## Z-Score

### Theory

Z-score standardizes the difference between observed and expected:

```math
z = \frac{O - E}{\sigma}
```

### Comparison with T-score

```@example zscore_comparison
using TextAssociations

# Compare T-score and Z-score
both = assoc_score([Tscore, Zscore], ct)

println("\nT-score vs Z-score:")
println("T-score: Uses observed frequency in denominator")
println("Z-score: Uses theoretical standard deviation")

for row in eachrow(first(both, 3))
    println("\n$(row.Collocate):")
    println("  T-score: $(round(row.Tscore, digits=2))")
    println("  Z-score: $(round(row.Zscore, digits=2))")

    if abs(row.Zscore) > abs(row.Tscore)
        println("  → Z-score shows stronger evidence")
    else
        println("  → T-score shows stronger evidence")
    end
end
```

## Fisher's Exact Test

While not directly implemented, Fisher's exact test is important for small samples:

```@example fisher_concept
using TextAssociations

function explain_fisher_test()
    println("Fisher's Exact Test:")
    println("  Use when: Sample sizes are small")
    println("  Advantage: Exact p-values (not asymptotic)")
    println("  Disadvantage: Computationally intensive")
    println("\nRule of thumb: Use Fisher's when any expected frequency < 5")

    # Check if Fisher's would be recommended
    function needs_fisher(ct::ContingencyTable)
        data = cached_data(ct.con_tbl)
        if !isempty(data)
            min_expected = minimum([data.E₁₁, data.E₁₂, data.E₂₁, data.E₂₂])
            return any(min_expected .< 5)
        end
        return false
    end

    return needs_fisher
end

checker = explain_fisher_test()

# Check our contingency table
if checker(ct)
    println("\n⚠ This data might benefit from Fisher's exact test")
else
    println("\n✓ Chi-square test is appropriate for this data")
end
```

## Critical Values and P-values

### Reference Table

```@example critical_values
using TextAssociations, DataFrames

critical_values = DataFrame(
    Test = ["Chi-square (df=1)", "T-score", "Z-score"],
    p_0_05 = [3.84, 1.96, 1.96],
    p_0_01 = [6.63, 2.576, 2.576],
    p_0_001 = [10.83, 3.291, 3.291],
    p_0_0001 = [15.13, 3.891, 3.891]
)

println("Critical Values for Statistical Tests:")
for row in eachrow(critical_values)
    println("\n$(row.Test):")
    println("  p < 0.05:  $(row.p_0_05)")
    println("  p < 0.01:  $(row.p_0_01)")
    println("  p < 0.001: $(row.p_0_001)")
    println("  p < 0.0001: $(row.p_0_0001)")
end
```

## Effect Size vs Significance

### Important Distinction

```@example effect_vs_sig
using TextAssociations

# High significance doesn't always mean large effect
text_large = repeat("word1 word2 ", 1000)  # Large sample
text_small = "word1 word2 word1 word2"     # Small sample

ct_large = ContingencyTable(text_large, "word1", 2, 1)
ct_small = ContingencyTable(text_small, "word1", 2, 1)

println("Statistical Significance vs Effect Size:")

# Large sample
if !isempty(assoc_score(ChiSquare, ct_large))
    large_chi = first(assoc_score(ChiSquare, ct_large)).ChiSquare
    large_pmi = first(assoc_score(PMI, ct_large)).PMI

    println("\nLarge sample (n=2000):")
    println("  χ² = $(round(large_chi, digits=2)) (high significance)")
    println("  PMI = $(round(large_pmi, digits=2)) (effect size)")
end

# Small sample
if !isempty(assoc_score(ChiSquare, ct_small))
    small_chi = first(assoc_score(ChiSquare, ct_small)).ChiSquare
    small_pmi = first(assoc_score(PMI, ct_small)).PMI

    println("\nSmall sample (n=4):")
    println("  χ² = $(round(small_chi, digits=2)) (low significance)")
    println("  PMI = $(round(small_pmi, digits=2)) (effect size)")
end

println("\n→ Large samples can show significance for small effects")
println("→ Always report both significance AND effect size")
```

## Multiple Testing Correction

### Bonferroni Correction

```@example bonferroni
using TextAssociations

function bonferroni_correction(p_values::Vector{Float64}, alpha::Float64=0.05)
    n_tests = length(p_values)
    corrected_alpha = alpha / n_tests

    println("Bonferroni Correction:")
    println("  Number of tests: $n_tests")
    println("  Original α: $alpha")
    println("  Corrected α: $(round(corrected_alpha, digits=4))")

    # Which tests remain significant?
    significant = p_values .< corrected_alpha
    println("  Significant after correction: $(sum(significant))/$n_tests")

    return corrected_alpha
end

# Example with multiple comparisons
p_values = [0.001, 0.01, 0.02, 0.03, 0.04]
bonferroni_correction(p_values)
```

## Best Practices

### 1. Combining Statistical Tests

```@example combining
using TextAssociations

function comprehensive_statistical_test(ct::ContingencyTable)
    # Use multiple tests
    results = assoc_score([ChiSquare, LLR, Tscore, Zscore], ct)

    if !isempty(results)
        # Add consensus column
        results.Consensus = zeros(Int, nrow(results))

        for i in 1:nrow(results)
            consensus = 0
            consensus += results[i, :ChiSquare] > 10.83 ? 1 : 0  # p < 0.001
            consensus += results[i, :LLR] > 10.83 ? 1 : 0
            consensus += abs(results[i, :Tscore]) > 3.291 ? 1 : 0
            consensus += abs(results[i, :Zscore]) > 3.291 ? 1 : 0
            results.Consensus[i] = consensus
        end

        # Filter by consensus
        strong = filter(row -> row.Consensus >= 3, results)

        println("Statistical Consensus (≥3/4 tests significant at p<0.001):")
        for row in eachrow(strong)
            println("  $(row.Collocate): $(row.Consensus)/4 tests agree")
        end
    end
end

comprehensive_statistical_test(ct)
```

### 2. Sample Size Considerations

```julia
# Minimum sample sizes for reliable statistical tests
const MIN_SAMPLES = Dict(
    :chisquare => 20,   # All expected frequencies > 5
    :tscore => 10,      # Reasonable minimum
    :zscore => 30,      # Central limit theorem
    :fisher => 0        # Works for any size
)
```

### 3. Reporting Guidelines

```@example reporting
using TextAssociations

function report_statistical_results(results::DataFrame)
    println("Statistical Analysis Report")
    println("="^40)

    for row in eachrow(first(results, 3))
        println("\nCollocate: $(row.Collocate)")
        println("  Frequency: $(row.Frequency)")

        if hasproperty(results, :ChiSquare)
            chi_p = row.ChiSquare > 10.83 ? "***" :
                   row.ChiSquare > 6.63 ? "**" :
                   row.ChiSquare > 3.84 ? "*" : "ns"
            println("  χ²(1) = $(round(row.ChiSquare, digits=2)) $chi_p")
        end

        if hasproperty(results, :Tscore)
            println("  t = $(round(row.Tscore, digits=2))")
        end

        # Report effect size alongside
        if hasproperty(results, :PMI)
            println("  Effect size (PMI) = $(round(row.PMI, digits=2))")
        end
    end

    println("\n" * "="^40)
    println("* p < 0.05, ** p < 0.01, *** p < 0.001")
end

results_full = assoc_score([ChiSquare, Tscore, PMI], ct)
report_statistical_results(results_full)
```

## Next Steps

- Learn about [Similarity Metrics](similarity.md) for symmetric measures
- Explore [Effect Size Metrics](epidemiological.md) for practical significance
- Review [Choosing Metrics](../guide/choosing_metrics.md) for selection guidance
