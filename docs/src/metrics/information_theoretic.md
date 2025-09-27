# Information-Theoretic Metrics

```@meta
CurrentModule = TextAssociations
```

Information-theoretic metrics measure the amount of information shared between words based on their co-occurrence patterns. Let's create a `ContingencyTable` that will be used later in this documentation.

```@example setup
using TextAssociations

text = """
Natural language processing uses computational linguistics.
Computational methods analyze natural language data.
Language models process natural text efficiently.
"""

ct = ContingencyTable(text, "language"; windowsize=3, minfreq=1)

println("Contingency table created:")
println("  node  : ", ct.node)
println("  window: 3, minfreq: 1")
```

## Pointwise Mutual Information (PMI)

### Theory

PMI measures how much more likely two words are to co-occur than would be expected by chance:

```math
PMI(x,y) = \log_2 \frac{P(x,y)}{P(x)P(y)}
```

### Implementation

```@example setup
results = assoc_score(PMI, ct)

println("PMI scores for 'language':")
for row in eachrow(results)
    interpretation = if row.PMI > 5
        "very strong"
    elseif row.PMI > 3
        "strong"
    elseif row.PMI > 0
        "positive"
    else
        "negative/no association"
    end

    println("  $(row.Collocate): $(round(row.PMI, digits=2)) ($interpretation)")
end
```

### PMI Variants

```@example setup
using TextAssociations

# Compare PMI variants
variants = assoc_score([PMI, PMI², PMI³, PPMI], ct)

println("\nPMI Variants Comparison:")
println("Standard PMI: Balanced information measure")
println("PMI²: Emphasizes frequency (f²/expected)")
println("PMI³: Strong frequency bias (f³/expected)")
println("PPMI: Positive PMI (negative values → 0)")

for row in eachrow(first(variants, 3))
    println("\n$(row.Collocate):")
    println("  PMI:  $(round(row.PMI, digits=2))")
    println("  PMI²: $(round(row.PMI², digits=2))")
    println("  PMI³: $(round(row.PMI³, digits=2))")
    println("  PPMI: $(round(row.PPMI, digits=2))")
end
```

## Log-Likelihood Ratio (LLR)

### Theory

LLR compares observed frequencies with expected frequencies under independence:

```math
LLR = 2 \sum_{ij} O_{ij} \log \frac{O_{ij}}{E_{ij}}
```

### Implementation

```@example setup
using TextAssociations

# LLR for statistical significance
llr_results = assoc_score([LLR, LLR²], ct)

println("\nLog-Likelihood Ratio:")
for row in eachrow(llr_results)
    # Critical values for significance
    significance = if row.LLR > 15.13
        "p < 0.0001 (****)"
    elseif row.LLR > 10.83
        "p < 0.001 (***)"
    elseif row.LLR > 6.63
        "p < 0.01 (**)"
    elseif row.LLR > 3.84
        "p < 0.05 (*)"
    else
        "not significant"
    end

    println("  $(row.Collocate): LLR=$(round(row.LLR, digits=2)) $significance")
    println("    LLR²=$(round(row.LLR², digits=2)) (squared variant)")
end
```

## Mutual Information

### Theory

Mutual Information measures the total information shared between two variables:

```math
MI(X;Y) = \sum_{x,y} P(x,y) \log \frac{P(x,y)}{P(x)P(y)}
```

While PMI is the pointwise version, MI is the expected value of PMI over all possible outcomes.

### Relationship to PMI

```@example mi_pmi
using TextAssociations

# PMI is the pointwise (local) version
# MI would be the weighted sum over all contexts

function explain_mi_pmi_relationship()
    println("Relationship between MI and PMI:")
    println("  PMI: Information for specific word pair")
    println("  MI: Average information over all pairs")
    println("\nMI = Σ P(x,y) × PMI(x,y)")
    println("\nPMI tells us about specific associations")
    println("MI tells us about overall dependency")
end

explain_mi_pmi_relationship()
```

## Information Gain

Information gain measures how much knowing one word reduces uncertainty about another:

```@example setup
using TextAssociations

# Demonstrate information gain concept
function information_gain_example(ct::ContingencyTable)
    results = assoc_score([PMI, PPMI], ct)

    # High PMI indicates high information gain
    high_info = filter(row -> row.PMI > 3, results)

    println("\nHigh Information Gain pairs (PMI > 3):")
    for row in eachrow(high_info)
        println("  Knowing '$(ct.node)' gives $(round(row.PMI, digits=2)) bits about '$(row.Collocate)'")
    end
end

information_gain_example(ct)
```

## Positive PMI (PPMI)

### Motivation

PPMI addresses the issue that PMI can be negative for words that co-occur less than expected:

```@example setup
using TextAssociations

# Compare PMI and PPMI
both = assoc_score([PMI, PPMI], ct)

println("\nPMI vs PPMI:")
for row in eachrow(both)
    if row.PMI < 0
        println("  $(row.Collocate): PMI=$(round(row.PMI, digits=2)) → PPMI=0")
        println("    (Negative association set to zero)")
    else
        println("  $(row.Collocate): PMI=$(round(row.PMI, digits=2)) = PPMI=$(round(row.PPMI, digits=2))")
    end
end
```

## Normalized Variants

### Normalized PMI

Some applications benefit from normalized PMI variants:

```julia
# Normalized PMI (NPMI) - scales to [-1, 1]
# NPMI = PMI / -log(P(x,y))

# This makes PMI values more comparable across different frequency ranges
```

## Practical Considerations

### Frequency Effects

```@example freq_effects
using TextAssociations

# Create examples with different frequencies
high_freq_text = repeat("the word the word the word ", 10)
low_freq_text = "rare unique special extraordinary unusual"

println("Frequency effects on PMI:")

# High frequency
ct_high = ContingencyTable(high_freq_text, "the"; windowsize=2, minfreq=1)
pmi_high = assoc_score(PMI, ct_high)
if !isempty(pmi_high)
    println("\nHigh frequency word 'the':")
    println("  Max PMI: $(round(maximum(pmi_high.PMI), digits=2))")
end

# Low frequency
ct_low = ContingencyTable(low_freq_text, "rare"; windowsize=3, minfreq=1)
pmi_low = assoc_score(PMI, ct_low)
if !isempty(pmi_low)
    println("\nLow frequency word 'rare':")
    println("  Max PMI: $(round(maximum(pmi_low.PMI), digits=2))")
end

println("\n→ PMI tends to favor low-frequency pairs")
```

### Sparse Data Problem

```@example sparse_data
using TextAssociations

# Small corpus - sparse data
small_text = "word1 word2 word3"
ct_small = ContingencyTable(small_text, "word1"; windowsize=2, minfreq=1)

# Large corpus - more reliable
large_text = repeat("word1 word2 word3 word4 word5 ", 100)
ct_large = ContingencyTable(large_text, "word1"; windowsize=2, minfreq=1)

println("Sparse data effects:")
println("  Small corpus: $(length(split(small_text))) tokens")
println("  Large corpus: $(length(split(large_text))) tokens")
println("\n→ Information-theoretic metrics need sufficient data")
```

## Choosing Information-Theoretic Metrics

### Decision Guide

| Use Case                  | Recommended Metric | Reason                                  |
| ------------------------- | ------------------ | --------------------------------------- |
| Finding rare associations | PMI                | Highlights low-frequency patterns       |
| Statistical validation    | LLR                | Provides p-values                       |
| Dimensionality reduction  | PPMI               | No negative values, works well with SVD |
| Frequency-weighted        | PMI² or PMI³       | Emphasizes common patterns              |
| Cross-corpus comparison   | Normalized PMI     | Comparable across corpora               |

### Threshold Guidelines

```@example thresholds
using TextAssociations, DataFrames

thresholds = DataFrame(
    Metric = ["PMI", "PPMI", "LLR", "LLR²"],
    Weak = ["0-2", "0-2", "0-3.84", "0-15"],
    Moderate = ["2-4", "2-4", "3.84-10.83", "15-50"],
    Strong = ["4-7", "4-7", "10.83-15.13", "50-100"],
    VeryStrong = [">7", ">7", ">15.13", ">100"]
)

println("Information-Theoretic Metric Thresholds:")
for row in eachrow(thresholds)
    println("\n$(row.Metric):")
    println("  Weak: $(row.Weak)")
    println("  Moderate: $(row.Moderate)")
    println("  Strong: $(row.Strong)")
    println("  Very Strong: $(row.VeryStrong)")
end
```

## Advanced Applications

### Semantic Similarity

```@example semantic_sim
using TextAssociations

# Use PMI for semantic similarity
function semantic_similarity(corpus_text::String, word1::String, word2::String)
    # Get PMI profiles for both words
    ct1 = ContingencyTable(corpus_text, word1; windowsize=5, minfreq=1)
    ct2 = ContingencyTable(corpus_text, word2; windowsize=5, minfreq=1)

    pmi1 = assoc_score(PPMI, ct1)
    pmi2 = assoc_score(PPMI, ct2)

    # Find common collocates
    if !isempty(pmi1) && !isempty(pmi2)
        common = intersect(pmi1.Collocate, pmi2.Collocate)
        println("Common contexts for '$word1' and '$word2': ", length(common))

        if length(common) > 0
            # Could compute cosine similarity of PMI vectors here
            println("Shared collocates: ", first(common, min(5, length(common))))
        end
    end
end

text = """
Dogs are loyal pets. Cats are independent pets.
Dogs need walks. Cats need litter boxes.
Both dogs and cats make great companions.
"""

semantic_similarity(text, "dogs", "cats")
```

### Feature Extraction

```@example features
using TextAssociations

# Use PPMI for feature extraction
function extract_features(corpus_text::String, target_words::Vector{String}, top_n::Int=10)
    features = Dict{String, Vector{Symbol}}()

    for word in target_words
        ct = ContingencyTable(corpus_text, word; windowsize=5, minfreq=1)
        ppmi = assoc_score(PPMI, ct)

        if !isempty(ppmi)
            # Top PPMI scores as features
            sorted = sort(ppmi, :PPMI, rev=true)
            features[word] = first(sorted.Collocate, min(top_n, nrow(sorted)))
        else
            features[word] = Symbol[]
        end
    end

    return features
end

sample_text = """
Machine learning uses algorithms. Deep learning uses neural networks.
Statistics uses probability. Mathematics uses logic.
"""

features = extract_features(sample_text, ["learning", "uses"], 5)
println("\nExtracted features:")
for (word, feat) in features
    println("  $word: $feat")
end
```

## References

1. Church, K. W., & Hanks, P. (1990). Word association norms, mutual information, and lexicography.
2. Bouma, G. (2009). Normalized (pointwise) mutual information in collocation extraction.
3. Dunning, T. (1993). Accurate methods for the statistics of surprise and coincidence.

## Next Steps

- Explore [Statistical Metrics](statistical.md) for hypothesis testing
- Learn about [Similarity Metrics](similarity.md) for symmetric measures
- See [Choosing Metrics](../guide/choosing_metrics.md) for practical guidance
