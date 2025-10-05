# [Basic Examples](@id getting_started_examples)

```@meta
CurrentModule = TextAssociations
```

This page provides practical examples for common use cases in word association analysis.

## Example 1: Academic Paper Analysis

Analyze abstracts to find domain-specific terminology:

```@example academic
using TextAssociations, DataFrames

# Sample academic abstracts
abstracts = """
Machine learning algorithms have revolutionized data analysis by enabling
automated pattern recognition. Deep learning, a subset of machine learning,
uses neural networks to process complex data structures.

Recent advances in artificial intelligence have led to breakthroughs in
natural language processing. Transformer models have become the foundation
for modern language understanding systems.

Computer vision applications leverage convolutional neural networks to
extract features from images. Object detection and image segmentation
are key tasks in computer vision research.
"""

# Find technical terminology
ct = ContingencyTable(abstracts, "learning"; windowsize=5, minfreq=2,
    norm_config=TextNorm(strip_case=true, strip_punctuation=true))

# Calculate multiple metrics for validation
results = assoc_score([PMI, LogDice, LLR], ct)

# Filter for domain-specific terms (high scores across metrics)
technical_terms = filter(row ->
    row.PMI > 3.0 &&
    row.LogDice > 7.0 &&
    row.LLR > 10.83,  # p < 0.001
    results
)

println("Domain-specific collocates of 'learning':")
for row in eachrow(sort(technical_terms, :PMI, rev=true))
    println("  $(row.Collocate): PMI=$(round(row.PMI, digits=2))")
end
```

## Example 2: Social Media Trend Detection

Identify trending word combinations:

```@example social
using TextAssociations

tweets = """
Breaking news: breakthrough in quantum computing announced today!
Quantum computing will transform cryptography and security.
Major tech companies investing billions in quantum research.
Scientists achieve quantum supremacy with new processor design.
Quantum algorithms solve problems classical computers cannot handle.
"""

# Analyze with larger window for social media
ct = ContingencyTable(tweets, "quantum"; windowsize=7, minfreq=1)

# Use LogDice for stable results across different sample sizes
results = assoc_score(LogDice, ct)

println("Trending with 'quantum' (LogDice scores):")
for row in eachrow(first(sort(results, :LogDice, rev=true), 5))
    println("  $(row.Collocate): $(round(row.LogDice, digits=2))")
end
```

## Example 3: Comparative Analysis

Compare collocations across different genres:

```@example comparative
using TextAssociations, DataFrames

# Two different text genres
technical = """
The algorithm optimizes performance through parallel processing.
System architecture supports distributed computing paradigms.
Database queries are optimized using indexing strategies.
"""

narrative = """
The story unfolds through multiple perspectives and timelines.
Character development drives the narrative forward compellingly.
Plot twists keep readers engaged throughout the journey.
"""

# Analyze same word in different contexts
function compare_genres(word::String)
    # Technical context
    ct_tech = ContingencyTable(technical, word; windowsize=3, minfreq=1)
    tech_results = assoc_score(PMI, ct_tech; scores_only=false)
    tech_results[!, :Genre] .= "Technical"

    # Narrative context
    ct_narr = ContingencyTable(narrative, word; windowsize=3, minfreq=1)
    narr_results = assoc_score(PMI, ct_narr; scores_only=false)
    narr_results[!, :Genre] .= "Narrative"

    # Combine results
    combined = vcat(tech_results, narr_results, cols=:union)
    return combined
end

# Compare "the" in both genres
comparison = compare_genres("the")
grouped = groupby(comparison, :Genre)

println("Word associations by genre:")
for group in grouped
    genre = first(group.Genre)
    println("\n$genre context:")
    for row in eachrow(first(sort(group, :PMI, rev=true), 3))
        println("  $(row.Collocate): PMI=$(round(row.PMI, digits=2))")
    end
end
```

## Example 4: Multi-word Expression Detection

Find fixed phrases and idioms:

```@example multiword
using TextAssociations

text = """
The project was completed on time and under budget.
We need to think outside the box for this solution.
Let's touch base next week to discuss progress.
The new approach is a game changer for our industry.
It's important to keep an eye on market trends.
The results speak for themselves in this case.
"""

# Identify components of multi-word expressions
function find_expressions(text::String)
    # Common function words that start expressions
    starters = ["on", "outside", "touch", "game", "keep", "speak"]

    expressions = DataFrame()

    for starter in starters
        ct = ContingencyTable(text, starter, windowsize=2, minfreq=1)
        results = assoc_score([PMI, Dice], ct)

        # High PMI + High Dice = likely fixed expression
        fixed = filter(row -> row.PMI > 2.0 && row.Dice > 0.3, results)

        if nrow(fixed) > 0
            fixed[!, :Starter] .= starter
            expressions = vcat(expressions, fixed, cols=:union)
        end
    end

    return expressions
end

expressions = find_expressions(text)
println("Potential multi-word expressions:")
for row in eachrow(expressions)
    println("  $(row.Starter) + $(row.Collocate)")
end
```

## Example 5: Time-sensitive Analysis

Track changing associations over document sections:

```@example temporal
using TextAssociations, DataFrames

# Documents with temporal progression
early_docs = """
Early computers used vacuum tubes for processing.
Punch cards were the primary input method.
Memory was measured in kilobytes.
"""

modern_docs = """
Modern computers use multi-core processors.
Cloud computing provides unlimited storage.
Memory is measured in terabytes.
"""

function temporal_comparison(word::String)
    # Early period
    ct_early = ContingencyTable(early_docs, word; windowsize=4, minfreq=1)
    early = assoc_score(PMI, ct_early)
    early[!, :Period] .= "Early"

    # Modern period
    ct_modern = ContingencyTable(modern_docs, word; windowsize=4, minfreq=1)
    modern = assoc_score(PMI, ct_modern)
    modern[!, :Period] .= "Modern"

    return vcat(early, modern, cols=:union)
end

temporal = temporal_comparison("computers")
println("\nEvolution of 'computers' associations:")
for period in ["Early", "Modern"]
    period_data = filter(row -> row.Period == period, temporal)
    if nrow(period_data) > 0
        println("\n$period period:")
        for row in eachrow(first(sort(period_data, :PMI, rev=true), 2))
            println("  $(row.Collocate): PMI=$(round(row.PMI, digits=2))")
        end
    end
end
```

## Example 6: Cross-linguistic Analysis

```@example crossling
using TextAssociations

# Greek text example
greek_text = """
Η τεχνητή νοημοσύνη αλλάζει τον κόσμο.
Η μηχανική μάθηση είναι μέρος της τεχνητής νοημοσύνης.
Τα νευρωνικά δίκτυα είναι ισχυρά εργαλεία.
"""

# Configure for Greek
greek_config = TextNorm(
    strip_case=true,
    strip_accents=true,  # Remove tonos marks
    unicode_form=:NFD,
    strip_punctuation=true
)

# Analyze Greek text
ct = ContingencyTable(greek_text, "τεχνητής"; windowsize=3, minfreq=1,
    norm_config=greek_config)

results = assoc_score(PMI, ct)
println("Greek text collocations:")
for row in eachrow(results)
    println("  $(row.Node) + $(row.Collocate): PMI=$(round(row.PMI, digits=2))")
end
```

## Example 7: Building a Collocation Dictionary

Create a reference resource of strong collocations:

```@example dictionary
using TextAssociations, DataFrames

function build_collocation_dict(text::String, min_pmi::Float64=3.0)
    # Key words to analyze
    keywords = ["data", "analysis", "model", "system", "process"]

    dict = DataFrame()

    for keyword in keywords
        # Skip if word not in text
        if !occursin(lowercase(keyword), lowercase(text))
            continue
        end

        ct = ContingencyTable(text, keyword; windowsize=5, minfreq=2)
        results = assoc_score([PMI, LogDice, LLR], ct)

        # Strong collocations only
        strong = filter(row -> row.PMI >= min_pmi, results)

        if nrow(strong) > 0
            dict = vcat(dict, strong, cols=:union)
        end
    end

    # Sort by node then PMI
    sort!(dict, [:Node, order(:PMI, rev=true)])

    return dict
end

sample_text = """
Data analysis requires careful data preparation and data validation.
Statistical models help analyze complex data patterns.
System design influences system performance and system reliability.
Process optimization improves process efficiency significantly.
Model validation ensures model accuracy and model robustness.
"""

dictionary = build_collocation_dict(sample_text, 2.0)
println("\nCollocation Dictionary:")
current_node = ""
for row in eachrow(dictionary)
    if row.Node != current_node
        current_node = row.Node
        println("\n$current_node:")
    end
    println("  → $(row.Collocate) (PMI: $(round(row.PMI, digits=2)))")
end
```

## Example 8: Performance Benchmarking

Compare efficiency of different approaches:

```@example benchmark
using TextAssociations
using BenchmarkTools

text = repeat("The quick brown fox jumps over the lazy dog. ", 100)

# Benchmark different configurations
function benchmark_configs()
    configs = [
        (window=3, minfreq=1, desc="Small window, low threshold"),
        (window=5, minfreq=5, desc="Medium window, medium threshold"),
        (window=10, minfreq=10, desc="Large window, high threshold")
    ]

    println("Configuration benchmarks:")
    for config in configs
        time = @elapsed begin
            ct = ContingencyTable(text, "the"; windowsize=config.window, minfreq=config.minfreq)
            results = assoc_score(PMI, ct; scores_only=true)
        end

        println("  $(config.desc):")
        println("    Time: $(round(time*1000, digits=2))ms")
    end
end

# Benchmark metrics
function benchmark_metrics()
    ct = ContingencyTable(text, "quick"; windowsize=5, minfreq=1)

    metrics = [PMI, LogDice, LLR, Dice]
    println("\nMetric benchmarks:")

    for metric in metrics
        time = @elapsed assoc_score(metric, ct; scores_only=true)
        println("  $metric: $(round(time*1000, digits=3))ms")
    end
end

benchmark_configs()
benchmark_metrics()
```

## Next Steps

- For corpus-level analysis, see [Working with Corpora](../guide/corpus_analysis.md)
- To understand metric selection, see [Choosing Metrics](../guide/choosing_metrics.md)
- For advanced features, see [Temporal Analysis](../advanced/temporal.md)
