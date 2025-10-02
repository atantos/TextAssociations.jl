# Temporal Analysis

```
@id advanced_temporal
```

```@meta
CurrentModule = TextAssociations
```

Analyze how word associations change over time periods.

## Overview

Temporal analysis tracks the evolution of collocations across time, revealing trends, emerging terminology, and changing language patterns.

## Basic Temporal Analysis

```@example basic_temporal
using TextAssociations, Dates, DataFrames

# Create corpus with temporal metadata
texts = [
    ("Early computers used vacuum tubes.", Date(1950)),
    ("Transistors replaced vacuum tubes.", Date(1960)),
    ("Integrated circuits revolutionized computing.", Date(1970)),
    ("Microprocessors enabled personal computers.", Date(1980)),
    ("Internet connected computers globally.", Date(1990)),
    ("Cloud computing emerged as dominant paradigm.", Date(2000)),
    ("AI and machine learning transform computing.", Date(2010)),
    ("Quantum computing shows promise.", Date(2020))
]

# Create corpus with temporal metadata
df = DataFrame(
    text = [t[1] for t in texts],
    year = [year(t[2]) for t in texts]
)

corpus = read_corpus_df(df;
    text_column=:text,
    metadata_columns=[:year]
)

# Analyze temporal trends
temporal = analyze_temporal(
    corpus,
    ["computing", "computers"],
    :year,
    PMI;
    time_bins=4,
    windowsize=5,
    minfreq=1
)

println("Temporal Analysis Results:")
println("Time periods analyzed: ", temporal.time_periods)

if !isempty(temporal.trend_analysis)
    println("\nTop trending associations:")
    trending = first(sort(temporal.trend_analysis, :Correlation, rev=true), 5)
    for row in eachrow(trending)
        println("  $(row.Node) + $(row.Collocate): r=$(round(row.Correlation, digits=2))")
    end
end
```

## Trend Detection

### Identifying Emerging Terms

```@example emerging
using TextAssociations, Statistics

function identify_emerging_terms(temporal_analysis::TemporalCorpusAnalysis,
                                threshold::Float64=0.5)
    trends = temporal_analysis.trend_analysis

    # Filter for positive trends
    emerging = filter(row -> row.Correlation > threshold, trends)

    # Sort by slope (rate of change)
    sort!(emerging, :Slope, rev=true)

    println("Emerging Terms (correlation > $threshold):")
    for row in eachrow(first(emerging, 10))
        trend = row.Slope > 0 ? "↑" : "↓"
        println("  $(row.Node) + $(row.Collocate): $trend slope=$(round(row.Slope, digits=3))")
    end

    return emerging
end

# Apply to our temporal analysis
# emerging_terms = identify_emerging_terms(temporal)
println("\nNote: Full trend detection requires more data points")
```

### Detecting Declining Associations

```@example declining
using TextAssociations

function identify_declining_terms(temporal_analysis::TemporalCorpusAnalysis)
    trends = temporal_analysis.trend_analysis

    # Filter for negative trends
    declining = filter(row -> row.Correlation < -0.3, trends)

    println("Declining Associations:")
    if !isempty(declining)
        for row in eachrow(declining)
            println("  $(row.Node) + $(row.Collocate): correlation=$(round(row.Correlation, digits=2))")
        end
    else
        println("  No strongly declining associations found")
    end

    return declining
end

# Apply to our analysis
# declining = identify_declining_terms(temporal)
```

## Period Comparison

### Cross-Period Analysis

```@example basic_temporal
using TextAssociations

function compare_periods(temporal_analysis::TemporalCorpusAnalysis,
                        period1::String, period2::String)
    results1 = temporal_analysis.results_by_period[period1]
    results2 = temporal_analysis.results_by_period[period2]

    # Get all nodes
    nodes = union(results1.nodes, results2.nodes)

    comparison = DataFrame()

    for node in nodes
        if haskey(results1.results, node) && haskey(results2.results, node)
            df1 = results1.results[node]
            df2 = results2.results[node]

            # Find common collocates
            common = intersect(df1.Collocate, df2.Collocate)

            for collocate in common
                idx1 = findfirst(==(collocate), df1.Collocate)
                idx2 = findfirst(==(collocate), df2.Collocate)

                if idx1 !== nothing && idx2 !== nothing
                    # Assume first metric column after standard columns
                    metric_col = names(df1)[findfirst(n -> n ∉ [:Node, :Collocate, :Frequency, :DocFrequency], names(df1))]

                    score1 = df1[idx1, metric_col]
                    score2 = df2[idx2, metric_col]

                    push!(comparison, (
                        Node = node,
                        Collocate = collocate,
                        Period1_Score = score1,
                        Period2_Score = score2,
                        Change = score2 - score1,
                        PercentChange = (score2 - score1) / abs(score1) * 100
                    ))
                end
            end
        end
    end

    return comparison
end

# Example comparison
if length(temporal.time_periods) >= 2
    period1 = temporal.time_periods[1]
    period2 = temporal.time_periods[end]
    println("\nComparing $period1 vs $period2:")
    # comparison = compare_periods(temporal, period1, period2)
end
```

## Visualization Preparation

### Time Series Data

```@example timeseries
using TextAssociations, DataFrames

function prepare_timeseries_data(temporal_analysis::TemporalCorpusAnalysis,
                                node::String, collocate::Symbol)
    periods = String[]
    scores = Float64[]

    for period in sort(temporal_analysis.time_periods)
        if haskey(temporal_analysis.results_by_period, period)
            results = temporal_analysis.results_by_period[period]

            if haskey(results.results, node)
                df = results.results[node]
                idx = findfirst(==(collocate), df.Collocate)

                if idx !== nothing
                    # Find metric column
                    metric_cols = filter(n -> n ∉ [:Node, :Collocate, :Frequency, :DocFrequency], names(df))
                    if !isempty(metric_cols)
                        push!(periods, period)
                        push!(scores, df[idx, metric_cols[1]])
                    end
                end
            end
        end
    end

    return DataFrame(Period=periods, Score=scores)
end

# Prepare data for plotting
# timeseries = prepare_timeseries_data(temporal, "computing", :ai)
println("\nTime series data structure prepared for visualization")
```

## Advanced Temporal Patterns

### Burst Detection

```@example burst
using TextAssociations, Statistics

function detect_bursts(temporal_analysis::TemporalCorpusAnalysis,
                      z_threshold::Float64=2.0)
    bursts = DataFrame()

    for (node, node_results) in temporal_analysis.results_by_period[1].results
        # Track each collocate over time
        collocate_scores = Dict{Symbol, Vector{Float64}}()

        for period in temporal_analysis.time_periods
            if haskey(temporal_analysis.results_by_period[period].results, node)
                period_df = temporal_analysis.results_by_period[period].results[node]

                for row in eachrow(period_df)
                    if !haskey(collocate_scores, row.Collocate)
                        collocate_scores[row.Collocate] = Float64[]
                    end
                    # Get first metric score
                    metric_cols = filter(n -> n ∉ [:Node, :Collocate, :Frequency, :DocFrequency], names(period_df))
                    if !isempty(metric_cols)
                        push!(collocate_scores[row.Collocate], row[metric_cols[1]])
                    end
                end
            end
        end

        # Detect bursts using z-scores
        for (collocate, scores) in collocate_scores
            if length(scores) > 2
                μ = mean(scores)
                σ = std(scores)

                if σ > 0
                    z_scores = (scores .- μ) ./ σ
                    max_z = maximum(z_scores)

                    if max_z > z_threshold
                        burst_period = temporal_analysis.time_periods[argmax(z_scores)]
                        push!(bursts, (
                            Node = node,
                            Collocate = collocate,
                            BurstPeriod = burst_period,
                            ZScore = max_z
                        ))
                    end
                end
            end
        end
    end

    if !isempty(bursts)
        sort!(bursts, :ZScore, rev=true)
        println("Detected Bursts (z > $z_threshold):")
        for row in eachrow(first(bursts, min(5, nrow(bursts))))
            println("  $(row.Node) + $(row.Collocate) in $(row.BurstPeriod): z=$(round(row.ZScore, digits=2))")
        end
    else
        println("No significant bursts detected")
    end

    return bursts
end

# Detect bursts in our data
# bursts = detect_bursts(temporal)
```

### Semantic Shift Detection

```@example semantic_shift
using TextAssociations

function detect_semantic_shift(temporal_analysis::TemporalCorpusAnalysis,
                              node::String, threshold::Float64=0.5)
    periods = temporal_analysis.time_periods

    if length(periods) < 2
        println("Need at least 2 periods for semantic shift detection")
        return DataFrame()
    end

    # Compare first and last periods
    first_period = periods[1]
    last_period = periods[end]

    shifts = DataFrame()

    if haskey(temporal_analysis.results_by_period[first_period].results, node) &&
       haskey(temporal_analysis.results_by_period[last_period].results, node)

        first_df = temporal_analysis.results_by_period[first_period].results[node]
        last_df = temporal_analysis.results_by_period[last_period].results[node]

        # Find collocates unique to each period
        early_only = setdiff(first_df.Collocate, last_df.Collocate)
        late_only = setdiff(last_df.Collocate, first_df.Collocate)

        println("Semantic shift for '$node':")
        println("  Lost associations ($(first_period)): ", first(early_only, 5))
        println("  New associations ($(last_period)): ", first(late_only, 5))

        # Calculate shift magnitude
        all_collocates = union(first_df.Collocate, last_df.Collocate)
        overlap = intersect(first_df.Collocate, last_df.Collocate)

        jaccard = length(overlap) / length(all_collocates)
        shift_magnitude = 1 - jaccard

        println("  Semantic shift magnitude: $(round(shift_magnitude, digits=2))")
    end

    return shifts
end

# Analyze semantic shift
# detect_semantic_shift(temporal, "computing")
```

## Best Practices

### 1. Time Bin Selection

```julia
# Guidelines for time bins
function optimal_time_bins(corpus_size::Int, time_span::Int)
    if time_span < 10
        return 2:3  # Few bins for short spans
    elseif time_span < 50
        return 5:10  # Moderate bins
    else
        return 10:20  # More bins for long spans
    end
end
```

### 2. Minimum Data Requirements

```julia
# Ensure sufficient data per period
const MIN_DOCS_PER_PERIOD = 10
const MIN_TOKENS_PER_PERIOD = 1000
const MIN_NODE_FREQ_PER_PERIOD = 5
```

### 3. Trend Validation

```julia
# Validate trends with multiple metrics
function validate_trend(temporal_analysis, node, collocate)
    # Check consistency across metrics
    # Require minimum correlation strength
    # Verify sufficient data points
end
```

## Next Steps

- Explore [Network Analysis](networks.md) for visualizing temporal changes
- See [Keyword Extraction](keywords.md) for period-specific keywords
- Review [Performance Guide](../performance.md) for large temporal corpora
