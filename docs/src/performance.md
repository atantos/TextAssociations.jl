# Performance Guide

```@meta
CurrentModule = TextAssociations
```

This guide covers performance optimization strategies for TextAssociations.jl.

## Benchmarking

### Basic Performance Measurement

```@example benchmark
using TextAssociations
using BenchmarkTools

# Create test data of different sizes
small_text = repeat("The quick brown fox jumps over the lazy dog. ", 10)
medium_text = repeat(small_text * " ", 100)
large_text = repeat(small_text * " ", 1000)

# Benchmark contingency table creation
println("Contingency Table Creation:")
for (name, text) in [("Small", small_text), ("Medium", medium_text), ("Large", large_text)]
    time = @elapsed ContingencyTable(text, "the", windowsize=5, minfreq=2)
    println("  $name ($(length(split(text))) words): $(round(time*1000, digits=2))ms")
end
```

### Metric Performance Comparison

```@example metric_benchmark
using TextAssociations

# Define sample texts for examples
small_text = repeat("The quick brown fox jumps over the lazy dog. ", 10)
medium_text = repeat("The quick brown fox jumps over the lazy dog. ", 100)
large_text = repeat("The quick brown fox jumps over the lazy dog. ", 1000)


ct = ContingencyTable(medium_text, "the", windowsize=5, minfreq=2)

# Benchmark different metrics
metrics = [PMI, LogDice, LLR, Dice, JaccardIdx, ChiSquare]
println("\nMetric Evaluation Times:")

for metric in metrics
    time = @elapsed assoc_score(metric, ct; scores_only=true)
    println("  $metric: $(round(time*1000, digits=3))ms")
end
```

## Memory Optimization

### Using scores_only Flag

```@example memory_opt
using TextAssociations

# Define sample texts for examples
small_text = repeat("The quick brown fox jumps over the lazy dog. ", 10)
medium_text = repeat("The quick brown fox jumps over the lazy dog. ", 100)
large_text = repeat("The quick brown fox jumps over the lazy dog. ", 1000)


ct = ContingencyTable(large_text, "the", windowsize=5, minfreq=2)

# Memory-efficient: returns only vector
@time scores_vector = assoc_score(PMI, ct; scores_only=true)

# More memory: returns full DataFrame
@time scores_df = assoc_score(PMI, ct; scores_only=false)

println("\nMemory usage comparison:")
println("  Vector: $(sizeof(scores_vector)) bytes")
# DataFrame has more overhead
```

### Lazy Evaluation Benefits

```@example lazy_eval
using TextAssociations

# Define sample texts for examples
small_text = repeat("The quick brown fox jumps over the lazy dog. ", 10)
medium_text = repeat("The quick brown fox jumps over the lazy dog. ", 100)
large_text = repeat("The quick brown fox jumps over the lazy dog. ", 1000)


# Lazy evaluation delays computation
println("Creating ContingencyTable (lazy)...")
@time ct = ContingencyTable(large_text, "fox", windowsize=5, minfreq=2)

println("\nFirst evaluation (computes):")
@time results1 = assoc_score(PMI, ct)

println("\nSecond evaluation (uses cache):")
@time results2 = assoc_score(LogDice, ct)
```

## Parallel Processing

### Using Multiple Threads

```@example parallel
using TextAssociations

# Define sample texts for examples
small_text = repeat("The quick brown fox jumps over the lazy dog. ", 10)
medium_text = repeat("The quick brown fox jumps over the lazy dog. ", 100)
large_text = repeat("The quick brown fox jumps over the lazy dog. ", 1000)

using Base.Threads

# Check available threads
println("Available threads: $(Threads.nthreads())")

# Parallel processing of multiple nodes
function parallel_analyze(text::String, nodes::Vector{String})
    results = Vector{Any}(undef, length(nodes))

    Threads.@threads for i in 1:length(nodes)
        ct = ContingencyTable(text, nodes[i], windowsize=5, minfreq=2)
        results[i] = assoc_score(PMI, ct; scores_only=true)
    end

    return results
end

nodes = ["the", "quick", "brown", "fox", "jumps"]
@time parallel_results = parallel_analyze(medium_text, nodes)
println("Processed $(length(nodes)) nodes in parallel")
```

### Distributed Computing

```julia
using Distributed

# Add worker processes
addprocs(4)

@everywhere using TextAssociations

# Distributed analysis
function distributed_analyze(corpus_files::Vector{String}, node::String)
    results = @distributed vcat for file in corpus_files
        text = read(file, String)
        ct = ContingencyTable(text, node; windowsize=5, minfreq=5)
        assoc_score(PMI, ct)
    end
    return results
end
```

## Optimization Strategies

### 1. Preprocessing Optimization

```@example preprocess_opt
using TextAssociations

# Minimal preprocessing for speed
fast_config = TextNorm(
    strip_case=true,
    strip_punctuation=false,
    normalize_whitespace=false,
    strip_accents=false
)

# Full preprocessing
full_config = TextNorm(
    strip_case=true,
    strip_punctuation=true,
    normalize_whitespace=true,
    strip_accents=true
)

text = "Sample text with punctuation!!! And CAPITALS... "

println("Preprocessing performance:")
@time fast_doc = prep_string(text, fast_config)
@time full_doc = prep_string(text, full_config)
```

### 2. Window Size Impact

```@example window_impact
using TextAssociations

# Define sample texts for examples
small_text = repeat("The quick brown fox jumps over the lazy dog. ", 10)
medium_text = repeat("The quick brown fox jumps over the lazy dog. ", 100)
large_text = repeat("The quick brown fox jumps over the lazy dog. ", 1000)


# Window size affects performance
window_sizes = [2, 5, 10, 20]

println("Window size impact:")
for ws in window_sizes
    time = @elapsed ContingencyTable(medium_text, "the"; windowsize=ws, minfreq=2)
    println("  Window $ws: $(round(time*1000, digits=2))ms")
end
```

### 3. Minimum Frequency Filtering

```@example minfreq_impact
using TextAssociations, DataFrames

# Define sample texts for examples
small_text = repeat("The quick brown fox jumps over the lazy dog. ", 10)
medium_text = repeat("The quick brown fox jumps over the lazy dog. ", 100)
large_text = repeat("The quick brown fox jumps over the lazy dog. ", 1000)


# Higher minfreq = fewer calculations
minfreqs = [1, 5, 10, 20]

println("Minimum frequency impact:")
for mf in minfreqs
    ct = ContingencyTable(large_text, "the"; windowsize=5, minfreq=mf)
    results = assoc_score(PMI, ct)
    println("  minfreq=$mf: $(nrow(results)) collocates")
end
```

## Memory-Efficient Patterns

### Streaming Large Corpora

```@example streaming
using TextAssociations

function stream_process(file_pattern::String, node::String, chunk_size::Int=1000)
    aggregated_scores = Dict{String, Float64}()
    file_count = 0

    # Process files in chunks
    for file_batch in Iterators.partition(glob(file_pattern), chunk_size)
        batch_scores = Dict{String, Vector{Float64}}()

        for file in file_batch
            text = read(file, String)
            ct = ContingencyTable(text, node; windowsize=5, minfreq=5)
            results = assoc_score(PMI, ct; scores_only=false)

            for row in eachrow(results)
                collocate = String(row.Collocate)
                push!(get!(batch_scores, collocate, Float64[]), row.PMI)
            end

            file_count += 1
        end

        # Aggregate batch
        for (collocate, scores) in batch_scores
            aggregated_scores[collocate] = mean(scores)
        end

        # Clear batch memory
        batch_scores = nothing
        GC.gc()
    end

    println("Processed $file_count files")
    return aggregated_scores
end

# Example usage (with mock pattern)
# scores = stream_process("corpus/*.txt", "learning", 100)
println("Stream processing function defined")
```

### Batch Processing

```@example batch_process
using TextAssociations, DataFrames

# Define sample texts for examples
small_text = repeat("The quick brown fox jumps over the lazy dog. ", 10)
medium_text = repeat("The quick brown fox jumps over the lazy dog. ", 100)
large_text = repeat("The quick brown fox jumps over the lazy dog. ", 1000)


function batch_process_nodes(text::String, nodes::Vector{String}, batch_size::Int=10)
    all_results = DataFrame()

    for batch_start in 1:batch_size:length(nodes)
        batch_end = min(batch_start + batch_size - 1, length(nodes))
        batch = nodes[batch_start:batch_end]

        println("Processing batch $batch_start-$batch_end...")

        for node in batch
            ct = ContingencyTable(text, node; windowsize=5, minfreq=5)
            results = assoc_score(PMI, ct)
            results[!, :QueryNode] .= node
            all_results = vcat(all_results, results, cols=:union)
        end

        # Force garbage collection between batches
        GC.gc()
    end

    return all_results
end

# Example
nodes = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
batch_results = batch_process_nodes(medium_text, nodes, 3)
println("Batch processed $(length(nodes)) nodes")
```

## Profiling and Optimization

### Using Julia's Profiler

```julia
using Profile
using TextAssociations

# Profile a function
function analyze_text(text, word)
    ct = ContingencyTable(text, word; windowsize=5, minfreq=5)
    return assoc_score([PMI, LogDice, LLR], ct)
end

# Clear previous profiling data
Profile.clear()

# Run with profiling
@profile for i in 1:100
    analyze_text(medium_text, "the")
end

# View profile results
Profile.print(format=:flat, sortby=:count)
```

### Memory Profiling

```julia
using Profile.Allocs

# Profile memory allocations
Profile.Allocs.@profile sample_rate=1 begin
    ct = ContingencyTable(large_text, "the"; windowsize=5, minfreq=5)
    results = assoc_score([PMI, LogDice], ct)
end

# Analyze allocations
results = Profile.Allocs.fetch()
```

## Performance Tips by Scale

### Small Corpora (< 1MB)

```julia
# Optimal settings for small corpora
const SMALL_CORPUS_CONFIG = (
    windowsize = 5,
    minfreq = 1,
    scores_only = false,  # DataFrame overhead negligible
    norm_config = TextNorm()  # Full preprocessing fine
)
```

### Medium Corpora (1-100MB)

```julia
# Balanced settings for medium corpora
const MEDIUM_CORPUS_CONFIG = (
    windowsize = 5,
    minfreq = 5,
    scores_only = false,
    norm_config = TextNorm(
        strip_case = true,
        strip_punctuation = true,
        strip_accents = false  # Skip if not needed
    )
)
```

### Large Corpora (> 100MB)

```julia
# Optimized settings for large corpora
const LARGE_CORPUS_CONFIG = (
    windowsize = 3,  # Smaller window
    minfreq = 10,    # Higher threshold
    scores_only = true,  # Avoid DataFrame overhead
    norm_config = TextNorm(
        strip_case = true,
        strip_punctuation = false,  # Minimal preprocessing
        normalize_whitespace = false
    )
)
```

## Caching Strategies

### Result Caching

```@example caching
using TextAssociations

# Define sample texts for examples
small_text = repeat("The quick brown fox jumps over the lazy dog. ", 10)
medium_text = repeat("The quick brown fox jumps over the lazy dog. ", 100)
large_text = repeat("The quick brown fox jumps over the lazy dog. ", 1000)


# Cache for reused computations
mutable struct CachedAnalyzer
    cache::Dict{Tuple{String, String, Int, Int}, ContingencyTable}
    hits::Int
    misses::Int
end

function analyze_with_cache(analyzer::CachedAnalyzer, text::String,
                           node::String, windowsize::Int, minfreq::Int)
    key = (text[1:min(100, length(text))], node, windowsize, minfreq)  # Use text prefix as key

    if haskey(analyzer.cache, key)
        analyzer.hits += 1
        return analyzer.cache[key]
    else
        analyzer.misses += 1
        ct = ContingencyTable(text, node; windowsize=windowsize, minfreq=minfreq)
        analyzer.cache[key] = ct
        return ct
    end
end

analyzer = CachedAnalyzer(Dict(), 0, 0)

# First call - miss
ct1 = analyze_with_cache(analyzer, medium_text, "the", 5, 2)

# Second call - hit
ct2 = analyze_with_cache(analyzer, medium_text, "the", 5, 2)

println("Cache stats: $(analyzer.hits) hits, $(analyzer.misses) misses")
println("Hit rate: $(round(analyzer.hits / (analyzer.hits + analyzer.misses) * 100, digits=1))%")
```

## Optimization Checklist

### Before Optimization

- [ ] Profile to identify bottlenecks
- [ ] Measure baseline performance
- [ ] Set performance targets
- [ ] Consider accuracy vs speed tradeoffs

### Quick Wins

- [ ] Use `scores_only=true` when DataFrames not needed
- [ ] Increase `minfreq` to reduce vocabulary
- [ ] Reduce `windowsize` for faster processing
- [ ] Disable unnecessary preprocessing

### Advanced Optimizations

- [ ] Implement parallel processing for multiple nodes
- [ ] Use streaming for large corpora
- [ ] Cache frequently used results
- [ ] Pre-compile critical functions

## Common Performance Issues

### Issue: Slow Processing

```@example slow_fix
using TextAssociations
using TextAnalysis: tokens

# Define sample texts for examples
small_text = repeat("The quick brown fox jumps over the lazy dog. ", 10)
medium_text = repeat("The quick brown fox jumps over the lazy dog. ", 100)
large_text = repeat("The quick brown fox jumps over the lazy dog. ", 1000)


function diagnose_performance(text::String, node::String)
    println("Performance Diagnosis:")
    println("-" ^ 40)

    # Check text size
    n_tokens = length(split(text))
    println("Text size: $n_tokens tokens")

    if n_tokens > 100000
        println("⚠ Large text - consider chunking")
    end

    # Check vocabulary size after preprocessing
    cfg = TextNorm()
    doc = prep_string(text, cfg)
    vocab_size = length(unique(tokens(doc)))
    println("Vocabulary size: $vocab_size unique tokens")

    if vocab_size > 10000
        println("⚠ Large vocabulary - increase minfreq")
    end

    # Test different configurations
    configs = [
        (window=2, minfreq=1),
        (window=5, minfreq=5),
        (window=10, minfreq=10)
    ]

    println("\nConfiguration impact:")
    for config in configs
        time = @elapsed ContingencyTable(text, node; windowsize=config.window, minfreq=config.minfreq)
        println("  w=$(config.window), mf=$(config.minfreq): $(round(time*1000, digits=2))ms")
    end
end

diagnose_performance(large_text, "the")
```

### Issue: Out of Memory

```julia
# Memory-efficient processing pattern
function memory_safe_analysis(corpus_files::Vector{String}, node::String)
    results = DataFrame()

    for file in corpus_files
        # Process one file at a time
        text = read(file, String)

        # Use aggressive filtering
        ct = ContingencyTable(text, node, windowsize=3, minfreq=20)  # Small window, high minfreq

        # Get scores only
        scores = assoc_score(PMI, ct; scores_only=true)

        # Aggregate results
        # ... aggregation logic ...

        # Clear memory
        text = nothing
        ct = nothing
        GC.gc()
    end

    return results
end
```

## Benchmarking Suite

```@example benchmark_suite
using TextAssociations

# Define sample texts for examples
small_text = repeat("The quick brown fox jumps over the lazy dog. ", 10)
medium_text = repeat("The quick brown fox jumps over the lazy dog. ", 100)
large_text = repeat("The quick brown fox jumps over the lazy dog. ", 1000)

using BenchmarkTools

function benchmark_suite(text::String)
    suite = BenchmarkGroup()

    # Preprocessing benchmarks
    suite["preprocessing"] = BenchmarkGroup()
    suite["preprocessing"]["minimal"] = @benchmarkable prep_string($text, TextNorm(strip_case=true))
    suite["preprocessing"]["full"] = @benchmarkable prep_string($text, TextNorm())

    # Contingency table benchmarks
    suite["contingency"] = BenchmarkGroup()
    for ws in [2, 5, 10]
        suite["contingency"]["window_$ws"] = @benchmarkable ContingencyTable($text, "the", windowsize=$ws, minfreq=5)
    end

    # Metric benchmarks
    ct = ContingencyTable(text, "the", windowsize=5, minfreq=5)
    suite["metrics"] = BenchmarkGroup()
    for metric in [PMI, LogDice, LLR]
        suite["metrics"]["$metric"] = @benchmarkable assoc_score($metric, $ct; scores_only=true)
    end

    return suite
end

# Run suite
suite = benchmark_suite(small_text)
# results = run(suite)
println("Benchmark suite created")
```

## Next Steps

- Review [Troubleshooting](troubleshooting.md) for common issues
- See [API Reference](api/functions.md) for function details
- Read [Contributing](contributing.md) to help optimize the package
