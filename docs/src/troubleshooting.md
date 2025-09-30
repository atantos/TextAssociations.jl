# Troubleshooting

```@meta
CurrentModule = TextAssociations
```

This guide helps you diagnose and solve common issues with TextAssociations.jl.

## Common Issues

### Empty Results

#### Problem

`assoc_score` returns an empty DataFrame or no collocates found.

#### Diagnosis

```@example empty_results
using TextAssociations
using TextAnalysis: tokens, text
using DataFrames: nrow

# Debug empty results
function debug_empty_results(s::String, node::String, windowsize::Int, minfreq::Int)
    println("Debugging empty results for '$node'")
    println("-" ^ 40)

    # Step 1: Check preprocessing
    config = TextNorm()
    normalized_node = normalize_node(node, config)
    println("Original node: '$node'")
    println("Normalized node: '$normalized_node'")

    # Step 2: Check if word exists in text
    doc = prep_string(s, config)
    doc_tokens = tokens(doc)
    node_count = count(==(normalized_node), doc_tokens)
    println("\nNode frequency in text: $node_count")

    if node_count == 0
        println("❌ Node not found in text after preprocessing!")
        println("   Check case, spelling, and normalization settings")

        # Find similar words
        similar = filter(t -> startswith(t, normalized_node[1:min(3,end)]), unique(doc_tokens))
        if !isempty(similar)
            println("   Similar words found: $similar")
        end
        return
    end

    # Step 3: Check window and frequency settings
    println("\nWindow size: $windowsize")
    println("Minimum frequency: $minfreq")

    # Step 4: Create contingency table
    ct = ContingencyTable(text(doc), normalized_node; windowsize=windowsize, minfreq=minfreq)
    ct_data = cached_data(ct.con_tbl)

    if isempty(ct_data)
        println("❌ No collocates meet minfreq=$minfreq threshold")

        # Try with minfreq=1 to see what's available
        ct_low = ContingencyTable(text(doc), normalized_node; windowsize=windowsize, minfreq=1)
        ct_low_data = cached_data(ct_low.con_tbl)

        if !isempty(ct_low_data)
            max_freq = maximum(ct_low_data.a)
            println("   Maximum collocate frequency: $max_freq")
            println("   Try lowering minfreq to $max_freq or less")
        end
    else
        println("✓ Found $(nrow(ct_data)) collocates")
    end
end

# Test with problematic case
sample_text = "The Quick Brown Fox Jumps Over The Lazy Dog"
debug_empty_results(sample_text, "quick", 3, 5)  # Case mismatch, high minfreq
```

#### Solutions

```@example empty_solutions
using TextAssociations, DataFrames

# Solution 1: Fix case sensitivity
text = "The Quick Brown Fox"
config_case = TextNorm(strip_case=true)  # Enable case normalization

ct = ContingencyTable(text, "quick"; windowsize=3, minfreq=1, norm_config=config_case)
results = assoc_score(PMI, ct)
println("With case normalization: $(nrow(results)) results")

# Solution 2: Adjust minfreq
ct_low_freq = ContingencyTable(text, "quick"; windowsize=3, minfreq=1)  # Lower minfreq
results_low = assoc_score(PMI, ct_low_freq)
println("With minfreq=1: $(nrow(results_low)) results")

# Solution 3: Increase window size
ct_large_window = ContingencyTable(text, "the"; windowsize=10, minfreq=1)  # Larger window
results_large = assoc_score(PMI, ct_large_window)
println("With windowsize=10: $(nrow(results_large)) results")
```

### Memory Errors

#### Problem

OutOfMemoryError or excessive memory usage.

#### Diagnosis and Solutions

```@example memory_issues
using TextAssociations

# Monitor memory usage
function check_memory_usage(text::String)
    initial_memory = Base.gc_live_bytes() / 1024^2  # MB

    # Create objects
    ct = ContingencyTable(text, "the"; windowsize=5, minfreq=1)
    results = assoc_score([PMI, LogDice, LLR], ct)

    current_memory = Base.gc_live_bytes() / 1024^2
    used_memory = current_memory - initial_memory

    println("Memory usage:")
    println("  Initial: $(round(initial_memory, digits=1)) MB")
    println("  Current: $(round(current_memory, digits=1)) MB")
    println("  Used: $(round(used_memory, digits=1)) MB")

    # Force garbage collection
    GC.gc()
    after_gc = Base.gc_live_bytes() / 1024^2
    println("  After GC: $(round(after_gc, digits=1)) MB")

    return results
end

# Solutions for memory issues
function memory_efficient_analysis(text::String, node::String)
    # 1. Use scores_only
    ct = ContingencyTable(text, node; windowsize=5, minfreq=10)  # Higher minfreq
    scores = assoc_score(PMI, ct; scores_only=true)  # Just vector

    # 2. Process in chunks
    text_length = length(text)
    if text_length > 1_000_000
        println("Large text detected, processing in chunks...")
        # Split and process
    end

    # 3. Clear intermediate results
    ct = nothing
    GC.gc()

    return scores
end

# Test
small_text = "test text " ^ 100
check_memory_usage(small_text)
```

### Unicode and Encoding Issues

#### Problem

Strange characters, encoding errors, or text not matching.

#### Diagnosis

```@example unicode_issues
using TextAssociations
using Unicode

# Diagnose Unicode issues
function diagnose_unicode(text::String)
    println("Unicode Diagnosis:")
    println("-" ^ 40)

    # Check for different Unicode forms
    nfc_form = Unicode.normalize(text, :NFC)
    nfd_form = Unicode.normalize(text, :NFD)

    println("Original length: $(length(text))")
    println("NFC form length: $(length(nfc_form))")
    println("NFD form length: $(length(nfd_form))")

    if length(nfc_form) != length(nfd_form)
        println("⚠ Text contains combining characters")
    end

    # Check for invisible characters
    if occursin(r"[\x00-\x1F\x7F-\x9F]", text)
        println("⚠ Text contains control characters")
    end

    # Check for mixed scripts
    has_latin = occursin(r"\p{Latin}", text)
    has_greek = occursin(r"\p{Greek}", text)
    has_cyrillic = occursin(r"\p{Cyrillic}", text)
    has_arabic = occursin(r"\p{Arabic}", text)
    has_cjk = occursin(r"\p{Han}", text)

    scripts = String[]
    has_latin && push!(scripts, "Latin")
    has_greek && push!(scripts, "Greek")
    has_cyrillic && push!(scripts, "Cyrillic")
    has_arabic && push!(scripts, "Arabic")
    has_cjk && push!(scripts, "CJK")

    println("\nScripts detected: ", join(scripts, ", "))
end

# Test with problematic text
problematic = "café vs café"  # Different Unicode forms
diagnose_unicode(problematic)

# Solution
config = TextNorm(unicode_form=:NFC)  # Normalize to NFC
ct = ContingencyTable(problematic, "café"; windowsize=2, minfreq=1, norm_config=config)
println("\nWith Unicode normalization: works correctly")
```

### Unexpected Metric Values

#### Problem

Metrics returning NaN, Inf, or unexpected values.

#### Diagnosis

```@example metric_issues
using TextAssociations

# Debug metric calculations
function debug_metrics(text::String, node::String, collocate::String)
    ct = ContingencyTable(text, node; windowsize=3, minfreq=1)
    ct_data = cached_data(ct.con_tbl)

    if !isempty(ct_data)
        # Find specific collocate
        row = filter(r -> String(r.Collocate) == collocate, ct_data)

        if !isempty(row)
            r = first(row)
            println("Contingency values for '$node' + '$collocate':")
            println("  a (co-occur): $(r.a)")
            println("  b (node only): $(r.b)")
            println("  c (collocate only): $(r.c)")
            println("  d (neither): $(r.d)")
            println("  N (total): $(r.N)")

            # Check for problematic values
            if r.a == 0
                println("⚠ Zero co-occurrence - most metrics will be -Inf or 0")
            end

            if r.b == 0 || r.c == 0
                println("⚠ Perfect association - some metrics may be Inf")
            end

            # Calculate metrics manually
            pmi = log((r.a * r.N) / ((r.a + r.b) * (r.a + r.c)))
            dice = 2 * r.a / (2 * r.a + r.b + r.c)

            println("\nManual calculations:")
            println("  PMI: $pmi")
            println("  Dice: $dice")
        end
    end
end

# Test case
test_text = "word1 word2 word1 word2 word1"
debug_metrics(test_text, "word1", "word2")
```

### Performance Issues

#### Problem

Analysis is too slow or times out.

#### Solutions

```@example performance_issues
using TextAssociations

# Performance diagnostic
function performance_diagnostic(text::String, node::String)
    println("Performance Diagnostic:")
    println("-" ^ 40)

    # Test different configurations
    configs = [
        (desc="Baseline", window=5, minfreq=5),
        (desc="Small window", window=2, minfreq=5),
        (desc="High threshold", window=5, minfreq=20),
        (desc="Optimized", window=3, minfreq=10)
    ]

    for config in configs
        time = @elapsed begin
            ct = ContingencyTable(text, node; windowsize=config.window, minfreq=config.minfreq)
            results = assoc_score(PMI, ct; scores_only=true)
        end

        println("$(config.desc):")
        println("  Time: $(round(time*1000, digits=2))ms")
        println("  Window: $(config.window), MinFreq: $(config.minfreq)")
    end
end

# Create test text
test_text = repeat("the quick brown fox jumps ", 100)
performance_diagnostic(test_text, "the")
```

## Error Messages Explained

### Common Error Messages

```@example error_messages
using TextAssociations

# Explain common errors
errors = Dict(
    "ArgumentError: Window size must be positive" =>
        "Set windowsize to 1 or greater",

    "ArgumentError: Minimum frequency must be positive" =>
        "Set minfreq to 1 or greater",

    "ArgumentError: Node word cannot be empty" =>
        "Provide a non-empty target word",

    "KeyError: key not found" =>
        "Check DataFrame column names or dictionary keys",

    "MethodError: no method matching" =>
        "Check function arguments and types",

    "OutOfMemoryError" =>
        "Reduce data size or use streaming/batching",

    "UndefVarError" =>
        "Variable not defined - check spelling and scope"
)

println("Common Error Messages and Solutions:")
for (error, solution) in errors
    println("\nError: $error")
    println("Solution: $solution")
end
```

## Debugging Tools

### Comprehensive Debugger

```@example debugger
using TextAssociations
using TextAnalysis: tokens, text
using DataFrames
using TextAnalysis

function comprehensive_debug(text::String, node::String, config::TextNorm,
                           windowsize::Int, minfreq::Int)
    println("="^50)
    println("COMPREHENSIVE DEBUG REPORT")
    println("="^50)

    # Input validation
    println("\n1. INPUT VALIDATION")
    println("   Text length: $(length(text)) characters")
    println("   Node: '$node'")
    println("   Window size: $windowsize")
    println("   Min frequency: $minfreq")

    # Preprocessing check
    println("\n2. PREPROCESSING")
    doc = prep_string(text, config)
    processed_text = TextAnalysis.text(doc)
    normalized_node = normalize_node(node, config)

    println("   Original text sample: '$(first(text, min(50, length(text))))...'")
    println("   Processed sample: '$(first(processed_text, min(50, length(processed_text))))...'")
    println("   Normalized node: '$normalized_node'")

    # Token analysis
    println("\n3. TOKEN ANALYSIS")
    doc_tokens = tokens(doc)
    unique_tokens = unique(doc_tokens)

    println("   Total tokens: $(length(doc_tokens))")
    println("   Unique tokens: $(length(unique_tokens))")
    println("   Node frequency: $(count(==(normalized_node), doc_tokens))")

    # Contingency table
    println("\n4. CONTINGENCY TABLE")
    try
        ct = ContingencyTable(processed_text, normalized_node; windowsize, minfreq)
        ct_data = cached_data(ct.con_tbl)

        if isempty(ct_data)
            println("   ❌ Empty contingency table")

            # Try with minfreq=1
            ct_test = ContingencyTable(processed_text, normalized_node; windowsize, minfreq=1)
            ct_test_data = cached_data(ct_test.con_tbl)

            if !isempty(ct_test_data)
                println("   With minfreq=1: $(nrow(ct_test_data)) collocates found")
                max_freq = maximum(ct_test_data.a)
                println("   Max frequency: $max_freq (current minfreq: $minfreq)")
            end
        else
            println("   ✓ $(nrow(ct_data)) collocates found")
            println("   Frequency range: $(minimum(ct_data.a))-$(maximum(ct_data.a))")
        end

        # Metric calculation
        println("\n5. METRIC CALCULATION")
        results = assoc_score([PMI, LogDice], ct)

        if isempty(results)
            println("   ❌ No results returned")
        else
            println("   ✓ $(nrow(results)) results")
            println("   PMI range: $(round(minimum(results.PMI), digits=2)) to $(round(maximum(results.PMI), digits=2))")
        end

    catch e
        println("   ❌ Error: $e")
        println("   Backtrace: ", catch_backtrace())
    end

    println("\n" * "="^50)
end

# Test the debugger
test_text = "The quick brown fox jumps over the lazy dog"
test_config = TextNorm()
comprehensive_debug(test_text, "fox", test_config, 3, 1)
```

## Getting Help

### Creating a Minimal Reproducible Example

```julia
using TextAssociations

# Minimal reproducible example template
function create_mre()
    # 1. Minimal text that shows the issue
    text = "Your minimal text here"

    # 2. Exact parameters used
    node = "problematic_word"
    windowsize = 5
    minfreq = 2
    config = TextNorm(strip_case=true)

    # 3. Show the error
    try
        ct = ContingencyTable(text, node; windowsize, minfreq, norm_config=config)
        results = assoc_score(PMI, ct)
        println("Results: ", results)
    catch e
        println("Error type: ", typeof(e))
        println("Error message: ", e)
        # Include stack trace
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
    end

    # 4. System information
    println("\nSystem info:")
    println("Julia version: ", VERSION)
    println("TextAssociations version: ", pkgversion(TextAssociations))
    println("OS: ", Sys.KERNEL)
end
```

### Where to Get Help

1. **Documentation**: Check this guide first
2. **GitHub Issues**: [Report bugs](https://github.com/yourusername/TextAssociations.jl/issues)
3. **Discussions**: [Ask questions](https://github.com/yourusername/TextAssociations.jl/discussions)
4. **Julia Discourse**: [Julia community](https://discourse.julialang.org/)

## Next Steps

- Review [Performance Guide](performance.md) for optimization
- See [API Reference](api/functions.md) for function details
- Check [Examples](getting_started/examples.md) for working code
