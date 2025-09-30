# Text Preprocessing

```@meta
CurrentModule = TextAssociations
```

Effective text preprocessing is crucial for accurate word association analysis. This guide covers all preprocessing options and best practices.

## The TextNorm Configuration

TextAssociations.jl uses the `TextNorm` struct to control all preprocessing:

```@example textnorm_intro
using TextAssociations

# Default configuration
default_config = TextNorm()

# Custom configuration
custom_config = TextNorm(
    strip_case=true,           # Lowercase text
    strip_accents=false,       # Keep diacritics
    unicode_form=:NFC,         # Unicode normalization
    strip_punctuation=true,    # Remove punctuation
    punctuation_to_space=true, # Replace punct with space
    normalize_whitespace=true, # Collapse multiple spaces
    strip_whitespace=false,    # Don't remove all spaces
    use_prepare=false         # Don't use TextAnalysis pipeline
)

println("Default settings:")
for field in fieldnames(TextNorm)
    println("  $field: $(getfield(default_config, field))")
end
```

## Preprocessing Options

### Case Normalization

```@example case
using TextAssociations
using DataFrames

text = "The IBM CEO visited NASA headquarters."

# Keep original case
ct_case = ContingencyTable(text, "IBM";
    windowsize=5,
    minfreq=1,
    norm_config=TextNorm(strip_case=false))
results_case = assoc_score(PMI, ct_case)

# Normalize to lowercase
ct_lower = ContingencyTable(text, "IBM";
    windowsize=5,
    minfreq=1,
    norm_config=TextNorm(strip_case=true))
results_lower = assoc_score(PMI, ct_lower)

println("With case preservation: $(nrow(results_case)) collocates")
println("With lowercasing: $(nrow(results_lower)) collocates")
```

### Punctuation Handling

```@example punctuation
using TextAssociations
using TextAnalysis: text

text1 = "Well-designed, user-friendly interface; however, performance issues..."

# Different punctuation strategies
configs = [
    ("Remove", TextNorm(strip_punctuation=true, punctuation_to_space=false)),
    ("To space", TextNorm(strip_punctuation=true, punctuation_to_space=true)),
    ("Keep", TextNorm(strip_punctuation=false))
]

for (name, config) in configs
    doc = prep_string(text1, config)
    println("$name: '$(text(doc))'")
end
```

### Whitespace Normalization

```@example whitespace
using TextAssociations
using TextAnalytics: text

text1 = "Multiple   spaces    and\t\ttabs\n\neverywhere"

# Normalize whitespace
normalized = prep_string(text1, TextNorm(normalize_whitespace=true))
println("Normalized: '$(text(normalized))'")

# Strip all whitespace (for certain languages)
stripped = prep_string(text1, TextNorm(strip_whitespace=true))
println("Stripped: '$(text(stripped))'")
```

### Accent Stripping

Critical for multilingual analysis:

```@example accents
using TextAssociations
using TextAnalysis: text

# Greek text with tonos marks
greek = "Η ανάλυση κειμένου είναι σημαντική"

# French text with accents
french = "L'analyse détaillée révèle des résultats intéressants"

# Spanish text
spanish = "El análisis lingüístico computacional avanzó rápidamente"

function compare_accent_handling(s::String, lang::String)
    println("\n$lang text:")

    # With accents
    with_config = TextNorm(strip_accents=false)
    with_doc = prep_string(s, with_config)
    println("  With accents: '$(text(with_doc))'")

    # Without accents
    without_config = TextNorm(strip_accents=true)
    without_doc = prep_string(s, without_config)
    println("  Without accents: '$(text(without_doc))'")
end

compare_accent_handling(greek, "Greek")
compare_accent_handling(french, "French")
compare_accent_handling(spanish, "Spanish")
```

## Unicode Normalization

### Understanding Unicode Forms

```@example unicode_forms
using TextAssociations, Unicode

# Same character in different forms
text_nfc = "café"  # NFC: é as single character
text_nfd = Unicode.normalize("café", :NFD)  # NFD: e + combining accent

println("Visual: both look like 'café'")
println("NFC length: $(length(text_nfc))")
println("NFD length: $(length(text_nfd))")
println("Equal? ", text_nfc == text_nfd)

# TextNorm handles this automatically
config_nfc = TextNorm(unicode_form=:NFC)
config_nfd = TextNorm(unicode_form=:NFD)
```

### Choosing Unicode Forms

| Form  | Use Case              | Example                 |
| ----- | --------------------- | ----------------------- |
| :NFC  | Default, most compact | General text            |
| :NFD  | Accent stripping      | Multilingual processing |
| :NFKC | Normalize variants    | Social media text       |
| :NFKD | Maximum decomposition | Special characters      |

## Language-Specific Preprocessing

### Greek Text

```@example greek_prep
using TextAssociations
using DataFrames: eachrow
using TextAnalysis: text

greek_text = """
Η φιλοσοφία και η επιστήμη συνδέονται στενά.
Οι Έλληνες φιλόσοφοι επηρέασαν τη σκέψη.
"""

# Greek-specific configuration
greek_config = TextNorm(
    strip_case=true,      # Greek has case
    strip_accents=true,   # Remove tonos/dialytika
    unicode_form=:NFD,    # Better accent stripping
    strip_punctuation=true
)

ct = ContingencyTable(greek_text, "φιλοσοφία"; windowsize=5, minfreq=1,
    norm_config=greek_config)
results = assoc_score(PMI, ct)

println("Greek collocations (normalized):")
for row in eachrow(results)
    println("  $(row.Collocate): PMI=$(round(row.PMI, digits=2))")
end
```

### Chinese/Japanese Text

```@example cjk_prep
using TextAssociations

# Chinese text (no spaces between words)
chinese = "机器学习是人工智能的重要组成部分"

# Japanese (mixed scripts)
japanese = "機械学習はAIの重要な分野です"

# CJK-specific configuration
cjk_config = TextNorm(
    strip_case=false,     # No case in CJK
    strip_accents=false,  # No accents
    strip_whitespace=true, # Remove spaces
    strip_punctuation=true
)

# Note: Proper CJK processing would require word segmentation
```

### Arabic Text

```@example arabic_prep
using TextAssociations

arabic = "التعلم الآلي يحول البيانات إلى معرفة"

# Arabic-specific configuration
arabic_config = TextNorm(
    strip_case=false,      # No case in Arabic
    strip_accents=false,   # Keep diacritics
    unicode_form=:NFC,     # Standard form
    strip_punctuation=true,
    normalize_whitespace=true
)

# Right-to-left text handling is automatic in Julia
```

## Advanced Preprocessing

### Custom Preprocessing Pipeline

```@example custom_pipeline
using TextAssociations
using TextAnalytics: text

function custom_preprocess(s::String)
    # Step 1: Remove URLs
    s = replace(text, r"https?://[^\s]+" => "[URL]")

    # Step 2: Remove email addresses
    s = replace(text, r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b" => "[EMAIL]")

    # Step 3: Expand contractions
    contractions = Dict(
        "don't" => "do not",
        "won't" => "will not",
        "can't" => "cannot",
        "n't" => " not",
        "'re" => " are",
        "'ve" => " have",
        "'ll" => " will",
        "'d" => " would",
        "'m" => " am"
    )

    for (contraction, expansion) in contractions
        s = replace(s, contraction => expansion)
    end

    # Step 4: Standard normalization
    config = TextNorm(
        strip_case=true,
        strip_punctuation=true,
        normalize_whitespace=true
    )

    doc = prep_string(s, config)
    return text(doc)
end

# Test custom preprocessing
test_text = "Don't forget to check https://example.com and email me at user@example.com"
processed = custom_preprocess(test_text)
println("Original: $test_text")
println("Processed: $processed")
```

### Handling Special Characters

```@example special_chars
using TextAssociations
using TextAnalysis: text

text_with_special = "Price: \$99.99 | Temperature: 25°C | Math: x² + y² = r²"

# Different strategies for special characters
configs = [
    ("Keep symbols", TextNorm(strip_punctuation=false)),
    ("Remove symbols", TextNorm(strip_punctuation=true)),
    ("Normalize", TextNorm(unicode_form=:NFKC))  # Converts ² to 2
]

for (name, config) in configs
    doc = prep_string(text_with_special, config)
    println("$name: '$(text(doc))'")
end
```

## Performance Considerations

### Preprocessing Impact on Speed

```@example perf_prep
using TextAssociations
using BenchmarkTools

text = repeat("Sample text for benchmarking. ", 1000)

# Minimal preprocessing
minimal = TextNorm(
    strip_case=false,
    strip_punctuation=false,
    normalize_whitespace=false
)

# Standard preprocessing
standard = TextNorm()

# Heavy preprocessing
heavy = TextNorm(
    strip_case=true,
    strip_accents=true,
    strip_punctuation=true,
    normalize_whitespace=true
)

# Compare performance
configs = [
    ("Minimal", minimal),
    ("Standard", standard),
    ("Heavy", heavy)
]

for (name, config) in configs
    time = @elapsed prep_string(text, config)
    println("$name: $(round(time*1000, digits=2))ms")
end
```

### Memory Usage

```@example memory_prep
using TextAssociations

# Memory-efficient preprocessing for large texts
function stream_preprocess(file_path::String, chunk_size::Int=1024*1024)
    config = TextNorm()

    open(file_path, "r") do io
        while !eof(io)
            chunk = read(io, chunk_size)
            chunk_text = String(chunk)

            # Process chunk
            doc = prep_string(chunk_text, config)

            # Yield processed chunk (in practice, write to output)
            # process_chunk(text(doc))
        end
    end
end

println("Stream processing implemented for large files")
```

## Validation and Testing

### Preprocessing Verification

```@example verify_prep
using TextAssociations
using TextAnalysis: text

function verify_preprocessing(original::String, config::TextNorm)
    processed = prep_string(original, config)
    processed_text = text(processed)

    println("Original: '$original'")
    println("Processed: '$processed_text'")
    println("Changes:")

    # Check case changes
    if config.strip_case && original != lowercase(original)
        println("  ✓ Case normalized")
    end

    # Check punctuation removal
    if config.strip_punctuation && occursin(r"[[:punct:]]", original)
        if !occursin(r"[[:punct:]]", processed_text)
            println("  ✓ Punctuation removed")
        end
    end

    # Check whitespace normalization
    if config.normalize_whitespace && occursin(r"\s{2,}", original)
        if !occursin(r"\s{2,}", processed_text)
            println("  ✓ Whitespace normalized")
        end
    end

    return processed_text
end

test = "HELLO,  World!!!   Multiple   spaces..."
config = TextNorm()
verify_preprocessing(test, config)
```

## Best Practices

### 1. Choose Appropriate Settings

```julia
# Research/academic text
ACADEMIC_CONFIG = TextNorm(
    strip_case=true,
    strip_punctuation=true,
    normalize_whitespace=true,
    strip_accents=false  # Preserve author names
)

# Social media text
SOCIAL_CONFIG = TextNorm(
    strip_case=true,
    strip_punctuation=false,  # Keep hashtags, mentions
    normalize_whitespace=true,
    unicode_form=:NFKC  # Normalize variants
)

# Multilingual text
MULTILINGUAL_CONFIG = TextNorm(
    strip_case=true,
    strip_accents=true,  # For cross-language matching
    unicode_form=:NFD,
    normalize_whitespace=true
)
```

### 2. Document Your Choices

Always document preprocessing decisions:

```julia
# Save configuration with results
function save_analysis_with_config(results::DataFrame, config::TextNorm, file::String)
    # Add preprocessing metadata
    metadata!(results, "preprocessing", config, style=:note)

    # Save to file
    CSV.write(file, results)

    # Save config separately
    config_file = replace(file, ".csv" => "_config.txt")
    open(config_file, "w") do io
        for field in fieldnames(TextNorm)
            println(io, "$field: $(getfield(config, field))")
        end
    end
end
```

### 3. Test Edge Cases

```julia
# Test suite for preprocessing
test_cases = [
    "normal text",
    "UPPERCASE TEXT",
    "MiXeD cAsE",
    "text-with-hyphens",
    "text_with_underscores",
    "email@example.com",
    "https://example.com",
    "café résumé naïve",
    "emoji 😀 text 🎉",
    "   extra    spaces   ",
    "text\twith\ttabs",
    "multi\nline\ntext"
]

function test_preprocessing(config::TextNorm)
    for test in test_cases
        processed = text(prep_string(test, config))
        println("'$test' → '$processed'")
    end
end
```

## Troubleshooting

### Common Issues

| Issue                | Cause                         | Solution                         |
| -------------------- | ----------------------------- | -------------------------------- |
| Words not matching   | Different Unicode forms       | Use consistent `unicode_form`    |
| Missing collocations | Over-aggressive preprocessing | Reduce stripping options         |
| Too much noise       | Insufficient preprocessing    | Enable more normalization        |
| Accent issues        | Inconsistent accent handling  | Set `strip_accents` consistently |

### Debug Helper

```@example debug_prep
using TextAssociations: TextNorm, prep_string, normalize_node
using TextAnalysis: text

function debug_preprocessing(s::String, word::String, config::TextNorm)
    # Show original
    println("Original text: '$s'")
    println("Looking for: '$word'")

    # Process text
    processed = prep_string(s, config)
    processed_text = text(processed)
    println("\nProcessed text: '$processed_text'")

    # Normalize word
    normalized_word = normalize_node(word, config)
    println("Normalized word: '$normalized_word'")

    # Check if word exists
    tokens = split(lowercase(processed_text))
    found = normalized_word in tokens
    println("\nWord found: $found")

    if !found
        prefix_len = min(3, length(normalized_word))
        # Find similar words
        similar = filter(t -> startswith(t, normalized_word[1:prefix_len]), tokens)
        if !isempty(similar)
            println("Similar words: ", similar)
        end
    end
end

debug_preprocessing("Café serves coffee", "cafe", TextNorm(strip_accents=false))
```

## Next Steps

- Learn about [Choosing Metrics](choosing_metrics.md) for your analysis
- Explore [Working with Corpora](corpus_analysis.md)
- See [Examples](../getting_started/examples.md) of preprocessing in action
