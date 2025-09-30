# Core Concepts

```
@id guide_concepts
```

```@meta
CurrentModule = TextAssociations
```

Understanding the fundamental concepts behind word association analysis is crucial for effective use of TextAssociations.jl.

## Word Co-occurrence

### Definition

Word co-occurrence is the foundation of collocation analysis. Two words co-occur when they appear near each other in text, within a defined window.

```@example cooc
using TextAssociations

text = "The data scientist analyzed the data carefully."

# Visualize co-occurrence windows
function show_cooccurrences(text::String, node::String, windowsize::Int)
    words = split(lowercase(text))
    node_positions = findall(==(lowercase(node)), words)

    println("Text: $text")
    println("Node word: '$node' at positions $node_positions")
    println("Window size: $windowsize")

    for pos in node_positions
        left_window = max(1, pos - windowsize)
        right_window = min(length(words), pos + windowsize)

        context = words[left_window:right_window]
        println("\nWindow around position $pos:")
        println("  ", join(context, " "))
    end
end

show_cooccurrences(text, "data", 2)
```

### Context Windows

The window size determines how far from the node word we look for collocates:

- **Small windows (1-3)**: Capture syntactic relations (adjective-noun, verb-object)
- **Medium windows (4-7)**: Balance syntactic and semantic relations
- **Large windows (8+)**: Capture semantic/topical associations

## Contingency Tables

### The 2×2 Table

Association metrics are calculated from contingency tables that count co-occurrences:

```@example contingency
using TextAssociations, DataFrames

# Create a simple example
text = "big data and data science require data analysis"
ct = ContingencyTable(text, "data"; windowsize=2, minfreq=1)

# Access the internal table
internal = cached_data(ct.con_tbl)
if !isempty(internal)
    println("Contingency table for 'data':")
    for row in eachrow(internal)
        println("\nCollocate: $(row.Collocate)")
        println("  a (both occur): $(row.a)")
        println("  b (only node): $(row.b)")
        println("  c (only collocate): $(row.c)")
        println("  d (neither): $(row.d)")
        println("  Total (N): $(row.N)")
    end
end
```

### Understanding the Cells

For each word pair (node, collocate):

| Cell | Meaning                 | Interpretation                 |
| ---- | ----------------------- | ------------------------------ |
| a    | Co-occurrence frequency | How often they appear together |
| b    | Node without collocate  | Node appears alone             |
| c    | Collocate without node  | Collocate appears alone        |
| d    | Neither appears         | Rest of the corpus             |

## Association Metrics

### Metric Categories

Different metrics capture different aspects of word association:

```@example metrics
using TextAssociations

# Demonstrate different metric properties
text = """
The bank provides financial services.
The river bank was steep and muddy.
Financial analysis requires careful consideration.
The bank offers investment opportunities.
"""

ct = ContingencyTable(text, "bank"; windowsize=3, minfreq=1)

# Calculate different metric types
info_metrics = assoc_score([PMI, PPMI], ct)
stat_metrics = assoc_score([LLR, ChiSquare], ct)
sim_metrics = assoc_score([Dice, JaccardIdx], ct)

println("Information-theoretic metrics (PMI, PPMI):")
println("  Focus: Surprise/informativeness")
println("  High when: Words occur together more than chance")

println("\nStatistical metrics (LLR, ChiSquare):")
println("  Focus: Significance/reliability")
println("  High when: Association is statistically significant")

println("\nSimilarity metrics (Dice, Jaccard):")
println("  Focus: Overlap/similarity")
println("  High when: Words share contexts")
```

### Interpreting Scores

```@example interpret
using TextAssociations
using DataFrames

# Score interpretation guidelines
function interpret_scores(results::DataFrame)
    for row in eachrow(results)
        collocate = row.Collocate

        # PMI interpretation
        pmi_strength = if row.PMI > 5
            "very strong"
        elseif row.PMI > 3
            "strong"
        elseif row.PMI > 0
            "positive"
        else
            "negative"
        end

        # LogDice interpretation (max 14)
        dice_reliability = if row.LogDice > 10
            "highly reliable"
        elseif row.LogDice > 7
            "reliable"
        else
            "weak"
        end

        println("$collocate:")
        println("  PMI: $(round(row.PMI, digits=2)) ($pmi_strength association)")
        println("  LogDice: $(round(row.LogDice, digits=2)) ($dice_reliability)")
    end
end

# Example
ct = ContingencyTable("machine learning uses learning algorithms", "learning"; windowsize=2, minfreq=1)
results = assoc_score([PMI, LogDice], ct)
interpret_scores(results)
```

## Text Normalization

### The TextNorm Configuration

Text preprocessing is controlled by the `TextNorm` struct:

```@example textnorm
using TextAssociations

# Different normalization strategies
configs = [
    (name="Minimal",
     config=TextNorm(strip_case=false, strip_punctuation=false)),
    (name="Standard",
     config=TextNorm(strip_case=true, strip_punctuation=true)),
    (name="Aggressive",
     config=TextNorm(strip_case=true, strip_punctuation=true,
                    strip_accents=true, normalize_whitespace=true))
]

test_text = "Hello, WORLD! Café résumé... Multiple   spaces."

for (name, config) in configs
    doc = prep_string(test_text, config)
    println("$name: '$(text(doc))'")
end
```

### Unicode Normalization

Important for multilingual text:

```@example unicode
using TextAssociations
using Unicode

# Different Unicode forms can affect matching
text1 = "café"  # é as single character
text2 = "café"  # e + combining accent

println("Visually identical: ", text1 == text2)
println("After NFC normalization: ",
    Unicode.normalize(text1, :NFC) == Unicode.normalize(text2, :NFC))

# TextNorm handles this automatically
config = TextNorm(unicode_form=:NFC)
```

## Frequency Thresholds

### Minimum Frequency Parameter

The `minfreq` parameter filters noise:

```@example minfreq
using TextAssociations

text = """
The main hypothesis was confirmed.
Preliminary results support the hypothesis.
The xyzabc appeared only once.
"""

# Compare different thresholds
for minfreq in [1, 2, 3]
    ct = ContingencyTable(text, "the"; windowsize=3, minfreq=minfreq)
    results = assoc_score(PMI, ct)
    println("minfreq=$minfreq: $(nrow(results)) collocates")
end
```

### Choosing Appropriate Thresholds

Guidelines for setting `minfreq`:

| Corpus Size    | Recommended minfreq | Rationale             |
| -------------- | ------------------- | --------------------- |
| < 1,000 words  | 1-2                 | Preserve all data     |
| 1,000-10,000   | 3-5                 | Filter hapax legomena |
| 10,000-100,000 | 5-10                | Remove noise          |
| > 100,000      | 10-20               | Focus on patterns     |

## Lazy Evaluation

### How LazyProcess Works

TextAssociations.jl uses lazy evaluation for efficiency:

```@example lazy
using TextAssociations

# Contingency tables are computed lazily
println("Creating ContingencyTable...")
ct = ContingencyTable("sample text here", "text"; windowsize=3, minfreq=1)
println("Created (not computed yet)")

# Computation happens on first use
println("\nFirst access (triggers computation):")
@time results = assoc_score(PMI, ct)

println("\nSecond access (uses cache):")
@time results2 = assoc_score(LogDice, ct)
```

### Benefits

1. **Memory efficiency**: Data computed only when needed
2. **Performance**: Cached results for multiple metrics
3. **Flexibility**: Chain operations without intermediate computation

## Statistical Significance

### Understanding P-values

Some metrics provide statistical significance:

```@example significance
using TextAssociations

text = """
Statistical analysis requires careful statistical methods.
The statistical approach yields statistical significance.
Random words appear randomly without pattern.
"""

ct = ContingencyTable(text, "statistical"; windowsize=3, minfreq=1)
results = assoc_score([LLR, ChiSquare], ct)

# Interpret statistical significance
for row in eachrow(results)
    llr = row.LLR
    chi2 = row.ChiSquare

    # LLR critical values
    p_value = if llr > 10.83
        "p < 0.001"
    elseif llr > 6.63
        "p < 0.01"
    elseif llr > 3.84
        "p < 0.05"
    else
        "not significant"
    end

    println("$(row.Collocate): LLR=$(round(llr, digits=2)) ($p_value)")
end
```

## Best Practices

### 1. Metric Selection

```julia
# For discovery
discovery_metrics = [PMI, PPMI]

# For validation
validation_metrics = [LLR, ChiSquare]

# For comparison across corpora
stable_metrics = [LogDice, PPMI]
```

### 2. Parameter Guidelines

```julia
# Default parameters for different purposes
const SYNTAX_PARAMS = (windowsize=2, minfreq=5)
const SEMANTIC_PARAMS = (windowsize=5, minfreq=5)
const TOPIC_PARAMS = (windowsize=10, minfreq=10)
```

### 3. Validation Strategy

Always validate findings with multiple approaches:

1. Use multiple metrics
2. Check different window sizes
3. Examine concordance lines
4. Compare with domain knowledge

## Next Steps

- Learn about [Text Preprocessing](preprocessing.md) options
- Understand [Choosing Metrics](choosing_metrics.md) for your task
- Explore [Working with Corpora](corpus_analysis.md)
