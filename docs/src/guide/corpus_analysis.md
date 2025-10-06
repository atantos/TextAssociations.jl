# [Working with Corpora](@id corpus_analysis_guide)

```@meta
CurrentModule = TextAssociations
```

This guide covers corpus-level analysis, from loading documents to advanced corpus operations.

## Loading Corpora

### From Various Sources

```@example loading
using TextAssociations
using DataFrames

# Create example files
temp_dir = mktempdir()

# Machine learning transforms data into insights.

# Create sample text files
texts = [
    "Machine learning transforms data into insights. It parses corpora at scale, aligns term frequencies with semantic relationships, and highlights the associations that warrant closer study. Each iteration refines the embeddings, letting nuanced usage patterns surface from raw text streams. Within minutes, what began as unstructured prose becomes an interpretable map of concepts, trends, and contextual signals your analysts can act on. Continual learning cycles tie human feedback to automated scoring, ensuring each model update remains accountable across the network. In parallel, learning analytics expose which network nodes contribute novel context, letting curators rebalance data pipelines. As the knowledge graph expands, network observers watch federated network edges synchronize in real time, while downstream services tap network dashboards that keep learning teams aligned.",
    "Deep learning uses neural networks extensively. Layers of nonlinear transformations enable the models to capture complex patterns, extracting latent representations that conventional features miss. When trained on rich corpora, the networks adapt to domain-specific nuances, delivering higher accuracy in tasks like entity recognition and sentiment analysis. With appropriate regularization and interpretability tools, practitioners can translate these dense embeddings into actionable insights while maintaining trust in the system’s predictions. Adaptive learning pipelines coordinate semantic parsers with a distributed network of annotators. Each network ingests curated corpora, while a secondary network synchronizes contextual metadata across regional clusters. During offline learning, analysts probe the recommendation network for bias signals, then trigger online learning updates that reweight features without stalling the monitoring network.",
    "Data science combines statistics and programming. Analysts unify probabilistic models with code-driven automation to surface trends that raw tables conceal. This fusion accelerates experimentation, supports reproducible pipelines, and turns exploratory questions into measurable, actionable metrics across evolving datasets. Integrated learning cohorts audit each network channel to confirm data provenance. A governance network curates feature stores while a delivery network pushes dashboards to product teams. Scenario-based learning guides analysts as a simulation network harmonizes with a resilience network that shields critical pipelines. Hands-on learning labs document their findings for future audits."
]

for (i, text) in enumerate(texts)
    write(joinpath(temp_dir, "doc$i.txt"), text)
end

# Load from directory
corpus = read_corpus(temp_dir;
    norm_config=TextNorm(strip_case=true, strip_punctuation=true),
    min_doc_length=5,
    max_doc_length=1000
)

println("Loaded $(length(corpus.documents)) documents")
println("Vocabulary size: $(length(corpus.vocabulary))")

# Clean up
rm(temp_dir, recursive=true)
```

### From DataFrames

```@example df_loading
using TextAssociations, DataFrames

# Create a DataFrame with text and metadata
df = DataFrame(
    text = [
        "Artificial intelligence revolutionizes technology.",
        "Machine learning enables pattern recognition.",
        "Deep learning mimics human neural networks."
    ],
    category = ["AI", "ML", "DL"],
    year = [2023, 2023, 2024],
    importance = ["high", "high", "medium"]
)

# Load corpus from DataFrame
corpus = read_corpus_df(df;
    text_column=:text,
    metadata_columns=[:category, :year, :importance],
    norm_config=TextNorm()
)

println("Corpus from DataFrame:")
println("  Documents: $(length(corpus.documents))")
println("  Metadata fields: $(keys(corpus.metadata))")
```

## Corpus Statistics

### Basic Statistics

```@example stats
using TextAssociations
using TextAnalysis: StringDocument

# Create a sample corpus
texts = [
    "Natural language processing enables computers to understand human language.",
    "Machine learning algorithms learn patterns from data automatically.",
    "Deep neural networks consist of multiple hidden layers.",
    "Artificial intelligence includes machine learning and deep learning."
]

docs = [StringDocument(t) for t in texts]
corpus = Corpus(docs)

# Get comprehensive statistics
stats = corpus_stats(corpus; include_token_distribution=true)

println("Corpus Statistics:")
println("  Documents: $(stats[:num_documents])")
println("  Total tokens: $(stats[:total_tokens])")
println("  Unique tokens: $(stats[:unique_tokens])")
println("  Type-token ratio: $(round(stats[:type_token_ratio], digits=4))")
println("  Hapax legomena: $(stats[:hapax_legomena])")
println("\nVocabulary coverage:")
println("  50% coverage: $(stats[:words_for_50_percent_coverage]) words")
println("  90% coverage: $(stats[:words_for_90_percent_coverage]) words")

# Display coverage summary
coverage_summary(stats)
```

### Token Distribution

```@example loading
using TextAssociations

# Analyze token distribution
dist = token_distribution(corpus)

println("\nTop 10 most frequent tokens:")
for row in eachrow(first(dist, 10))
    println("  $(row.Token): $(row.Frequency) (TF-IDF: $(round(row.TFIDF, digits=2)))")
end
```

## Corpus-Level Analysis

### Single Node Analysis

```@example loading
using TextAssociations

# Analyze a single word across the corpus
results = analyze_corpus(
    corpus,
    "learning",
    PMI;
    windowsize=5,
    minfreq=1
)

println("Top collocates of 'learning' across corpus:")
for row in eachrow(first(results, 5))
    println("  $(row.Collocate): Score=$(round(row.Score, digits=2)), DocFreq=$(row.DocFrequency)")
end
```

### Multiple Nodes Analysis

```@example loading
using TextAssociations

# Analyze multiple nodes
nodes = ["machine", "learning", "neural"]
metrics = [PMI, LogDice, LLR]

analysis = analyze_nodes(
    corpus,
    nodes,
    metrics;
    windowsize=5,
    minfreq=1,
    top_n=10
)

# Access results for each node
for node in analysis.nodes
    node_results = analysis.results[node]
    if !isempty(node_results)
        println("\nTop collocates for '$node':")
        for row in eachrow(first(node_results, 3))
            println("  $(row.Collocate): PMI=$(round(row.PMI, digits=2))")
        end
    end
end
```

## Advanced Corpus Operations

### Temporal Analysis

```@example temporal
using TextAssociations, Dates, DataFrames

# Create DataFrame with temporal metadata
df = DataFrame(
    text = [
        "Early AI used rule-based systems.",
        "Machine learning emerged as dominant approach.",
        "Deep learning revolutionized the field.",
        "Transformers changed natural language processing."
    ],
    year = [1980, 1990, 2010, 2020]
)

# Use read_corpus_df to properly store metadata
temporal_corpus = read_corpus_df(df;
    text_column=:text,
    metadata_columns=[:year]
)

# Analyze temporal trends
temporal_analysis = analyze_temporal(
    temporal_corpus,
    ["AI", "learning"],
    :year,
    PMI;
    time_bins=2,
    windowsize=5,
    minfreq=1
)

println("Temporal Analysis Results:")
println("Time periods: ", temporal_analysis.time_periods)

if !isempty(temporal_analysis.trend_analysis)
    println("\nTrend analysis:")
    for row in eachrow(first(temporal_analysis.trend_analysis, 5))
        println("  $(row.Node) + $(row.Collocate): correlation=$(round(row.Correlation, digits=2))")
    end
end
```

### Subcorpus Comparison

```@example subcorpus
using TextAssociations, DataFrames

# Create corpus with categories
df = DataFrame(
    text = [
        "Scientific research requires rigorous methodology before the analysis is conducted.",
        "Business analysis focuses on market trends.",
        "Scientific experiments test hypotheses systematically and analyze the resulting data.",
        "Business strategy and analysis drives organizational success."
    ],
    field = ["Science", "Business", "Science", "Business"]
)

categorized_corpus = read_corpus_df(df;
    text_column=:text,
    metadata_columns=[:field]
)

# Compare subcorpora
comparison = compare_subcorpora(
    categorized_corpus,
    :field,
    "analysis",
    PMI;
    windowsize=5,
    minfreq=1
)

println("Subcorpus Comparison:")
for (subcorpus_name, results) in comparison.results
    if !isempty(results)
        println("\n$subcorpus_name subcorpus:")
        for row in eachrow(first(results, 2))
            println("  $(row.Collocate): Score=$(round(row.Score, digits=2))")
        end
    end
end
```

### Keyword Extraction

```@example loading
using TextAssociations

# Extract keywords using TF-IDF
keywords = keyterms(
    corpus;
    method=:tfidf,
    num_keywords=10,
    min_doc_freq=1,
    max_doc_freq_ratio=0.8
)

println("\nTop Keywords (TF-IDF):")
for row in eachrow(keywords)
    println("  $(row.Keyword): TFIDF=$(round(row.TFIDF, digits=2)), DocFreq=$(row.DocFreq)")
end
```

### Building Collocation Networks

```@example loading
using TextAssociations

# Build collocation network
network = colloc_graph(
    corpus,
    ["learning", "network"];  # Seed words
    metric=PMI,
    depth=1,
    min_score=-10.0,
    max_neighbors=5,
    windowsize=5,
    minfreq=1
)

println("\nCollocation Network:")
println("  Nodes: $(length(network.nodes))")
println("  Edges: $(nrow(network.edges))")

if !isempty(network.edges)
    println("\nStrongest connections:")
    for row in eachrow(first(sort(network.edges, :Weight, rev=true), 5))
        println("  $(row.Source) → $(row.Target): $(round(row.Weight, digits=2))")
    end
end
```

## Memory-Efficient Processing

### Batch Processing

```@example loading
using TextAssociations

function batch_analyze_corpus(corpus::Corpus, nodes::Vector{String}, batch_size::Int=10)
    all_results = Dict{String, DataFrame}()

    for batch_start in 1:batch_size:length(nodes)
        batch_end = min(batch_start + batch_size - 1, length(nodes))
        batch_nodes = nodes[batch_start:batch_end]

        println("Processing batch: nodes $batch_start-$batch_end")

        # Analyze batch
        batch_analysis = analyze_nodes(
            corpus, batch_nodes, [PMI];
            windowsize=5, minfreq=1
        )

        # Store results
        for (node, results) in batch_analysis.results
            all_results[node] = results
        end

        # Force garbage collection between batches
        GC.gc()
    end

    return all_results
end

# Example with many nodes
many_nodes = ["machine", "learning", "deep", "neural", "network",
              "algorithm", "data", "pattern"]

batch_results = batch_analyze_corpus(corpus, many_nodes, 3)
println("\nBatch processing complete: $(length(batch_results)) nodes analyzed")
```

### Streaming Analysis

```@example streaming
using TextAssociations

function stream_analyze(file_pattern::String, node::String)
    aggregated_scores = Dict{String, Float64}()
    doc_count = 0

    # Process files one at a time
    for file in glob(file_pattern)
        # Read single file
        text = read(file, String)

        # Analyze
        ct = ContingencyTable(text, node; windowsize=5, minfreq=1)
        results = assoc_score(PMI, ct; scores_only=false)

        # Aggregate results
        for row in eachrow(results)
            collocate = String(row.Collocate)
            score = row.PMI

            # Running average
            current = get(aggregated_scores, collocate, 0.0)
            aggregated_scores[collocate] = (current * doc_count + score) / (doc_count + 1)
        end

        doc_count += 1
    end

    return aggregated_scores, doc_count
end

println("Streaming analysis function defined for large corpora")
```

## Corpus Filtering and Sampling

### Document Filtering

```@example loading
using TextAssociations
using TextAnalysis

docs = [
    StringDocument("Machine learning algorithms learn from data."),
    StringDocument("Deep learning uses neural networks."),
    StringDocument("AI includes machine learning.")
]
corpus = Corpus(docs)

function filter_corpus(corpus::Corpus, min_length::Int, max_length::Int)
    filtered_docs = StringDocument{String}[]  # Specify type

    for doc in corpus.documents
        doc_length = length(tokens(doc))
        if min_length <= doc_length <= max_length
            push!(filtered_docs, doc)
        end
    end

    return Corpus(filtered_docs, norm_config=corpus.norm_config)
end

filtered = filter_corpus(corpus, 5, 15)
println("Filtered: $(length(filtered.documents)) documents")
```

### Vocabulary Filtering

```@example loading
using TextAssociations
using TextAnalysis: tokens
using OrderedCollections

function filter_vocabulary(corpus::Corpus, min_freq::Int, max_freq_ratio::Float64)
    # Count token frequencies
    token_counts = Dict{String, Int}()

    for doc in corpus.documents
        for token in tokens(doc)
            token_counts[token] = get(token_counts, token, 0) + 1
        end
    end

    # Filter vocabulary
    total_docs = length(corpus.documents)
    max_freq = total_docs * max_freq_ratio

    filtered_vocab = OrderedDict{String, Int}()
    idx = 0

    for (token, count) in token_counts
        if min_freq <= count <= max_freq
            idx += 1
            filtered_vocab[token] = idx
        end
    end

    println("Vocabulary filtered: $(length(corpus.vocabulary)) → $(length(filtered_vocab))")

    return filtered_vocab
end

filtered_vocab = filter_vocabulary(corpus, 1, 0.8)
```

## Export and Persistence

### Saving Results

```@example loading
using TextAssociations, CSV, Dates

# Analyze and save results
results = analyze_corpus(corpus, "learning", PMI, windowsize=3, minfreq=2)

# Save to CSV
temp_file = tempname() * ".csv"
CSV.write(temp_file, results)
println("Results saved to temporary file")

# Save with metadata
results_with_meta = copy(results)
metadata!(results_with_meta, "corpus_size", length(corpus.documents), style=:note)
metadata!(results_with_meta, "analysis_date", Dates.today(), style=:note)

# Clean up
rm(temp_file)
```

### Multi-format Export

```@example multiformat
using TextAssociations

function export_analysis(analysis::MultiNodeAnalysis, base_path::String)
    # Export as CSV
    write_results(analysis, base_path * ".csv"; format=:csv)

    # Export as JSON
    write_results(analysis, base_path * ".json"; format=:json)

    # Export summary
    summary = DataFrame(
        Node = analysis.nodes,
        NumCollocates = [nrow(analysis.results[n]) for n in analysis.nodes],
        WindowSize = analysis.parameters[:windowsize],
        MinFreq = analysis.parameters[:minfreq]
    )

    CSV.write(base_path * "_summary.csv", summary)

    println("Exported to multiple formats")
end

# Example (would create files)
# export_analysis(analysis, "corpus_analysis")
```

## Performance Optimization

### Corpus Size Guidelines

| Corpus Size     | Recommended Approach | Memory Usage | Processing Time |
| --------------- | -------------------- | ------------ | --------------- |
| < 100 docs      | Load all in memory   | ~10MB        | < 1s            |
| 100-1000 docs   | Standard processing  | ~100MB       | < 10s           |
| 1000-10000 docs | Batch processing     | ~500MB       | < 1min          |
| > 10000 docs    | Streaming            | Constant     | Linear          |

### Optimization Tips

```julia
# 1. Pre-filter vocabulary
const MIN_WORD_LENGTH = 2
const MAX_WORD_LENGTH = 20

# 2. Use appropriate data structures
const USE_SPARSE_MATRIX = true  # For large vocabularies

# 3. Optimize window sizes
const OPTIMAL_WINDOW = Dict(
    :syntactic => 2,
    :semantic => 5,
    :topical => 10
)
```

## Troubleshooting

### Common Issues

```@example loading
using TextAssociations
using TextAnalysis: tokens
using Statistics

function diagnose_corpus(corpus::Corpus)
    println("Corpus Diagnostics:")
    println("="^40)

    # Check document distribution
    doc_lengths = [length(tokens(doc)) for doc in corpus.documents]
    println("Document lengths:")
    println("  Min: $(minimum(doc_lengths))")
    println("  Max: $(maximum(doc_lengths))")
    println("  Mean: $(round(mean(doc_lengths), digits=1))")

    # Check vocabulary
    println("\nVocabulary:")
    println("  Size: $(length(corpus.vocabulary))")

    # Check for issues
    if minimum(doc_lengths) < 5
        println("\n⚠ Warning: Very short documents detected")
    end

    if maximum(doc_lengths) > 10000
        println("\n⚠ Warning: Very long documents may slow processing")
    end

    if length(corpus.vocabulary) > 100000
        println("\n⚠ Warning: Large vocabulary may require more memory")
    end
end

diagnose_corpus(corpus)
```

## Next Steps

- Explore [Temporal Analysis](../advanced/temporal.md) for time-based patterns
- Learn about [Network Analysis](../advanced/networks.md) for visualization
- See [Performance](../performance.md) guide for large-scale processing
