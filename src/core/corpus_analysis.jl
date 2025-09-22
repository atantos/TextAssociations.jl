# =====================================
# File: src/core/corpus_analysis.jl
# Corpus-level analysis functionality
# =====================================

using CSV
using DataFrames
using Distributed: @distributed
using Glob: glob
using JSON
using Statistics: mean, median
using TextAnalysis
using XLSX
using Printf: @sprintf

"""
    Corpus <: AssociationDataFormat

Represents a collection of documents for corpus-level analysis.
"""
struct Corpus <: AssociationDataFormat
    documents::Vector{StringDocument{String}}
    metadata::Dict{String,Any}
    vocabulary::OrderedDict{String,Int}
    doc_term_matrix::Union{Nothing,SparseMatrixCSC}

    function Corpus(docs::Vector{StringDocument{String}};
        build_dtm::Bool=false,
        metadata::Dict{String,Any}=Dict{String,Any}())

        # Build vocabulary
        all_tokens = String[]
        for doc in docs
            append!(all_tokens, tokens(doc))
        end
        vocabulary = createvocab(unique(all_tokens))

        # Optionally build document-term matrix
        dtm = nothing
        if build_dtm
            dtm = build_document_term_matrix(docs, vocabulary)
        end

        new(docs, metadata, vocabulary, dtm)
    end
end

"""
    CorpusContingencyTable

Aggregated contingency table across an entire corpus.
"""
struct CorpusContingencyTable <: AssociationDataFormat
    tables::Vector{ContingencyTable}
    aggregated_table::LazyProcess{T,DataFrame} where T
    node::AbstractString
    windowsize::Int
    minfreq::Int64
    corpus_ref::Corpus

    function CorpusContingencyTable(corpus::Corpus,
        node::AbstractString,
        windowsize::Int,
        minfreq::Int64=5;
        strip_accents::Bool=false)  # <-- NEW

        # Create contingency tables for each document
        tables = ContingencyTable[]
        @showprogress desc = "Processing documents..." for doc in corpus.documents
            try
                # Always go through ContingencyTable’s string path so we can apply the toggle
                ct = ContingencyTable(text(doc), node, windowsize, 1;
                    auto_prep=true, strip_accents=strip_accents)
                push!(tables, ct)
            catch e
                @warn "Skipping document due to error: $e"
            end
        end

        # Lazy aggregation as before
        f = () -> aggregate_contingency_tables(tables, minfreq)
        aggregated = LazyProcess(f)

        new(tables, aggregated, node, windowsize, minfreq, corpus)
    end
end

"""
    MultiNodeAnalysis

Analysis results for multiple node words across a corpus.
"""
struct MultiNodeAnalysis
    nodes::Vector{String}
    results::Dict{String,DataFrame}
    corpus_ref::Corpus
    parameters::Dict{Symbol,Any}
end

# =====================================
# Corpus Loading Functions
# =====================================

"""
    load_corpus(path::AbstractString; kwargs...) -> Corpus

Load a corpus from various sources.

# Arguments
- `path`: Directory path, CSV file, or JSON file containing documents
- `text_column`: Column name for text (for CSV/JSON)
- `metadata_columns`: Columns to include as metadata
- `preprocess`: Apply preprocessing (default: true)
- `preprocess_options`: Dict or NamedTuple of preprocessing options
- `min_doc_length`: Minimum document length in tokens (default: 10)
- `max_doc_length`: Maximum document length in tokens (default: nothing)

# Preprocessing Options
Pass preprocessing options as a Dict or NamedTuple:
```julia
load_corpus("path/", preprocess_options=(strip_accents=true, strip_case=false))
# or
load_corpus("path/", preprocess_options=Dict(:strip_accents => true))
```

Available options (see prepstring for details):
- strip_punctuation (default: true)
- punctuation_to_space (default: true) 
- normalize_whitespace (default: true)
- strip_case (default: true)
- strip_accents (default: false)
- unicode_form (default: :NFC)
- use_prepare (default: false)
"""
function load_corpus(path::AbstractString;
    text_column::Symbol=:text,
    metadata_columns::Vector{Symbol}=Symbol[],
    preprocess::Bool=true,
    preprocess_options::Union{Dict,NamedTuple,Nothing}=nothing,
    min_doc_length::Int=10,
    max_doc_length::Union{Nothing,Int}=nothing)

    # Set up default preprocessing options
    default_prep_opts = (
        strip_punctuation=true,
        punctuation_to_space=true,
        normalize_whitespace=true,
        strip_case=true,
        strip_accents=false,
        unicode_form=:NFC,
        use_prepare=false
    )

    # Merge user options with defaults
    prep_opts = if preprocess_options === nothing
        default_prep_opts
    elseif isa(preprocess_options, Dict)
        merge(default_prep_opts, NamedTuple(preprocess_options))
    else
        merge(default_prep_opts, preprocess_options)
    end

    documents = StringDocument{String}[]
    metadata = Dict{String,Any}()

    if isdir(path)
        # Load from directory of text files
        files = filter(f -> endswith(f, ".txt"), readdir(path, join=true))
        @showprogress desc = "Loading files..." for file in files
            content = read_text_smart(file)

            # Apply preprocessing with options
            if preprocess
                doc = prepstring(content; prep_opts...)
                typed_doc = StringDocument(text(doc))
            else
                typed_doc = StringDocument(content)
            end

            # Check document length
            doc_tokens = tokens(typed_doc)
            if length(doc_tokens) >= min_doc_length &&
               (max_doc_length === nothing || length(doc_tokens) <= max_doc_length)
                push!(documents, typed_doc)
                metadata[basename(file)] = Dict(:source => file)
            end
        end

    elseif endswith(lowercase(path), ".csv")
        # Load from CSV
        df = DataFrame(CSV.File(path))

        @showprogress desc = "Processing CSV rows..." for row in eachrow(df)
            text_content = string(row[text_column])

            # Apply preprocessing with options
            if preprocess
                doc = prepstring(text_content; prep_opts...)
                typed_doc = StringDocument(text(doc))
            else
                typed_doc = StringDocument(text_content)
            end

            # Check document length
            doc_tokens = tokens(typed_doc)
            if length(doc_tokens) >= min_doc_length &&
               (max_doc_length === nothing || length(doc_tokens) <= max_doc_length)
                push!(documents, typed_doc)

                # Store metadata
                row_metadata = Dict{Symbol,Any}()
                for col in metadata_columns
                    if col in names(df)
                        row_metadata[col] = row[col]
                    end
                end
                metadata["doc_$(length(documents))"] = row_metadata
            end
        end

    elseif endswith(lowercase(path), ".json")
        # Load from JSON
        json_data = JSON.parsefile(path)

        if isa(json_data, Vector)
            @showprogress desc = "Processing JSON entries..." for (i, entry) in enumerate(json_data)
                text_content = string(get(entry, string(text_column), ""))
                if !isempty(text_content)
                    # Apply preprocessing with options
                    if preprocess
                        doc = prepstring(text_content; prep_opts...)
                        typed_doc = StringDocument(text(doc))
                    else
                        typed_doc = StringDocument(text_content)
                    end

                    # Check document length
                    doc_tokens = tokens(typed_doc)
                    if length(doc_tokens) >= min_doc_length &&
                       (max_doc_length === nothing || length(doc_tokens) <= max_doc_length)
                        push!(documents, typed_doc)
                        metadata["doc_$(length(documents))"] = entry
                    end
                end
            end
        end
    else
        throw(ArgumentError("Unsupported file format. Use directory, CSV, or JSON."))
    end

    println("Loaded $(length(documents)) documents")

    # Store preprocessing options in corpus metadata for reproducibility
    corpus = Corpus(documents, metadata=metadata)

    # You might want to store prep options in the corpus somehow
    # For example, add to the metadata dict:
    corpus.metadata["_preprocessing_options"] = Dict(pairs(prep_opts))

    return corpus
end

"""
    load_corpus_df(df::DataFrame; kwargs...) -> Corpus

Load corpus directly from a DataFrame.

# Arguments  
- `df`: DataFrame containing documents
- `text_column`: Column containing text (default: :text)
- `metadata_columns`: Columns to preserve as metadata
- `preprocess`: Whether to preprocess (default: true)
- `preprocess_options`: Dict or NamedTuple of preprocessing options

# Example
```julia
corpus = load_corpus_df(
    df,
    text_column=:content,
    metadata_columns=[:author, :date],
    preprocess_options=(strip_accents=true, strip_case=false)
)
```
"""
function load_corpus_df(df::DataFrame;
    text_column::Symbol=:text,
    metadata_columns::Vector{Symbol}=Symbol[],
    preprocess::Bool=true,
    preprocess_options::Union{Dict,NamedTuple,Nothing}=nothing)

    # Set up default preprocessing options
    default_prep_opts = (
        strip_punctuation=true,
        punctuation_to_space=true,
        normalize_whitespace=true,
        strip_case=true,
        strip_accents=false,
        unicode_form=:NFC,
        use_prepare=false
    )

    # Merge user options with defaults
    prep_opts = if preprocess_options === nothing
        default_prep_opts
    elseif isa(preprocess_options, Dict)
        merge(default_prep_opts, NamedTuple(preprocess_options))
    else
        merge(default_prep_opts, preprocess_options)
    end

    documents = StringDocument{String}[]
    metadata = Dict{String,Any}()

    @showprogress desc = "Processing DataFrame..." for (idx, row) in enumerate(eachrow(df))
        text_content = string(row[text_column])

        # Apply preprocessing with options
        if preprocess
            doc = prepstring(text_content; prep_opts...)
            typed_doc = StringDocument(text(doc))
        else
            typed_doc = StringDocument(text_content)
        end

        push!(documents, typed_doc)

        # Store metadata
        row_metadata = Dict{Symbol,Any}()
        for col in metadata_columns
            if col in names(df)
                row_metadata[col] = row[col]
            end
        end
        metadata["doc_$idx"] = row_metadata
    end

    corpus = Corpus(documents, metadata=metadata)

    # Store preprocessing options for reproducibility
    corpus.metadata["_preprocessing_options"] = Dict(pairs(prep_opts))

    return corpus
end

# =====================================
# Aggregation Functions
# =====================================

"""
    aggregate_contingency_tables(tables::Vector{ContingencyTable}, minfreq::Int) -> DataFrame

Aggregate multiple contingency tables into a single table.
"""
function aggregate_contingency_tables(tables::Vector{ContingencyTable}, minfreq::Int)
    if isempty(tables)
        return DataFrame()
    end

    # Collect all collocates across documents
    all_collocates = Set{Symbol}()
    for table in tables
        ct = extract_cached_data(table.con_tbl)
        !isempty(ct) && union!(all_collocates, ct.Collocate)
    end

    # Initialize aggregated data
    agg_data = Dict{Symbol,Vector{Int}}()
    for collocate in all_collocates
        agg_data[collocate] = zeros(Int, 4)  # [a, b, c, d]
    end

    # Aggregate across documents
    for table in tables
        ct = extract_cached_data(table.con_tbl)
        isempty(ct) && continue

        for row in eachrow(ct)
            collocate = row.Collocate
            if haskey(agg_data, collocate)
                agg_data[collocate][1] += row.a
                agg_data[collocate][2] += row.b
                agg_data[collocate][3] += row.c
                agg_data[collocate][4] += row.d
            end
        end
    end

    # Build aggregated DataFrame
    collocates = collect(keys(agg_data))
    a_vals = [agg_data[c][1] for c in collocates]
    b_vals = [agg_data[c][2] for c in collocates]
    c_vals = [agg_data[c][3] for c in collocates]
    d_vals = [agg_data[c][4] for c in collocates]

    agg_df = DataFrame(
        Collocate=collocates,
        a=a_vals,
        b=b_vals,
        c=c_vals,
        d=d_vals
    )

    # Filter by minimum frequency
    filter!(row -> row.a >= minfreq, agg_df)

    # Add derived columns
    @chain agg_df begin
        transform!([:a, :b] => ((a, b) -> a .+ b) => :m)
        transform!([:c, :d] => ((c, d) -> c .+ d) => :n)
        transform!([:a, :c] => ((a, c) -> a .+ c) => :k)
        transform!([:b, :d] => ((b, d) -> b .+ d) => :l)
        transform!([:m, :n] => ((m, n) -> m .+ n) => :N)
        transform!([:m, :k, :N] => ((m, k, N) -> (m .* k) ./ N) => :E₁₁)
        transform!([:m, :l, :N] => ((m, l, N) -> (m .* l) ./ N) => :E₁₂)
        transform!([:n, :k, :N] => ((n, k, N) -> (n .* k) ./ N) => :E₂₁)
        transform!([:n, :l, :N] => ((n, l, N) -> (n .* l) ./ N) => :E₂₂)
    end

    return agg_df
end

# =====================================
# Corpus Analysis Functions - UPDATED
# =====================================


"""
    analyze_corpus(corpus::Corpus, node::AbstractString, metric::Type{<:AssociationMetric};
                  windowsize::Int=5, minfreq::Int=5) -> DataFrame

Analyze a single node word across the entire corpus.
Returns DataFrame with Node, Collocate, Score, Frequency, and DocFrequency columns.
"""
function analyze_corpus(corpus::Corpus,
    node::AbstractString,
    metric::Type{<:AssociationMetric};
    windowsize::Int=5,
    minfreq::Int=5)

    # Create corpus contingency table
    cct = CorpusContingencyTable(corpus, node, windowsize, minfreq)

    # Evaluate metric on aggregated data - now returns DataFrame by default
    scores_df = evalassoc(metric, cct)

    # If no results, return empty DataFrame
    if nrow(scores_df) == 0
        return DataFrame(
            Node=String[],
            Collocate=Symbol[],
            Score=Float64[],
            Frequency=Int[],
            DocFrequency=Int[]
        )
    end

    # Get aggregated table for additional info
    agg_table = extract_cached_data(cct.aggregated_table)

    # The evalassoc already returns DataFrame with Node, Collocate, Frequency, and metric column
    # We just need to add DocFrequency and rename the metric column to Score

    # Calculate document frequency for each collocate
    doc_freq = [count(t -> begin
            ct = extract_cached_data(t.con_tbl)
            !isempty(ct) && col in ct.Collocate
        end, cct.tables) for col in scores_df.Collocate]

    # Build final result DataFrame
    result = DataFrame(
        Node=scores_df.Node,
        Collocate=scores_df.Collocate,
        Score=scores_df[!, Symbol(string(metric))],  # Extract the metric column as Score
        Frequency=scores_df.Frequency,
        DocFrequency=doc_freq
    )

    # Sort by Score column (descending)
    sort!(result, :Score, rev=true)

    # Add metadata about the analysis
    metadata!(result, "metric", string(metric), style=:note)
    metadata!(result, "node", node, style=:note)
    metadata!(result, "windowsize", windowsize, style=:note)
    metadata!(result, "minfreq", minfreq, style=:note)
    metadata!(result, "analysis_type", "corpus_analysis", style=:note)

    return result
end

"""
    analyze_multiple_nodes(corpus::Corpus,
                          nodes::Vector{String},
                          metrics::Vector{DataType};
                          windowsize::Int=5,
                          minfreq::Int=5,
                          top_n::Int=100,
                          parallel::Bool=false) -> MultiNodeAnalysis

Analyze multiple node words with multiple metrics across a corpus.
Each result DataFrame now includes the Node column and metadata.
"""
function analyze_multiple_nodes(corpus::Corpus,
    nodes::Vector{String},
    metrics::Vector{DataType};
    windowsize::Int=5,
    minfreq::Int=5,
    top_n::Int=100,
    parallel::Bool=false)

    results = Dict{String,DataFrame}()

    if parallel && nworkers() > 1
        # Parallel processing implementation would go here
        # (omitted for brevity)
    else
        # Sequential processing
        @showprogress desc = "Analyzing nodes..." for node in nodes
            # Create corpus contingency table
            cct = CorpusContingencyTable(corpus, node, windowsize, minfreq)

            # Get aggregated table
            agg_table = extract_cached_data(cct.aggregated_table)

            if !isempty(agg_table)
                # Evaluate all metrics using the new API
                metric_results = evalassoc(metrics, cct)

                if !isempty(metric_results)
                    # The evalassoc with multiple metrics returns DataFrame with all metric columns
                    # Keep top N by first metric
                    first_metric = Symbol(string(metrics[1]))
                    sort!(metric_results, first_metric, rev=true)
                    result = first(metric_results, min(top_n, nrow(metric_results)))

                    # Add metadata about the metrics used
                    metric_names = join(string.(metrics), ", ")
                    metadata!(result, "metrics", metric_names, style=:note)
                    metadata!(result, "node", node, style=:note)
                    metadata!(result, "windowsize", windowsize, style=:note)
                    metadata!(result, "minfreq", minfreq, style=:note)
                    metadata!(result, "top_n", top_n, style=:note)

                    results[node] = result
                else
                    results[node] = DataFrame()
                end
            else
                results[node] = DataFrame()
            end
        end
    end

    parameters = Dict(
        :windowsize => windowsize,
        :minfreq => minfreq,
        :metrics => metrics,
        :top_n => top_n
    )

    return MultiNodeAnalysis(nodes, results, corpus, parameters)
end

"""
    corpus_statistics(corpus::Corpus; 
                     include_token_distribution::Bool=true) -> Dict

Get comprehensive statistics about the corpus.
"""
function corpus_statistics(corpus::Corpus;
    include_token_distribution::Bool=true,
    # NEW: control normalization & accent stripping
    unicode_form::Symbol=:NFC,
    strip_accents::Bool=true)

    total_tokens = 0
    unique_tokens_set = Set{String}()
    doc_lengths = Int[]
    token_frequencies = Dict{String,Int}()

    for doc in corpus.documents
        doc_tokens = tokens(doc)

        # --- NEW: normalize (and optionally strip accents) per token ---
        toks = if strip_accents || unicode_form != :NFC
            [
                begin
                    s = Unicode.normalize(t, unicode_form)
                    strip_accents ? strip_diacritics(s; target_form=unicode_form) : s
                end for t in doc_tokens
            ]
        else
            doc_tokens
        end
        # ---------------------------------------------------------------

        total_tokens += length(toks)
        union!(unique_tokens_set, toks)
        push!(doc_lengths, length(toks))

        # Count token frequencies
        for token in toks
            token_frequencies[token] = get(token_frequencies, token, 0) + 1
        end
    end

    # Calculate type-token ratio (lexical diversity)
    type_token_ratio = length(unique_tokens_set) / total_tokens

    # Calculate hapax legomena (words appearing only once)
    hapax_count = count(freq -> freq == 1, values(token_frequencies))

    # Calculate vocabulary coverage
    sorted_freqs = sort(collect(values(token_frequencies)), rev=true)
    cumsum_freqs = cumsum(sorted_freqs)

    # Find how many words cover 25%, 50%, 75%, 90%, 95%, 99% of corpus
    twenty_five_percent_coverage = findfirst(x -> x >= total_tokens * 0.25, cumsum_freqs)
    fifty_percent_coverage = findfirst(x -> x >= total_tokens * 0.5, cumsum_freqs)
    seventy_five_percent_coverage = findfirst(x -> x >= total_tokens * 0.75, cumsum_freqs)
    ninety_percent_coverage = findfirst(x -> x >= total_tokens * 0.9, cumsum_freqs)
    ninety_five_percent_coverage = findfirst(x -> x >= total_tokens * 0.95, cumsum_freqs)
    ninety_nine_percent_coverage = findfirst(x -> x >= total_tokens * 0.99, cumsum_freqs)

    stats = Dict{Symbol,Any}(
        # Document statistics
        :num_documents => length(corpus.documents),
        :avg_doc_length => mean(doc_lengths),
        :median_doc_length => median(doc_lengths),
        :min_doc_length => minimum(doc_lengths),
        :max_doc_length => maximum(doc_lengths),
        :std_doc_length => std(doc_lengths),

        # Token statistics (now reflect normalized/stripped tokens)
        :total_tokens => total_tokens,
        :vocabulary_size => length(unique_tokens_set),  # ← UPDATED
        :unique_tokens => length(unique_tokens_set),

        # Lexical diversity metrics
        :type_token_ratio => type_token_ratio,
        # :standardized_ttr => mean(length(unique(collect(w))) / length(w) for w in partition(doc_tokens, window_size)), # to implement as a separate function cf. https://chatgpt.com/g/g-p-68a3278f66bc81918deff6fb2d51139c-adjectives-sentiment-analysis/c/68cf06d7-a9f4-832f-91df-f4f7945e1de5
        :root_ttr => type_token_ratio * sqrt(total_tokens),
        :hapax_legomena => hapax_count,
        :hapax_ratio => hapax_count / length(unique_tokens_set),

        # Vocabulary coverage (quartiles and key percentiles)
        :words_for_25_percent_coverage => twenty_five_percent_coverage,
        :words_for_50_percent_coverage => fifty_percent_coverage,
        :words_for_75_percent_coverage => seventy_five_percent_coverage,
        :words_for_90_percent_coverage => ninety_percent_coverage,
        :words_for_95_percent_coverage => ninety_five_percent_coverage,
        :words_for_99_percent_coverage => ninety_nine_percent_coverage,
        :words_for_100_percent_coverage => length(unique_tokens_set),

        # Coverage ratios (proportion of vocabulary needed)
        :coverage_ratio_25 => twenty_five_percent_coverage / length(unique_tokens_set),
        :coverage_ratio_50 => fifty_percent_coverage / length(unique_tokens_set),
        :coverage_ratio_75 => seventy_five_percent_coverage / length(unique_tokens_set),
        :coverage_ratio_90 => ninety_percent_coverage / length(unique_tokens_set),
        :coverage_ratio_95 => ninety_five_percent_coverage / length(unique_tokens_set),
        :coverage_ratio_99 => ninety_nine_percent_coverage / length(unique_tokens_set)
    )

    if include_token_distribution
        freq_type_values = collect(values(token_frequencies))
        stats[:mean_type_frequency] = mean(freq_type_values)
        stats[:median_type_frequency] = median(freq_type_values)
        stats[:max_type_frequency] = maximum(freq_type_values)
        stats[:min_type_frequency] = minimum(freq_type_values)

        # Zipf's law coefficient (rough estimate)
        top_n = min(1000, length(freq_type_values))
        top_freqs = sorted_freqs[1:top_n]
        ranks = 1:top_n
        if top_n > 10
            log_ranks = log.(ranks)
            log_freqs = log.(top_freqs)
            mean_x = mean(log_ranks)
            mean_y = mean(log_freqs)
            zipf_slope = sum((log_ranks .- mean_x) .* (log_freqs .- mean_y)) /
                         sum((log_ranks .- mean_x) .^ 2)
            stats[:zipf_coefficient] = abs(zipf_slope)
        end
    end

    return stats
end

# Alternative: Separate function for detailed token analysis
"""
    token_distribution(corpus::Corpus) -> DataFrame

Analyze the distribution of tokens in the corpus.
"""
function token_distribution(corpus::Corpus)
    token_frequencies = Dict{String,Int}()
    doc_frequencies = Dict{String,Int}()  # How many docs each token appears in

    for doc in corpus.documents
        doc_tokens = tokens(doc)
        doc_unique = unique(doc_tokens)

        # Count total frequencies
        for token in doc_tokens
            token_frequencies[token] = get(token_frequencies, token, 0) + 1
        end

        # Count document frequencies
        for token in doc_unique
            doc_frequencies[token] = get(doc_frequencies, token, 0) + 1
        end
    end

    # Create DataFrame with token statistics
    word_tokens = collect(keys(token_frequencies))
    df = DataFrame(
        Token=word_tokens,
        Frequency=[token_frequencies[t] for t in word_tokens],
        DocFrequency=[doc_frequencies[t] for t in word_tokens],
        DocFrequencyRatio=[doc_frequencies[t] / length(corpus.documents) for t in word_tokens]
    )

    # Calculate TF-IDF scores
    n_docs = length(corpus.documents)
    df.IDF = log.(n_docs ./ df.DocFrequency)
    df.TFIDF = df.Frequency .* df.IDF

    # Sort by frequency
    sort!(df, :Frequency, rev=true)

    return df
end

"""
    vocab_coverage(corpus::Corpus; 
                             percentiles=0.01:0.01:1.0) -> DataFrame

Calculate vocabulary coverage curve showing how many words are needed 
to cover various percentages of the corpus. Uses the corpus vocabulary
for consistent calculations.
"""
function vocab_coverage(corpus::Corpus;
    percentiles=0.01:0.01:1.0)
    # Build token frequencies using corpus vocabulary as reference
    token_frequencies = Dict{String,Int}()

    # Initialize with all vocabulary words at 0 frequency
    for word in keys(corpus.vocabulary)
        token_frequencies[word] = 0
    end

    # Count actual frequencies
    total_tokens = 0
    for doc in corpus.documents
        doc_tokens = tokens(doc)
        total_tokens += length(doc_tokens)
        for token in doc_tokens
            if haskey(token_frequencies, token)
                token_frequencies[token] += 1
            end
            # Note: tokens not in corpus.vocabulary are ignored
            # This ensures consistency with the corpus vocabulary
        end
    end

    # Sort by frequency (descending)
    sorted_pairs = sort(collect(token_frequencies), by=x -> x[2], rev=true)
    sorted_freqs = [p[2] for p in sorted_pairs]
    cumsum_freqs = cumsum(sorted_freqs)

    # Use the corpus vocabulary size
    vocab_size = length(corpus.vocabulary)

    # Calculate coverage for each percentile
    coverage_data = []
    for p in percentiles
        target = total_tokens * p
        n_words = findfirst(x -> x >= target, cumsum_freqs)
        if n_words === nothing
            n_words = vocab_size
        end

        push!(coverage_data, (
            Percentile=p * 100,
            WordsNeeded=n_words,
            ProportionOfVocab=n_words / vocab_size,
            CumulativeTokens=n_words > 0 && n_words <= length(cumsum_freqs) ?
                             cumsum_freqs[n_words] : total_tokens
        ))
    end

    return DataFrame(coverage_data)
end

"""
    coverage_summary(stats::Dict)

Pretty print the vocabulary coverage statistics.
"""
function coverage_summary(stats::Dict)
    println("\n=== Vocabulary Coverage Summary ===")
    println("Total tokens: ", stats[:total_tokens])
    println("Vocabulary size: ", stats[:vocabulary_size])
    println("\nCoverage Statistics:")
    println("─"^50)

    percentages = [25, 50, 75, 90, 95, 99, 100]
    for pct in percentages
        if pct == 100
            words = stats[:words_for_100_percent_coverage]
            ratio = 1.0
        else
            key = Symbol("words_for_$(pct)_percent_coverage")
            words = stats[key]
            ratio_key = Symbol("coverage_ratio_$(pct)")
            ratio = stats[ratio_key]
        end

        println(@sprintf("%3d%% of corpus: %7d words (%6.2f%% of vocabulary)",
            pct, words, ratio * 100))
    end

    println("\nLexical Diversity:")
    println("─"^50)
    println(@sprintf("Type-Token Ratio: %.4f", stats[:type_token_ratio]))
    println(@sprintf("Hapax Legomena: %d (%.2f%% of vocabulary)",
        stats[:hapax_legomena], stats[:hapax_ratio] * 100))
end


# =====================================
# Export Functions - UPDATED
# =====================================

"""
    export_results(analysis::MultiNodeAnalysis, path::AbstractString; format::Symbol=:csv)

Export analysis results to file. Results now include Node column.
"""
function export_results(analysis::MultiNodeAnalysis, path::AbstractString; format::Symbol=:csv)
    if format == :csv
        # Option 1: Export each node's results to separate files
        for (node, results) in analysis.results
            if !isempty(results)
                filename = joinpath(path, "$(node)_results.csv")
                CSV.write(filename, results)
            end
        end

        # Option 2: Also create a combined file with all results
        all_results = DataFrame()
        for (node, results) in analysis.results
            if !isempty(results)
                all_results = vcat(all_results, results, cols=:union)
            end
        end
        if !isempty(all_results)
            CSV.write(joinpath(path, "all_results_combined.csv"), all_results)
        end

        # Export summary
        summary = DataFrame(
            Node=analysis.nodes,
            NumCollocates=[nrow(analysis.results[node]) for node in analysis.nodes]
        )
        CSV.write(joinpath(path, "summary.csv"), summary)

    elseif format == :json
        # Export all results as JSON
        json_data = Dict(
            "parameters" => analysis.parameters,
            "results" => Dict(node => Dict(
                "collocates" => [Dict(pairs(row)) for row in eachrow(results)]
            ) for (node, results) in analysis.results if !isempty(results))
        )

        open(path, "w") do io
            JSON.print(io, json_data, 2)
        end

    elseif format == :excel
        # Requires XLSX package
        XLSX.writetable(path,
            [("$(node)" => analysis.results[node]) for node in analysis.nodes]...,
            overwrite=true)
    end
end

# =====================================
# Extend evalassoc for corpus types
# =====================================

"""
    evalassoc(metric::Type{<:AssociationMetric}, cct::CorpusContingencyTable)

Evaluate a metric on a corpus contingency table by wrapping the corpus-level
lazy aggregated table into a `ContingencyTable` without materializing it.
"""
function evalassoc(::Type{T}, cct::CorpusContingencyTable) where {T<:AssociationMetric}
    # Keep the aggregation lazy: pass the existing LazyProcess straight through.
    temp_ct = ContingencyTable(
        cct.aggregated_table,            # LazyProcess{…,DataFrame}
        cct.node,
        cct.windowsize,
        cct.minfreq,
        LazyInput(StringDocument(""))    # dummy input; fine for corpus-level metrics
    )

    return evalassoc(T, temp_ct)
end


# =====================================
# Example Usage - UPDATED
# =====================================

function demonstrate_corpus_analysis()
    # Example 1: Load corpus from directory
    corpus = load_corpus("path/to/texts/", preprocess=true, min_doc_length=50)

    # Get corpus statistics
    stats = corpus_statistics(corpus)
    println("Corpus contains $(stats[:num_documents]) documents with $(stats[:total_tokens]) tokens")

    # Example 2: Analyze single node word - NOW WITH NODE COLUMN
    results = analyze_corpus(corpus, "important", PMI, windowsize=5, minfreq=10)
    println("Top collocates for 'important':")
    println(first(results, 10))
    # Output now shows: Node | Collocate | Score | Frequency | DocFrequency

    # Example 3: Analyze multiple nodes with multiple metrics
    nodes = ["important", "significant", "critical", "essential"]
    metrics = [PMI, LogDice, LLR]

    multi_analysis = analyze_multiple_nodes(
        corpus, nodes, metrics,
        windowsize=5, minfreq=10, top_n=50
    )

    # Each result DataFrame now includes the Node column
    # Export results - the exported files will include the Node column
    export_results(multi_analysis, "results/", format=:csv)

    # Example 4: Combine results from multiple nodes into single DataFrame
    all_results = DataFrame()
    for (node, df) in multi_analysis.results
        if !isempty(df)
            all_results = vcat(all_results, df, cols=:union)
        end
    end

    # Now you can filter, sort, and analyze across all nodes
    println("All results combined:")
    println(first(sort(all_results, :PMI, rev=true), 20))

    # Example 5: Load from CSV with metadata
    df = DataFrame(
        text=["Document 1 text...", "Document 2 text..."],
        author=["Author A", "Author B"],
        year=[2020, 2021]
    )

    corpus_from_df = load_corpus_df(
        df,
        text_column=:text,
        metadata_columns=[:author, :year]
    )

    return multi_analysis
end

# =====================================
# Batch Processing Functions - UPDATED
# =====================================

"""
    batch_process_corpus(corpus::Corpus,
                        node_file::AbstractString,
                        output_dir::AbstractString;
                        metrics::Vector{DataType}=[PMI, LogDice],
                        windowsize::Int=5,
                        minfreq::Int=5,
                        batch_size::Int=100)

Process a large list of node words in batches. Results include Node column.
"""
function batch_process_corpus(corpus::Corpus,
    node_file::AbstractString,
    output_dir::AbstractString;
    metrics::Vector{DataType}=[PMI, LogDice],
    windowsize::Int=5,
    minfreq::Int=5,
    batch_size::Int=100)

    # Read node words
    nodes = String[]
    open(node_file, "r") do io
        for line in eachline(io)
            node = strip(line)
            !isempty(node) && push!(nodes, node)
        end
    end

    println("Processing $(length(nodes)) node words in batches of $batch_size")

    # Process in batches
    mkpath(output_dir)
    batch_num = 1
    all_batch_results = DataFrame()  # To collect all results

    for batch_start in 1:batch_size:length(nodes)
        batch_end = min(batch_start + batch_size - 1, length(nodes))
        batch_nodes = nodes[batch_start:batch_end]

        println("Processing batch $batch_num (nodes $batch_start-$batch_end)")

        # Analyze batch
        analysis = analyze_multiple_nodes(
            corpus, batch_nodes, metrics,
            windowsize=windowsize, minfreq=minfreq
        )

        # Save batch results
        batch_dir = joinpath(output_dir, "batch_$batch_num")
        mkpath(batch_dir)
        export_results(analysis, batch_dir, format=:csv)

        # Collect all results
        for (node, df) in analysis.results
            if !isempty(df)
                all_batch_results = vcat(all_batch_results, df, cols=:union)
            end
        end

        batch_num += 1
    end

    # Save combined results
    if !isempty(all_batch_results)
        CSV.write(joinpath(output_dir, "all_batches_combined.csv"), all_batch_results)
    end

    println("Batch processing complete. Results saved to $output_dir")
end

# =====================================
# Memory-Efficient Streaming
# =====================================

"""
    stream_corpus_analysis(file_pattern::AbstractString,
                          node::AbstractString,
                          metric::Type{<:AssociationMetric};
                          windowsize::Int=5,
                          chunk_size::Int=1000)

Stream-process large corpora without loading everything into memory.
"""
function stream_corpus_analysis(file_pattern::AbstractString,
    node::AbstractString,
    metric::Type{<:AssociationMetric};
    windowsize::Int=5,
    chunk_size::Int=1000)

    files = glob(file_pattern)
    # aggregated_data = Dict{Symbol,Vector{Int}}()
    aggregated_data = Dict{String,Vector{Int}}()

    @showprogress desc = "Streaming files..." for file_chunk in Iterators.partition(files, chunk_size)
        # Process chunk
        chunk_docs = StringDocument[]
        for file in file_chunk
            content = read(file, String)
            push!(chunk_docs, prepstring(content))
        end

        # Create temporary corpus
        temp_corpus = Corpus(chunk_docs)

        # Analyze chunk
        cct = CorpusContingencyTable(temp_corpus, node, windowsize, 1)
        chunk_table = extract_cached_data(cct.aggregated_table)

        # Aggregate with previous chunks
        # for row in eachrow(chunk_table)
        #     collocate = row.Collocate
        #     if !haskey(aggregated_data, collocate)
        #         aggregated_data[collocate] = zeros(Int, 4)
        #     end
        #     aggregated_data[collocate][1] += row.a
        #     aggregated_data[collocate][2] += row.b
        #     aggregated_data[collocate][3] += row.c
        #     aggregated_data[collocate][4] += row.d
        # end
        for row in eachrow(chunk_table)
            coll = String(row.Collocate)
            v = get!(aggregated_data, coll, zeros(Int, 4))
            @inbounds begin
                v[1] += row.a
                v[2] += row.b
                v[3] += row.c
                v[4] += row.d
            end
        end

        # Clear memory
        chunk_docs = nothing
        GC.gc()
    end

    # Build final DataFrame and evaluate metric
    # ... (similar to aggregate_contingency_tables)
    # --- Build final DataFrame from aggregated_data and evaluate metric ---
    # Convert Dict{String, Vector{Int}} => DataFrame with A,B,C,D and Score
    collocates = collect(keys(aggregated_data))
    A = Vector{Int}(undef, length(collocates))
    B = similar(A)
    C = similar(A)
    D = similar(A)

    for (i, w) in enumerate(collocates)
        v = aggregated_data[w]
        @inbounds begin
            A[i] = v[1]
            B[i] = v[2]
            C[i] = v[3]
            D[i] = v[4]
        end
    end

    df = DataFrame(
        Collocate=collocates,
        A=A, B=B, C=C, D=D
    )

    # Evaluate the metric per aggregated contingency table
    df.Score = Vector{Float64}(undef, nrow(df))
    m = metric()  # instantiate your metric type

    for i in 1:nrow(df)
        ct = ContingencyTable(df.A[i], df.B[i], df.C[i], df.D[i])
        df.Score[i] = evaluate(m, ct)  # or whatever your API uses (e.g., score(m, ct))
    end

    sort!(df, :Score, rev=true)

    return df
end