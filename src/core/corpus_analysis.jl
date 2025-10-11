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
    norm_config::TextNorm  # Single normalization config for entire corpus

    function Corpus(docs::Vector{StringDocument{String}};
        build_dtm::Bool=false,
        metadata::Dict{String,Any}=Dict{String,Any}(),
        norm_config::TextNorm=TextNorm())

        # Build vocabulary
        all_tokens = String[]
        for doc in docs
            append!(all_tokens, tokens(doc))
        end
        vocabulary = build_vocab(unique(all_tokens))

        # Optionally build document-term matrix
        dtm = nothing
        if build_dtm
            dtm = build_document_term_matrix(docs, vocabulary)
        end

        new(docs, metadata, vocabulary, dtm, norm_config)
    end
end

"""
    CorpusContingencyTable

Aggregated contingency table across an entire corpus.
Uses the corpus's normalization configuration.
"""
struct CorpusContingencyTable <: AssociationDataFormat
    tables::Vector{ContingencyTable}
    aggregated_table::LazyProcess{T,DataFrame} where T
    node::AbstractString
    windowsize::Int
    minfreq::Int
    corpus_ref::Corpus
    norm_config::TextNorm

    function CorpusContingencyTable(corpus::Corpus,
        node::AbstractString;
        windowsize::Int,
        minfreq::Int=5)

        # Use corpus's normalization config
        norm_config = corpus.norm_config
        normalized_node = normalize_node(node, norm_config)

        # Create contingency tables for each document using same config
        tables = ContingencyTable[]
        @showprogress desc = "Processing documents..." for doc in corpus.documents
            try
                ct = ContingencyTable(text(doc), normalized_node; windowsize, minfreq=1,
                    # # per-doc minfreq (different from corpus minfreq). I need to keep minfreq=1 here to aggregate properly later for the whole corpus. 
                    norm_config)
                push!(tables, ct)
            catch e
                @warn "Skipping document due to error: $e"
            end
        end

        # Lazy aggregation
        f = () -> aggregate_contingency_tables(tables, minfreq)
        aggregated = LazyProcess(f)

        new(tables, aggregated, normalized_node, windowsize, minfreq, corpus, norm_config)
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
    read_corpus(path::AbstractString;
                text_column::Symbol = :text,
                metadata_columns::Vector{Symbol} = Symbol[],
                preprocess::Bool = true,
                norm_config::TextNorm = TextNorm(),
                min_doc_length::Int = 10,
                max_doc_length::Union{Nothing,Int} = nothing) -> Corpus

Load a text corpus from a **directory of `.txt` files**, a **CSV**, or a **JSON** file,
optionally applying a consistent normalization pipeline (`norm_config`) before wrapping
each text as a `StringDocument`. Returns a `Corpus` that stores the same `norm_config`.

# Inputs
- `path`: Path to a directory, `.csv`, or `.json` file.
- `text_column`: Name of the text column (used for CSV/JSON).
- `metadata_columns`: Columns to attach as per-document metadata (CSV/JSON). Stored under
  keys `doc_1`, `doc_2`, …
- `preprocess`: If `true`, normalizes text via `prep_string(text, norm_config)` before
  creating `StringDocument`; otherwise uses raw text.
- `norm_config`: A `TextNorm` that defines normalization (case/diacritics/unicode, etc.).
- `min_doc_length`: Minimum token count to include a document.
- `max_doc_length`: Optional maximum token count to include a document.

# Behavior
- **Directory**: Reads every `*.txt` file; file path is stored in corpus metadata.
- **CSV**: Extracts `text_column`; attaches `metadata_columns` per row.
- **JSON**: Expects an array of objects; extracts `string(get(entry, string(text_column), ""))`.
- Documents failing length thresholds are skipped.
- Returns `Corpus(documents; metadata=..., norm_config=norm_config)`.
  (By default, no DTM is built here; you can build one later if needed.)

# Returns
A `Corpus` with:
- `documents` = preprocessed `Vector{StringDocument{String}}`
- `metadata`   = `Dict{String,Any}` of per-document info
- `vocabulary` = built from final tokenized documents
- `doc_term_matrix = nothing` (use other APIs to build if desired)
- `norm_config = norm_config` (stored for downstream consistency)

# Examples
```julia
c = read_corpus("data/articles")  # directory with .txt files

c = read_corpus("data/news.csv";
    text_column=:body,
    metadata_columns=[:id, :date, :section],
    norm_config=TextNorm(strip_case=true, strip_accents=true),
    min_doc_length=20)

c = read_corpus("data/posts.json"; text_column=:content, preprocess=false)
```
"""
function read_corpus(path::AbstractString;
    text_column::Symbol=:text,
    metadata_columns::Vector{Symbol}=Symbol[],
    preprocess::Bool=true,
    norm_config::TextNorm=TextNorm(),  # Direct TextNorm config
    min_doc_length::Int=10,
    max_doc_length::Union{Nothing,Int}=nothing)

    documents = StringDocument{String}[]
    metadata = Dict{String,Any}()

    if isdir(path)
        # Load from directory of text files
        files = filter(f -> endswith(f, ".txt"), readdir(path, join=true))
        @showprogress desc = "Loading files..." for file in files
            content = read_text_smart(file)

            typed_doc = preprocess ? StringDocument(text(prep_string(content, norm_config))) :
                        StringDocument(content)

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
        df_names = names(df)
        df_syms = Set(Symbol.(df_names))

        # Helper to fetch a column value by Symbol/String name
        getcol = function (row, colsym::Symbol)
            if colsym in df_syms
                return row[colsym]
            elseif String(colsym) in df_names
                return row[String(colsym)]
            else
                throw(ArgumentError("Column '$(colsym)' not found in CSV. Available: $(df_names)"))
            end
        end

        @showprogress desc = "Processing CSV rows..." for row in eachrow(df)
            text_content = string(getcol(row, text_column))

            typed_doc = preprocess ? StringDocument(text(prep_string(text_content, norm_config))) :
                        StringDocument(text_content)

            doc_tokens = tokens(typed_doc)
            if length(doc_tokens) >= min_doc_length &&
               (max_doc_length === nothing || length(doc_tokens) <= max_doc_length)
                push!(documents, typed_doc)

                # Per-row metadata
                row_metadata = Dict{Symbol,Any}()
                for col in metadata_columns
                    if col in df_syms
                        row_metadata[col] = row[col]                # Symbol access
                    elseif String(col) in df_names
                        row_metadata[col] = row[String(col)]        # String fallback
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
                    typed_doc = preprocess ? StringDocument(text(prep_string(text_content, norm_config))) :
                                StringDocument(text_content)

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

    # Record preprocessing options used for this load
    metadata["_preprocessing_options"] = Dict(
        :norm_config => norm_config,
        :preprocess => preprocess,
        :min_doc_length => min_doc_length,
        :max_doc_length => max_doc_length
    )
    # Create corpus with normalization config
    return Corpus(documents, metadata=metadata, norm_config=norm_config)
end


"""
    read_corpus(c::Corpus;
                preprocess::Bool = true,
                norm_config::Union{Nothing,TextNorm} = nothing,
                min_doc_length::Int = 0,
                max_doc_length::Union{Nothing,Int} = nothing,
                build_dtm::Bool = false) -> Corpus

(Re)process an existing `Corpus` by **applying normalization to its documents** and
return a **new** `Corpus`. By default, uses the corpus's own `c.norm_config`. You can
optionally override it with a new `TextNorm`.

This is useful when:
- You loaded/constructed a corpus without normalization and want to standardize it.
- You want to change normalization (e.g., add accent/case folding) and rebuild vocab/DTM.
- You want to filter very short/long documents after normalization.

# Inputs
- `c`: The source `Corpus`.
- `preprocess`: If `true`, runs `prep_string(text, cfg)` on each document, where
  `cfg = isnothing(norm_config) ? c.norm_config : norm_config`.
- `norm_config`: Override normalization for this pass; defaults to `c.norm_config`.
- `min_doc_length`: Minimum token count to keep a document after reprocessing.
- `max_doc_length`: Optional maximum token count to keep.
- `build_dtm`: If `true`, builds the document–term matrix in the returned corpus.

# Behavior
- Produces a new `Vector{StringDocument}` from `c.documents`, applying normalization
  only if `preprocess=true`.
- Preserves (copies) `c.metadata`.
- Rebuilds `vocabulary` (and `doc_term_matrix` if `build_dtm=true`) from the
  **reprocessed** documents.
- Stores the effective `cfg` as the returned corpus's `norm_config`.

> Idempotency note: Re-running with the same `cfg` typically yields the same result,
> so multiple passes are safe.

# Returns
A new `Corpus` reflecting the (re)applied normalization and filters.

# Examples
```julia
# Re-apply the corpus’s own normalization (no change if already applied)
c2 = read_corpus(c1)

# Override with stricter normalization and build the DTM
cfg = TextNorm(strip_accents=true, strip_case=true, unicode_form=:NFC)
c3 = read_corpus(c1; norm_config=cfg, build_dtm=true)

# Keep only docs with at least 20 tokens after normalization
c4 = read_corpus(c1; min_doc_length=20)
```
"""
# Add alongside the other read_corpus methods
function read_corpus(c::Corpus;
    preprocess::Bool=true,
    norm_config::Union{Nothing,TextNorm}=nothing,  # defaults to c.norm_config
    min_doc_length::Int=0,
    max_doc_length::Union{Nothing,Int}=nothing,
    build_dtm::Bool=false)

    cfg = isnothing(norm_config) ? c.norm_config : norm_config

    # Reprocess each document using the same pipeline used by file/df loaders
    documents = StringDocument{String}[]
    for doc in c.documents
        raw = text(doc)
        typed_doc = preprocess ? StringDocument(text(prep_string(raw, cfg))) : StringDocument(raw)

        # Optional length filters (token-based, consistent with other loaders)
        toks = tokens(typed_doc)
        if length(toks) >= min_doc_length &&
           (max_doc_length === nothing || length(toks) <= max_doc_length)
            push!(documents, typed_doc)
        end
    end

    # Preserve metadata; attach effective preprocessing options
    meta = copy(c.metadata)
    meta["_preprocessing_options"] = Dict(
        :norm_config => cfg,
        :preprocess => preprocess,
        :min_doc_length => min_doc_length,
        :max_doc_length => max_doc_length
    )

    # Preserve metadata; keep/override the config; optionally rebuild DTM
    return Corpus(
        documents;
        metadata=meta,
        norm_config=cfg,
        build_dtm=build_dtm
    )
end

"""
    read_corpus_df(df::DataFrame;
                   text_column::Symbol = :text,
                   metadata_columns::Vector{Symbol} = Symbol[],
                   preprocess::Bool = true,
                   norm_config::TextNorm = TextNorm()) -> Corpus

Load a corpus **directly from a `DataFrame`** using a consistent normalization pipeline
(`norm_config`). Each row becomes a `StringDocument`. Optional `metadata_columns` are
stored per document under keys `doc_1`, `doc_2`, …

# Inputs
- `df`: Source `DataFrame`.
- `text_column`: Column containing the raw text.
- `metadata_columns`: Additional columns to store alongside each document.
- `preprocess`: If `true`, applies `prep_string(text, norm_config)` before wrapping.
- `norm_config`: A `TextNorm` that defines normalization behavior.

# Behavior
- Accesses columns by either `Symbol` or `String` name (graceful fallback).
- Builds a `Corpus` (no DTM by default) and **stores** `norm_config` on the corpus.
- Vocabulary is built from the final tokenized documents.

# Returns
A `Corpus` as above (documents, metadata, vocabulary, `doc_term_matrix=nothing`,
and `norm_config` set to the provided config).

# Example
```julia
using DataFrames
df = DataFrame(id=1:3, text=["One two", "Two three", "Three four"])

c = read_corpus_df(df;
    text_column=:text,
    metadata_columns=[:id],
    norm_config=TextNorm(strip_case=true))
```
"""
function read_corpus_df(df::DataFrame;
    text_column::Symbol=:text,
    metadata_columns::Vector{Symbol}=Symbol[],
    preprocess::Bool=true,
    norm_config::TextNorm=TextNorm())

    documents = StringDocument{String}[]
    corpus_metadata = Dict{String,Any}()

    df_names = names(df)
    df_syms = Set(Symbol.(df_names))

    # Helper to fetch a column value by Symbol/String name
    getcol = function (row, colsym::Symbol)
        if colsym in df_syms
            return row[colsym]
        elseif String(colsym) in df_names
            return row[String(colsym)]
        else
            throw(ArgumentError("Column '$(colsym)' not found in DataFrame. Available: $(df_names)"))
        end
    end

    @showprogress desc = "Processing DataFrame..." for (idx, row) in enumerate(eachrow(df))
        text_content = string(getcol(row, text_column))

        typed_doc = preprocess ? StringDocument(text(prep_string(text_content, norm_config))) :
                    StringDocument(text_content)

        push!(documents, typed_doc)

        # Store document metadata with index
        if !isempty(metadata_columns)
            doc_meta = Dict{Symbol,Any}()
            for col in metadata_columns
                if col in df_syms
                    doc_meta[col] = row[col]
                elseif String(col) in df_names
                    doc_meta[col] = row[String(col)]
                end
            end
            corpus_metadata["doc_$idx"] = doc_meta
        end
    end

    # Record preprocessing options used for this load
    corpus_metadata["_preprocessing_options"] = Dict(
        :norm_config => norm_config,
        :preprocess => preprocess
        # (min/max lengths aren’t used in read_corpus_df yet; add if you add filters)
    )
    return Corpus(documents, metadata=corpus_metadata, norm_config=norm_config)
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
    all_collocates = Set{String}()
    for table in tables
        ct = cached_data(table.con_tbl)
        !isempty(ct) && union!(all_collocates, ct.Collocate)
    end

    # Initialize aggregated data
    agg_data = Dict{String,Vector{Int}}()
    for collocate in all_collocates
        agg_data[collocate] = zeros(Int, 4)  # [a, b, c, d]
    end

    # Aggregate across documents
    for table in tables
        ct = cached_data(table.con_tbl)
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
    analyze_node(corpus::Corpus, node::AbstractString, metric::Type{<:AssociationMetric};
                  windowsize::Int, minfreq::Int=5) -> DataFrame

Analyze a single node word across the entire corpus using corpus's normalization.
Returns DataFrame with Node, Collocate, Score, Frequency, and DocFrequency columns.
"""
function analyze_node(corpus::Corpus,
    node::AbstractString,
    metric::Type{<:AssociationMetric};
    windowsize::Int,
    minfreq::Int=5)

    # Create corpus contingency table (will use corpus's norm_config)
    cct = CorpusContingencyTable(corpus, node; windowsize, minfreq)

    # Evaluate metric
    scores_df = assoc_score(metric, cct)

    # eltype ensures Collocate is String
    @assert eltype(scores_df.Collocate) === String "Collocate column must be String"

    if nrow(scores_df) == 0
        result = DataFrame(
            Node=String[],
            Collocate=String[],
            Score=Float64[],
            Frequency=Int[],
            DocFrequency=Int[]
        )
        metadata!(result, "status", "empty", style=:note)
        present = assoc_node_present(cct)
        msg = present === false ?
              "Node '$(cct.node)' not found in the corpus." :
              "Node found, but no collocates met the thresholds for node='$(cct.node)' (windowsize=$(windowsize), minfreq=$(minfreq))."
        metadata!(result, "message", msg, style=:note)
        metadata!(result, "metric", string(metric), style=:note)
        metadata!(result, "node", cct.node, style=:note)
        metadata!(result, "windowsize", windowsize, style=:note)
        metadata!(result, "minfreq", minfreq, style=:note)
        metadata!(result, "analysis_type", "corpus_analysis", style=:note)
        return result
    end

    # Calculate document frequency
    doc_freq = [count(t -> begin
            ct = cached_data(t.con_tbl)
            !isempty(ct) && col in ct.Collocate
        end, cct.tables) for col in scores_df.Collocate]

    # Build result DataFrame
    result = DataFrame(
        Node=scores_df.Node,
        Collocate=scores_df.Collocate,
        Score=scores_df[!, Symbol(string(metric))],
        Frequency=scores_df.Frequency,
        DocFrequency=doc_freq
    )

    sort!(result, :Score, rev=true)

    # Add metadata
    metadata!(result, "status", "ok", style=:note)
    if haskey(metadata(scores_df), "message")
        metadata!(result, "message", metadata(scores_df)["message"], style=:note)
    end
    metadata!(result, "metric", string(metric), style=:note)
    metadata!(result, "node", cct.node, style=:note)  # Use normalized node
    metadata!(result, "windowsize", windowsize, style=:note)
    metadata!(result, "minfreq", minfreq, style=:note)
    metadata!(result, "analysis_type", "corpus_analysis", style=:note)

    return result
end

"""
    analyze_nodes(corpus::Corpus, nodes::Vector{String}, metrics::Vector{DataType};
                 windowsize::Int, minfreq::Int=5, top_n::Int=100,
                 parallel::Bool=false) -> MultiNodeAnalysis

Analyze multiple nodes with consistent normalization.
Each result DataFrame now includes the Node column and metadata.
"""
function analyze_nodes(corpus::Corpus,
    nodes::Vector{String},
    metrics::Vector{DataType};
    windowsize::Int,
    minfreq::Int=5,
    top_n::Int=100,
    parallel::Bool=false)

    results = Dict{String,DataFrame}()

    if parallel && nworkers() > 1
        # Parallel processing (implementation omitted for brevity)
    else
        @showprogress desc = "Analyzing nodes..." for node in nodes
            # Create corpus contingency table (node normalized inside)
            cct = CorpusContingencyTable(corpus, node; windowsize, minfreq)

            agg_table = cached_data(cct.aggregated_table)

            if !isempty(agg_table)
                metric_results = assoc_score(metrics, cct)

                if !isempty(metric_results)
                    first_metric = Symbol(string(metrics[1]))
                    sort!(metric_results, first_metric, rev=true)
                    result = first(metric_results, min(top_n, nrow(metric_results)))

                    # Add metadata
                    metric_names = join(string.(metrics), ", ")
                    metadata!(result, "metrics", metric_names, style=:note)
                    metadata!(result, "node", cct.node, style=:note)
                    metadata!(result, "windowsize", windowsize, style=:note)
                    metadata!(result, "minfreq", minfreq, style=:note)
                    metadata!(result, "top_n", top_n, style=:note)

                    results[cct.node] = result  # Use normalized node as key
                else
                    results[cct.node] = DataFrame()
                end
            else
                results[cct.node] = DataFrame()
            end
        end
    end

    parameters = Dict(
        :windowsize => windowsize,
        :minfreq => minfreq,
        :metrics => metrics,
        :top_n => top_n
    )

    normalized_nodes = collect(keys(results))

    return MultiNodeAnalysis(normalized_nodes, results, corpus, parameters)
end

"""
    corpus_stats(corpus::Corpus; 
                     include_token_distribution::Bool=true) -> Dict

Get comprehensive statistics about the corpus.
"""
function corpus_stats(corpus::Corpus;
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
    total_tokens = sum(values(token_frequencies))
    n_docs = length(corpus.documents)
    relative_freqs = total_tokens == 0 ? zeros(Float64, length(word_tokens)) :
                     [token_frequencies[t] / total_tokens for t in word_tokens]
    doc_freq_ratio = n_docs == 0 ? zeros(Float64, length(word_tokens)) :
                     [get(doc_frequencies, t, 0) / n_docs for t in word_tokens]
    df = DataFrame(
        Token=word_tokens,
        Frequency=[token_frequencies[t] for t in word_tokens],
        DocFrequency=[get(doc_frequencies, t, 0) for t in word_tokens],
        DocFrequencyRatio=doc_freq_ratio,
        RelativeFrequency=relative_freqs
    )

    # Calculate TF-IDF scores
    df.IDF = log.(n_docs ./ df.DocFrequency)
    df.TFIDF = df.Frequency .* df.IDF

    # Sort by frequency
    sort!(df, :Frequency, rev=true)

    return df
end

# Alternative: Separate function for detailed token analysis
"""
    token_distribution(text::AbstractString) -> DataFrame

Analyze the distribution of tokens in a string and return a DataFrame with info about absolute and relative token frequencies.
"""
function token_distribution(text::AbstractString)
    token_frequencies = Dict{String,Int}()

    doc = StringDocument(text)
    text_tokens = tokens(doc)

    # Count total frequencies
    for token in text_tokens
        token_frequencies[token] = get(token_frequencies, token, 0) + 1
    end

    # Create DataFrame with token statistics
    word_tokens = collect(keys(token_frequencies))
    total_tokens = sum(values(token_frequencies))
    relative_freqs = total_tokens == 0 ? zeros(Float64, length(word_tokens)) :
                     [token_frequencies[t] / total_tokens for t in word_tokens]
    df = DataFrame(
        Token=word_tokens,
        Frequency=[token_frequencies[t] for t in word_tokens],
        RelativeFrequency=relative_freqs
    )

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
    write_results(analysis::MultiNodeAnalysis, path::AbstractString; format::Symbol=:csv)

Export analysis results to file. Results now include Node column.
"""
function write_results(analysis::MultiNodeAnalysis, path::AbstractString; format::Symbol=:csv)
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
# Extend assoc_score for corpus types
# =====================================

# """
#     assoc_score(metric::Type{<:AssociationMetric}, cct::CorpusContingencyTable)

# Evaluate a metric on a corpus contingency table by wrapping the corpus-level
# lazy aggregated table into a `ContingencyTable` without materializing it.
# """
# function assoc_score(::Type{T}, cct::CorpusContingencyTable) where {T<:AssociationMetric}
#     # Keep the aggregation lazy: pass the existing LazyProcess straight through.
#     temp_ct = ContingencyTable(
#         cct.aggregated_table,            # LazyProcess{…,DataFrame}
#         cct.node,
#         cct.windowsize,
#         cct.minfreq,
#         LazyInput(StringDocument("")),
#         cct.norm_config   # Use the corpus's normalization config
#     )

#     return assoc_score(T, temp_ct)
# end


# =====================================
# Example Usage - UPDATED
# =====================================

function demonstrate_corpus_analysis()
    # Example 1: Load corpus from directory
    corpus = read_corpus("path/to/texts/", preprocess=true, min_doc_length=50)

    # Get corpus statistics
    stats = corpus_stats(corpus)
    println("Corpus contains $(stats[:num_documents]) documents with $(stats[:total_tokens]) tokens")

    # Example 2: Analyze single node word - NOW WITH NODE COLUMN
    results = analyze_node(corpus, "important", PMI; windowsize=5, minfreq=10)
    println("Top collocates for 'important':")
    println(first(results, 10))
    # Output now shows: Node | Collocate | Score | Frequency | DocFrequency

    # Example 3: Analyze multiple nodes with multiple metrics
    nodes = ["important", "significant", "critical", "essential"]
    metrics = [PMI, LogDice, LLR]

    multi_analysis = analyze_nodes(
        corpus, nodes, metrics;
        windowsize=5, minfreq=10, top_n=50
    )

    # Each result DataFrame now includes the Node column
    # Export results - the exported files will include the Node column
    write_results(multi_analysis, "results/", format=:csv)

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

    corpus_from_df = read_corpus_df(
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
                        windowsize::Int,
                        minfreq::Int=5,
                        batch_size::Int=100)

Process a large list of node words in batches. Results include Node column.
"""
function batch_process_corpus(corpus::Corpus,
    node_file::AbstractString,
    output_dir::AbstractString;
    metrics::Vector{DataType},
    windowsize::Int,
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
        analysis = analyze_nodes(
            corpus, batch_nodes, metrics;
            windowsize, minfreq
        )

        # Save batch results
        batch_dir = joinpath(output_dir, "batch_$batch_num")
        mkpath(batch_dir)
        write_results(analysis, batch_dir, format=:csv)

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
                          windowsize::Int,
                          chunk_size::Int=1000)

Stream-process large corpora without loading everything into memory.
"""
function stream_corpus_analysis(file_pattern::AbstractString,
    node::AbstractString,
    metric::Type{<:AssociationMetric};
    windowsize::Int,
    chunk_size::Int=1000)

    files = glob(file_pattern)
    # aggregated_data = Dict{Symbol,Vector{Int}}()
    aggregated_data = Dict{String,Vector{Int}}()

    @showprogress desc = "Streaming files..." for file_chunk in Iterators.partition(files, chunk_size)
        # Process chunk
        chunk_docs = StringDocument[]
        for file in file_chunk
            content = read(file, String)
            push!(chunk_docs, prep_string(content, TextNorm()))
        end

        # Create temporary corpus
        temp_corpus = Corpus(chunk_docs)

        # Analyze chunk
        cct = CorpusContingencyTable(temp_corpus, node; windowsize, minfreq=1)
        chunk_table = cached_data(cct.aggregated_table)

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
            coll = row.Collocate
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
