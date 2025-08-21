# =====================================
# File: src/core/corpus_analysis.jl
# Corpus-level analysis functionality
# =====================================

using TextAnalysis
using DataFrames
using ProgressMeter
using Distributed

"""
    Corpus <: AssociationDataFormat

Represents a collection of documents for corpus-level analysis.
"""
struct Corpus <: AssociationDataFormat
    documents::Vector{StringDocument}
    metadata::Dict{String,Any}
    vocabulary::OrderedDict{String,Int}
    doc_term_matrix::Union{Nothing,SparseMatrixCSC}

    function Corpus(docs::Vector{StringDocument};
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
        minfreq::Int64=5)

        # Create contingency tables for each document
        tables = ContingencyTable[]
        @showprogress desc = "Processing documents..." for doc in corpus.documents
            try
                ct = ContingencyTable(text(doc), node, windowsize, 1)  # Use minfreq=1 per doc
                push!(tables, ct)
            catch e
                @warn "Skipping document due to error: $e"
            end
        end

        # Define lazy aggregation
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
- `min_doc_length`: Minimum document length in tokens (default: 10)
- `max_doc_length`: Maximum document length in tokens (default: nothing)
"""
function load_corpus(path::AbstractString;
    text_column::Symbol=:text,
    metadata_columns::Vector{Symbol}=Symbol[],
    preprocess::Bool=true,
    min_doc_length::Int=10,
    max_doc_length::Union{Nothing,Int}=nothing)

    documents = StringDocument[]
    metadata = Dict{String,Any}()

    if isdir(path)
        # Load from directory of text files
        files = filter(f -> endswith(f, ".txt"), readdir(path, join=true))
        @showprogress desc = "Loading files..." for file in files
            content = read(file, String)
            doc = preprocess ? prepstring(content) : StringDocument(content)

            # Check document length
            doc_tokens = tokens(doc)
            if length(doc_tokens) >= min_doc_length &&
               (max_doc_length === nothing || length(doc_tokens) <= max_doc_length)
                push!(documents, doc)
                metadata[basename(file)] = Dict(:source => file)
            end
        end

    elseif endswith(lowercase(path), ".csv")
        # Load from CSV
        df = DataFrame(CSV.File(path))

        @showprogress desc = "Processing CSV rows..." for row in eachrow(df)
            text_content = string(row[text_column])
            doc = preprocess ? prepstring(text_content) : StringDocument(text_content)

            # Check document length
            doc_tokens = tokens(doc)
            if length(doc_tokens) >= min_doc_length &&
               (max_doc_length === nothing || length(doc_tokens) <= max_doc_length)
                push!(documents, doc)

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
                    doc = preprocess ? prepstring(text_content) : StringDocument(text_content)

                    # Check document length
                    doc_tokens = tokens(doc)
                    if length(doc_tokens) >= min_doc_length &&
                       (max_doc_length === nothing || length(doc_tokens) <= max_doc_length)
                        push!(documents, doc)
                        metadata["doc_$(length(documents))"] = entry
                    end
                end
            end
        end
    else
        throw(ArgumentError("Unsupported file format. Use directory, CSV, or JSON."))
    end

    println("Loaded $(length(documents)) documents")
    return Corpus(documents, metadata=metadata)
end

"""
    load_corpus_from_dataframe(df::DataFrame; kwargs...) -> Corpus

Load corpus directly from a DataFrame.
"""
function load_corpus_from_dataframe(df::DataFrame;
    text_column::Symbol=:text,
    metadata_columns::Vector{Symbol}=Symbol[],
    preprocess::Bool=true)

    documents = StringDocument[]
    metadata = Dict{String,Any}()

    @showprogress desc = "Processing DataFrame..." for (idx, row) in enumerate(eachrow(df))
        text_content = string(row[text_column])
        doc = preprocess ? prepstring(text_content) : StringDocument(text_content)
        push!(documents, doc)

        # Store metadata
        row_metadata = Dict{Symbol,Any}()
        for col in metadata_columns
            if col in names(df)
                row_metadata[col] = row[col]
            end
        end
        metadata["doc_$idx"] = row_metadata
    end

    return Corpus(documents, metadata=metadata)
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
# Corpus Analysis Functions
# =====================================

"""
    analyze_corpus(corpus::Corpus, node::AbstractString, metric::Type{<:AssociationMetric};
                  windowsize::Int=5, minfreq::Int=5) -> DataFrame

Analyze a single node word across the entire corpus.
"""
function analyze_corpus(corpus::Corpus,
    node::AbstractString,
    metric::Type{<:AssociationMetric};
    windowsize::Int=5,
    minfreq::Int=5)

    # Create corpus contingency table
    cct = CorpusContingencyTable(corpus, node, windowsize, minfreq)

    # Evaluate metric on aggregated data
    scores = evalassoc(metric, cct)

    # Get aggregated table for collocates
    agg_table = extract_cached_data(cct.aggregated_table)

    # Combine results
    result = DataFrame(
        Collocate=agg_table.Collocate,
        Score=scores,
        Frequency=agg_table.a,
        DocFrequency=[count(t -> begin
                ct = extract_cached_data(t.con_tbl)
                !isempty(ct) && col in ct.Collocate
            end, cct.tables) for col in agg_table.Collocate]
    )

    # Sort by score
    sort!(result, :Score, rev=true)

    return result
end

"""
    analyze_multiple_nodes(corpus::Corpus, 
                          nodes::Vector{String}, 
                          metrics::Vector{DataType};
                          windowsize::Int=5,
                          minfreq::Int=5,
                          parallel::Bool=false) -> MultiNodeAnalysis

Analyze multiple node words with multiple metrics across a corpus.
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
        # Parallel processing
        node_results = @distributed (vcat) for node in nodes
            println("Processing node: $node")

            # Create corpus contingency table
            cct = CorpusContingencyTable(corpus, node, windowsize, minfreq)

            # Evaluate all metrics
            metric_results = DataFrame()
            for metric in metrics
                scores = evalassoc(metric, cct)
                metric_results[!, string(metric)] = scores
            end

            # Get aggregated table
            agg_table = extract_cached_data(cct.aggregated_table)

            if !isempty(agg_table)
                # Combine with collocate info
                result = DataFrame(
                    Collocate=agg_table.Collocate,
                    Frequency=agg_table.a
                )

                # Add metric scores
                for col in names(metric_results)
                    result[!, col] = metric_results[!, col]
                end

                # Keep top N by first metric
                first_metric = string(metrics[1])
                sort!(result, Symbol(first_metric), rev=true)
                result = first(result, min(top_n, nrow(result)))

                [(node, result)]
            else
                [(node, DataFrame())]
            end
        end

        # Collect results
        for (node, result) in node_results
            results[node] = result
        end
    else
        # Sequential processing
        @showprogress desc = "Analyzing nodes..." for node in nodes
            # Create corpus contingency table
            cct = CorpusContingencyTable(corpus, node, windowsize, minfreq)

            # Get aggregated table
            agg_table = extract_cached_data(cct.aggregated_table)

            if !isempty(agg_table)
                # Evaluate all metrics
                metric_results = DataFrame()
                for metric in metrics
                    scores = evalassoc(metric, cct)
                    metric_results[!, string(metric)] = scores
                end

                # Combine results
                result = DataFrame(
                    Collocate=agg_table.Collocate,
                    Frequency=agg_table.a
                )

                # Add metric scores
                for col in names(metric_results)
                    result[!, col] = metric_results[!, col]
                end

                # Keep top N by first metric
                first_metric = string(metrics[1])
                sort!(result, Symbol(first_metric), rev=true)
                results[node] = first(result, min(top_n, nrow(result)))
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
    corpus_statistics(corpus::Corpus) -> Dict

Get basic statistics about the corpus.
"""
function corpus_statistics(corpus::Corpus)
    total_tokens = 0
    unique_tokens = Set{String}()
    doc_lengths = Int[]

    for doc in corpus.documents
        doc_tokens = tokens(doc)
        total_tokens += length(doc_tokens)
        union!(unique_tokens, doc_tokens)
        push!(doc_lengths, length(doc_tokens))
    end

    return Dict(
        :num_documents => length(corpus.documents),
        :total_tokens => total_tokens,
        :unique_tokens => length(unique_tokens),
        :avg_doc_length => mean(doc_lengths),
        :median_doc_length => median(doc_lengths),
        :min_doc_length => minimum(doc_lengths),
        :max_doc_length => maximum(doc_lengths),
        :vocabulary_size => length(corpus.vocabulary)
    )
end

# =====================================
# Export Functions
# =====================================

"""
    export_results(analysis::MultiNodeAnalysis, path::AbstractString; format::Symbol=:csv)

Export analysis results to file.
"""
function export_results(analysis::MultiNodeAnalysis, path::AbstractString; format::Symbol=:csv)
    if format == :csv
        # Export each node's results to a separate CSV
        for (node, results) in analysis.results
            if !isempty(results)
                filename = joinpath(path, "$(node)_results.csv")
                CSV.write(filename, results)
            end
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

Evaluate a metric on a corpus contingency table.
"""
function evalassoc(metric::Type{<:AssociationMetric}, cct::CorpusContingencyTable)
    # Use aggregated table
    agg_table = extract_cached_data(cct.aggregated_table)

    # Create a temporary ContingencyTable-like structure
    temp_ct = ContingencyTable(
        LazyProcess(() -> agg_table),
        cct.node,
        cct.windowsize,
        cct.minfreq,
        LazyInput(StringDocument(""))  # Dummy for compatibility
    )

    # Evaluate metric
    return evalassoc(metric, temp_ct)
end

# =====================================
# Example Usage
# =====================================

function demonstrate_corpus_analysis()
    # Example 1: Load corpus from directory
    corpus = load_corpus("path/to/texts/", preprocess=true, min_doc_length=50)

    # Get corpus statistics
    stats = corpus_statistics(corpus)
    println("Corpus contains $(stats[:num_documents]) documents with $(stats[:total_tokens]) tokens")

    # Example 2: Analyze single node word
    results = analyze_corpus(corpus, "important", PMI, windowsize=5, minfreq=10)
    println("Top collocates for 'important':")
    println(first(results, 10))

    # Example 3: Analyze multiple nodes with multiple metrics
    nodes = ["important", "significant", "critical", "essential"]
    metrics = [PMI, LogDice, LLR]

    multi_analysis = analyze_multiple_nodes(
        corpus, nodes, metrics,
        windowsize=5, minfreq=10, top_n=50
    )

    # Export results
    export_results(multi_analysis, "results/", format=:csv)

    # Example 4: Load from CSV with metadata
    df = DataFrame(
        text=["Document 1 text...", "Document 2 text..."],
        author=["Author A", "Author B"],
        year=[2020, 2021]
    )

    corpus_from_df = load_corpus_from_dataframe(
        df,
        text_column=:text,
        metadata_columns=[:author, :year]
    )

    return multi_analysis
end

# =====================================
# Batch Processing Functions
# =====================================

"""
    batch_process_corpus(corpus::Corpus, 
                        node_file::AbstractString,
                        output_dir::AbstractString;
                        metrics::Vector{DataType}=[PMI, LogDice],
                        windowsize::Int=5,
                        minfreq::Int=5,
                        batch_size::Int=100)

Process a large list of node words in batches.
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

        batch_num += 1
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
    aggregated_data = Dict{Symbol,Vector{Int}}()

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
        for row in eachrow(chunk_table)
            collocate = row.Collocate
            if !haskey(aggregated_data, collocate)
                aggregated_data[collocate] = zeros(Int, 4)
            end
            aggregated_data[collocate][1] += row.a
            aggregated_data[collocate][2] += row.b
            aggregated_data[collocate][3] += row.c
            aggregated_data[collocate][4] += row.d
        end

        # Clear memory
        chunk_docs = nothing
        GC.gc()
    end

    # Build final DataFrame and evaluate metric
    # ... (similar to aggregate_contingency_tables)

    return aggregated_data
end