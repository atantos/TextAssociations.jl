# =====================================
# File: src/core/advanced_corpus.jl
# Advanced corpus analysis features
# =====================================

using DataFrames
using StatsBase: countmap
using Statistics: mean, median, std, var, cor
using ProgressMeter: @showprogress
using SparseArrays: sparse
using Dates
using Graphs

"""
    TemporalCorpusAnalysis

Analysis of word associations over time periods.
"""
struct TemporalCorpusAnalysis
    time_periods::Vector{String}
    results_by_period::Dict{String,MultiNodeAnalysis}
    trend_analysis::DataFrame
    corpus_ref::Corpus
end

# Helper: robust date parsing for common formats; returns `nothing` if it fails
parse_time_value(x) = x isa Date ? x :
                      x isa DateTime ? Date(x) :
                      (
    try
        Date(String(x))
    catch
        try
            Date(String(x), dateformat"yyyy-mm-dd")
        catch
            try
                Date(String(x), dateformat"dd/mm/yyyy")
            catch
                nothing
            end
        end
    end
)

"""
    SubcorpusComparison

Comparison of word associations between subcorpora.
"""
struct SubcorpusComparison
    subcorpora::Dict{String,Corpus}
    node::String
    results::Dict{String,DataFrame}
    summary::DataFrame
    effect_sizes::DataFrame
    parameters::DataFrame
end

# =====================================
# Temporal Analysis
# =====================================

"""
    analyze_temporal(corpus::Corpus,
                            nodes::Vector{String},
                            time_field::Symbol,
                            metric::Type{<:AssociationMetric};
                            time_bins::Int=5,
                            windowsize::Int=5,
                            minfreq::Int=5) -> TemporalCorpusAnalysis

Analyze how word associations change over time.
"""
function analyze_temporal(corpus::Corpus,
    nodes::Vector{String},
    time_field::Symbol,
    metric::Type{<:AssociationMetric};
    time_bins::Int=5,
    windowsize::Int=5,
    minfreq::Int=5)

    # 1) Collect time values aligned to documents[i] via "doc_i" keys
    raw_times = Any[]
    kept_doc_indices = Int[]

    for i in 1:length(corpus.documents)
        meta = get(corpus.metadata, "doc_$i", nothing)
        if meta !== nothing && haskey(meta, time_field)
            push!(raw_times, meta[time_field])
            push!(kept_doc_indices, i)
        end
    end

    isempty(raw_times) && throw(ArgumentError("No documents have the specified time field: $time_field"))

    # 2) Normalize times to numeric for binning (supports Int/Real/Date/DateTime)
    #    Default: keep numbers; Dates => use year.
    tvals = Vector{Float64}(undef, length(raw_times))
    for (k, v) in enumerate(raw_times)
        if v isa Dates.Date
            tvals[k] = float(Dates.year(v))
        elseif v isa Dates.DateTime
            tvals[k] = float(Dates.year(v))
        elseif v isa Integer
            tvals[k] = float(v)
        elseif v isa Real
            tvals[k] = float(v)
        else
            throw(ArgumentError("Unsupported time value type $(typeof(v)) for $time_field"))
        end
    end

    # Edge case: all docs same timestamp -> put everything in one bin
    tmin, tmax = extrema(tvals)
    if tmin == tmax
        time_bins = 1
    end

    # 3) Build bin edges and assign each kept doc to a bin
    edges = collect(range(tmin, tmax; length=time_bins + 1))
    # For a value t, searchsortedlast(edges, t) returns the index in 1..length(edges).
    # We clamp to 1..time_bins so each t goes to a valid interval [edges[i], edges[i+1]].
    doc_bins = [clamp(searchsortedlast(edges, t), 1, time_bins) for t in tvals]

    # 4) Group kept documents into periods (bins), preserving original doc indices
    period_docs = Dict(i => Int[] for i in 1:time_bins)
    @inbounds for (k, bin) in enumerate(doc_bins)
        push!(period_docs[bin], kept_doc_indices[k])
    end

    # 5) Build human-readable period labels (use rounded numeric edges)
    #    If you want integer years, these will already be whole numbers.
    period_label = i -> string(round(edges[i]; digits=0), "-", round(edges[i+1]; digits=0))
    time_periods = [period_label(i) for i in 1:time_bins]

    # 6) Analyze each non-empty period
    results_by_period = Dict{String,MultiNodeAnalysis}()
    for i in 1:time_bins
        idxs = period_docs[i]
        if isempty(idxs)
            continue
        end

        # Subcorpus reuses normalization config
        period_corpus = Corpus(corpus.documents[idxs], norm_config=corpus.norm_config)

        period_results = analyze_nodes(
            period_corpus,
            nodes,
            [metric];
            windowsize=windowsize,
            minfreq=minfreq,
            top_n=100,         # keep your package default if different
            parallel=false
        )

        results_by_period[time_periods[i]] = period_results
    end

    # 7) Optional: compute trend_analysis (keep empty if you haven’t wired this yet)
    trend_analysis = DataFrame()

    return TemporalCorpusAnalysis(
        time_periods,
        results_by_period,
        trend_analysis,
        corpus,   # 4th positional arg must be the Corpus
    )
end



"""
    compute_association_trends(results_by_period, nodes, metric) -> DataFrame

Compute trend statistics for associations over time.
"""
function compute_association_trends(results_by_period::Dict{String,MultiNodeAnalysis},
    nodes::Vector{String},
    metric::Type{<:AssociationMetric})

    trend_data = []

    for node in nodes
        # Collect top collocates across all periods
        all_collocates = Set{String}()
        for (period, analysis) in results_by_period
            if haskey(analysis.results, node) && !isempty(analysis.results[node])
                union!(all_collocates, analysis.results[node].Collocate)
            end
        end

        # Track each collocate over time
        for collocate in all_collocates
            scores_over_time = Float64[]
            periods = String[]

            for (period, analysis) in sort(collect(results_by_period))
                if haskey(analysis.results, node)
                    df = analysis.results[node]
                    idx = findfirst(==(collocate), df.Collocate)

                    if idx !== nothing
                        push!(scores_over_time, df[idx, Symbol(string(metric))])
                        push!(periods, period)
                    end
                end
            end

            if length(scores_over_time) > 1
                # Calculate trend statistics
                xs = collect(1:length(scores_over_time))
                correlation = cor(xs, scores_over_time)

                # Simple linear regression without GLM
                # Calculate slope using least squares formula
                mean_x = mean(xs)
                mean_y = mean(scores_over_time)
                numerator = sum((xs .- mean_x) .* (scores_over_time .- mean_y))
                denominator = sum((xs .- mean_x) .^ 2)
                slope = denominator > 0 ? numerator / denominator : 0.0

                push!(trend_data, (
                    Node=node,
                    Collocate=collocate,
                    Correlation=correlation,
                    Slope=slope,
                    MeanScore=mean(scores_over_time),
                    StdScore=std(scores_over_time),
                    NumPeriods=length(scores_over_time)
                ))
            end
        end
    end

    return DataFrame(trend_data)
end

# =====================================
# Subcorpus Comparison
# =====================================

"""
    compare_subcorpora(corpus::Corpus,
                      split_field::Symbol,
                      node::String,
                      metric::Type{<:AssociationMetric};
                      windowsize::Int=5,
                      minfreq::Int=5) -> SubcorpusComparison

Compare word associations across different subcorpora.

# Arguments
- `corpus`: The corpus to analyze
- `split_field`: Metadata field to split on (e.g., :field, :year, :author)
- `node`: The target word to analyze
- `metric`: Association metric to use (e.g., PMI, LogDice, LLR)
- `windowsize`: Context window size (default: 5)
- `minfreq`: Minimum co-occurrence frequency (default: 5)

# Returns
`SubcorpusComparison` containing:
- `subcorpora`: Dictionary of subcorpora by group
- `node`: The normalized node word
- `results`: Association scores for each subcorpus
- `statistical_tests`: Statistical tests comparing groups
- `effect_sizes`: Effect sizes for differences between groups

# Example
```julia
# Load corpus with metadata
corpus = read_corpus_df(
    df;
    text_column = :text,
    metadata_columns = [:category, :year]
)

# Compare associations across categories
comparison = compare_subcorpora(
    corpus,
    :category,
    "innovation",
    PMI;
    windowsize = 5,
    minfreq = 3
)

# Examine results
for (group, result) in comparison.results
    println("Group: \$group")
    println(first(result, 10))
end
```
"""
function compare_subcorpora(corpus::Corpus,
    split_field::Symbol,
    node::AbstractString,
    metric::Type{<:AssociationMetric};
    windowsize::Int=5,
    minfreq::Int=5)

    # 1) Gather doc indices per group
    groups = Dict{String,Vector{Int}}()
    for i in 1:length(corpus.documents)
        meta = get(corpus.metadata, "doc_$i", nothing)
        if meta !== nothing && haskey(meta, split_field)
            g = string(meta[split_field])
            push!(get!(groups, g, Int[]), i)
        end
    end
    group_names = sort(collect(keys(groups)))

    # 2) Build subcorpora + run analysis
    subcorpora = Dict{String,Corpus}()
    results = Dict{String,DataFrame}()
    metric_col = Symbol(string(metric))

    for g in group_names
        idxs = groups[g]
        isempty(idxs) && continue

        # fast slice -> Vector{StringDocument{String}}
        docs = corpus.documents[idxs]

        # carry per-doc metadata, reindexed to doc_1..doc_n
        sub_meta = Dict{String,Any}()
        for (j, i) in pairs(idxs)
            key = "doc_$i"
            if haskey(corpus.metadata, key)
                sub_meta["doc_$j"] = corpus.metadata[key]
            end
        end

        subcorpus = Corpus(docs; metadata=sub_meta, norm_config=corpus.norm_config)
        subcorpora[g] = subcorpus

        df = analyze_node(subcorpus, node, metric; windowsize=windowsize, minfreq=minfreq)

        # Adding group column
        if !isempty(df)
            insertcols!(df, 1, :Group => g)
        end

        results[g] = df
    end

    # 3) Summary statistics 
    summary = DataFrame(
        Subcorpus=group_names,
        NumCollocates=[haskey(results, g) && !isempty(results[g]) ? nrow(results[g]) : 0 for g in group_names],
        AvgScore=[haskey(results, g) && !isempty(results[g]) ? mean(results[g][!, metric_col]) : NaN for g in group_names]
    )

    # 4) Parameters info
    parameters = DataFrame(
        Parameter=["split_field", "node", "metric", "windowsize", "minfreq"],
        Value=[string(split_field), String(node), string(metric), string(windowsize), string(minfreq)]
    )

    # 5) Effect sizes
    effect_sizes = calculate_effect_sizes(results, metric)

    return SubcorpusComparison(
        subcorpora,               # Dict{String,Corpus}
        String(node),             # node
        results,                  # Dict{String,DataFrame}
        summary,                  # Summary stats
        effect_sizes,             # Effect sizes
        parameters                # Parameters info
    )
end




"""
    perform_statistical_tests(results, metric) -> DataFrame

Perform statistical tests between subcorpora.
"""
function perform_statistical_tests(results::Dict{String,DataFrame},
    metric::Type{<:AssociationMetric})

    test_results = []
    groups = collect(keys(results))

    if length(groups) < 2
        return DataFrame()
    end

    # Get common collocates
    common_collocates = Set{String}()
    for (_, df) in results
        isempty(df) && continue
        if isempty(common_collocates)
            # first non-empty group
            common_collocates = Set(df.Collocate)
        else
            intersect!(common_collocates, Set(df.Collocate))
        end
    end

    score_col = :Score

    for collocate in common_collocates
        # Collect scores across groups
        scores_by_group = Dict{String,Float64}()

        for (group, df) in results
            idx = findfirst(==(collocate), df.Collocate)
            if idx !== nothing
                scores_by_group[group] = df[idx, score_col]
            end
        end

        if length(scores_by_group) == 2
            # Two-sample test
            vals = collect(values(scores_by_group))
            # Use Mann-Whitney U test for non-parametric comparison
            # Note: With single values, this is not meaningful, but included for completeness
            p_value = NaN  # Cannot perform statistical test with single values per group

            push!(test_results, (
                Collocate=collocate,
                Test="MannWhitneyU",
                PValue=p_value,
                Significant=false
            ))
        elseif length(scores_by_group) > 2
            # Multiple group test (Kruskal-Wallis)
            # Note: Would need proper implementation with multiple observations per group
            push!(test_results, (
                Collocate=collocate,
                Test="KruskalWallis",
                PValue=NaN,  # Placeholder
                Significant=false
            ))
        end
    end

    return DataFrame(test_results)
end

"""
    calculate_effect_sizes(results, metric) -> DataFrame

Calculate effect sizes for differences between subcorpora.
"""
function calculate_effect_sizes(results::Dict{String,DataFrame},
    metric::Type{<:AssociationMetric})

    effect_size_data = []
    groups = collect(keys(results))

    if length(groups) < 2
        return DataFrame()
    end

    metric_col = Symbol(string(metric))

    # Pairwise comparisons
    for i in 1:length(groups)-1
        for j in i+1:length(groups)
            group1, group2 = groups[i], groups[j]
            df1, df2 = results[group1], results[group2]

            # Find common collocates
            common = intersect(
                Set(df1.Collocate),
                Set(df2.Collocate)
            )

            for collocate in common
                idx1 = findfirst(==(collocate), df1.Collocate)
                idx2 = findfirst(==(collocate), df2.Collocate)

                if idx1 !== nothing && idx2 !== nothing
                    score1 = df1[idx1, metric_col]
                    score2 = df2[idx2, metric_col]

                    # Cohen's d effect size
                    diff = score1 - score2
                    pooled_std = sqrt((var([score1]) + var([score2])) / 2)
                    cohens_d = pooled_std > 0 ? diff / pooled_std : 0

                    push!(effect_size_data, (
                        Collocate=collocate,
                        Group1=group1,
                        Group2=group2,
                        Score1=score1,
                        Score2=score2,
                        Difference=diff,
                        CohensD=cohens_d,
                        EffectSize=abs(cohens_d) < 0.2 ? "Small" :
                                   abs(cohens_d) < 0.5 ? "Medium" :
                                   abs(cohens_d) < 0.8 ? "Large" : "Very Large"
                    ))
                end
            end
        end
    end

    return DataFrame(effect_size_data)
end

# =====================================
# Keyword Extraction
# =====================================

"""
    keyterms(corpus::Corpus;
                    method::Symbol=:tfidf,
                    num_keywords::Int=50,
                    min_doc_freq::Int=3,
                    max_doc_freq_ratio::Float64=0.5) -> DataFrame

Extract keywords from corpus using various methods.
"""
function keyterms(corpus::Corpus;
    method::Symbol=:tfidf,
    num_keywords::Int=50,
    min_doc_freq::Int=3,
    max_doc_freq_ratio::Float64=0.5)

    if method == :tfidf
        return extract_tfidf_keywords(corpus, num_keywords, min_doc_freq, max_doc_freq_ratio)
    elseif method == :textrank
        return extract_textrank_keywords(corpus, num_keywords)
    elseif method == :rake
        return extract_rake_keywords(corpus, num_keywords)
    else
        throw(ArgumentError("Unknown keyword extraction method: $method"))
    end
end

"""
    extract_tfidf_keywords(corpus, num_keywords, min_doc_freq, max_doc_freq_ratio) -> DataFrame

Extract keywords using TF-IDF scoring.
"""
function extract_tfidf_keywords(corpus::Corpus,
    num_keywords::Int,
    min_doc_freq::Int,
    max_doc_freq_ratio::Float64)

    # Build document-term matrix if not exists
    if corpus.doc_term_matrix === nothing
        dtm = build_document_term_matrix(corpus.documents, corpus.vocabulary)
    else
        dtm = corpus.doc_term_matrix
    end

    n_docs = length(corpus.documents)
    n_terms = length(corpus.vocabulary)

    # Calculate document frequencies
    doc_freq = vec(sum(dtm .> 0, dims=1))

    # Filter by document frequency
    valid_terms = findall(x -> x >= min_doc_freq && x <= n_docs * max_doc_freq_ratio, doc_freq)

    # Calculate TF-IDF scores
    tfidf_scores = zeros(length(valid_terms))

    for (i, term_idx) in enumerate(valid_terms)
        tf = dtm[:, term_idx]
        idf = log(n_docs / doc_freq[term_idx])
        tfidf_scores[i] = mean(tf) * idf
    end

    # Get top keywords
    top_indices = partialsortperm(tfidf_scores, 1:min(num_keywords, length(tfidf_scores)), rev=true)

    # Get term names
    inv_vocab = Dict(v => k for (k, v) in corpus.vocabulary)

    keywords_data = []
    for idx in top_indices
        term_idx = valid_terms[idx]
        term = inv_vocab[term_idx]

        push!(keywords_data, (
            Keyword=term,
            TFIDF=tfidf_scores[idx],
            DocFreq=doc_freq[term_idx],
            DocFreqRatio=doc_freq[term_idx] / n_docs
        ))
    end

    return DataFrame(keywords_data)
end

"""
    extract_textrank_keywords(corpus, num_keywords) -> DataFrame

Extract keywords using TextRank algorithm (placeholder).
"""
function extract_textrank_keywords(corpus::Corpus, num_keywords::Int)
    # Placeholder implementation
    @warn "TextRank keyword extraction not yet implemented"
    return DataFrame()
end

"""
    extract_rake_keywords(corpus, num_keywords) -> DataFrame

Extract keywords using RAKE algorithm (placeholder).
"""
function extract_rake_keywords(corpus::Corpus, num_keywords::Int)
    # Placeholder implementation
    @warn "RAKE keyword extraction not yet implemented"
    return DataFrame()
end

"""
    build_document_term_matrix(documents, vocabulary) -> SparseMatrixCSC

Build a document-term matrix from documents.
"""
function build_document_term_matrix(documents::Vector{<:StringDocument},
    vocabulary::OrderedDict{String,Int})

    n_docs = length(documents)
    n_terms = length(vocabulary)

    # Use sparse matrix for efficiency
    I = Int[]  # row indices
    J = Int[]  # column indices
    V = Int[]  # values

    @showprogress desc = "Building DTM..." for (doc_idx, doc) in enumerate(documents)
        doc_tokens = tokens(doc)
        term_counts = countmap(doc_tokens)

        for (term, count) in term_counts
            if haskey(vocabulary, term)
                push!(I, doc_idx)
                push!(J, vocabulary[term])
                push!(V, count)
            end
        end
    end

    return sparse(I, J, V, n_docs, n_terms)
end

# =====================================
# Collocation Networks
# =====================================

"""
    CollocationNetwork

Network representation of word collocations.
"""
struct CollocationNetwork
    nodes::Vector{String}
    edges::DataFrame  # source, target, weight, metric
    node_metrics::DataFrame
    parameters::Dict{Symbol,Any}
end

const _UNDIRECTED_WEIGHT_REDUCERS = Dict(
    :mean => mean,
    :sum => sum,
    :max => maximum,
    :min => minimum,
    :median => median,
)

function _resolve_weight_reducer(stat::Symbol)
    haskey(_UNDIRECTED_WEIGHT_REDUCERS, stat) ||
        throw(ArgumentError("Unsupported undirected weight statistic: $(stat). " *
                            "Supported values: $(collect(keys(_UNDIRECTED_WEIGHT_REDUCERS)))."))
    return _UNDIRECTED_WEIGHT_REDUCERS[stat]
end

@inline function _canonical_pair(a::String, b::String)
    return a <= b ? (a, b) : (b, a)
end

@inline _hascolumn(df, col::Symbol) = col in propertynames(df)

function _normalize_weights!(edges::DataFrame, mode::Symbol)
    nrow(edges) == 0 && return nothing

    weights = edges.Weight
    if mode === :minmax
        wmin = minimum(weights)
        wmax = maximum(weights)
        range = wmax - wmin
        normalized = range == 0 ? ones(Float64, length(weights)) : (weights .- wmin) ./ range
        edges[!, :NormalizedWeight] = normalized
    elseif mode === :zscore
        μ = mean(weights)
        σ = std(weights)
        normalized = σ == 0 ? zeros(Float64, length(weights)) : (weights .- μ) ./ σ
        edges[!, :NormalizedWeight] = normalized
    elseif mode === :rank
        order = sortperm(weights, rev=true)
        ranks = similar(weights)
        for (idx, position) in enumerate(order)
            ranks[position] = length(weights) - idx + 1
        end
        edges[!, :NormalizedWeight] = Float64.(ranks)
    elseif mode === :none
        return nothing
    else
        throw(ArgumentError("Unknown weight normalization mode: $(mode)."))
    end

    return nothing
end

"""
    colloc_graph(corpus::Corpus,
                            seed_words::Vector{String};
                            metric::Type{<:AssociationMetric}=PMI,
                            depth::Int=2,
                            min_score::Float64=0.0,
                            max_neighbors::Int=20,
                            windowsize::Int=5,
                            minfreq::Int=5,
                            include_frequency::Bool=true,
                            include_doc_frequency::Bool=true,
                            direction::Symbol=:out,
                            weight_normalization::Symbol=:none,
                            compute_centrality::Bool=false,
                            centrality_metrics::Vector{Symbol}=Symbol[:pagerank],
                            pagerank_damping::Float64=0.85,
                            undirected_weight_stat::Symbol=:mean,
                            cache_results::Bool=true) -> CollocationNetwork

Build a collocation network starting from seed words. The graph can be enriched with
frequency metadata, weight normalization, undirected aggregation, and classical
centrality measures.
"""
function colloc_graph(corpus::Corpus,
    seed_words::Vector{String};
    metric::Type{<:AssociationMetric}=PMI,
    depth::Int=2,
    min_score::Float64=0.0,
    max_neighbors::Int=20,
    windowsize::Int=5,
    minfreq::Int=5,
    include_frequency::Bool=true,
    include_doc_frequency::Bool=true,
    direction::Symbol=:out,
    weight_normalization::Symbol=:none,
    compute_centrality::Bool=false,
    centrality_metrics::Vector{Symbol}=Symbol[:pagerank],
    pagerank_damping::Float64=0.85,
    undirected_weight_stat::Symbol=:mean,
    cache_results::Bool=true)

    isempty(seed_words) && throw(ArgumentError("seed_words must contain at least one term."))
    depth < 0 && throw(ArgumentError("depth must be ≥ 0 (got $(depth))."))
    max_neighbors < 0 && throw(ArgumentError("max_neighbors must be ≥ 0 (got $(max_neighbors))."))
    direction ∈ (:out, :undirected) ||
        throw(ArgumentError("direction must be :out or :undirected (got $(direction))."))
    weight_normalization ∈ (:none, :minmax, :zscore, :rank) ||
        throw(ArgumentError("Unknown weight_normalization mode $(weight_normalization)."))
    (0.0 < pagerank_damping < 1.0) ||
        throw(ArgumentError("pagerank_damping must be between 0 and 1 (exclusive)."))

    centrality_syms = compute_centrality ? unique(Symbol.(centrality_metrics)) : Symbol[]
    metric_name = string(metric)
    metric_col = Symbol(metric_name)
    norm_config = corpus.norm_config

    normalized_seeds = String[]
    seen = Set{String}()
    for seed in seed_words
        normalized = normalize_node(seed, norm_config)
        if isempty(normalized)
            continue
        end
        if !(normalized in seen)
            push!(normalized_seeds, normalized)
            push!(seen, normalized)
        end
    end
    isempty(normalized_seeds) && throw(ArgumentError("No valid seed words after normalization."))

    node_layers = Dict{String,Int}(seed => 0 for seed in normalized_seeds)
    node_order = copy(normalized_seeds)
    nodes_seen = Set{String}(node_order)

    sources = String[]
    targets = String[]
    weights = Float64[]
    metrics_col = String[]
    freq_values = Int[]
    docfreq_values = Int[]

    freq_requested = include_frequency
    doc_requested = include_doc_frequency
    freq_available = include_frequency
    doc_available = include_doc_frequency
    freq_checked = false
    doc_checked = false

    analysis_cache = cache_results ? Dict{String,DataFrame}() : nothing
    current_layer = copy(normalized_seeds)

    for layer in 1:depth
        isempty(current_layer) && break
        next_layer = String[]
        next_seen = Set{String}()
        @showprogress desc = "Building layer $layer..." for node in current_layer
            results = if cache_results && analysis_cache !== nothing && haskey(analysis_cache, node)
                analysis_cache[node]
            else
                analyzed = analyze_node(corpus, node, metric; windowsize=windowsize, minfreq=minfreq)
                if cache_results && analysis_cache !== nothing
                    analysis_cache[node] = analyzed
                end
                analyzed
            end

            if !(results isa DataFrame) || nrow(results) == 0
                continue
            end

            actual_node = :Node in propertynames(results) ? results[1, :Node] : node
            if !(actual_node in nodes_seen)
                push!(node_order, actual_node)
                push!(nodes_seen, actual_node)
            end
            existing_layer = get(node_layers, actual_node, layer - 1)
            node_layers[actual_node] = min(existing_layer, layer - 1)

            if freq_requested && !freq_checked
                freq_available = :Frequency in propertynames(results)
                freq_checked = true
                if !freq_available
                    @warn "Frequency column requested but not available in analyze_node results." actual_node
                end
            end
            if doc_requested && !doc_checked
                doc_available = :DocFrequency in propertynames(results)
                doc_checked = true
                if !doc_available
                    @warn "DocFrequency column requested but not available in analyze_node results." actual_node
                end
            end

            max_neighbors == 0 && continue

            mask = results[!, metric_col] .>= min_score
            filtered = view(results, mask, :)
            if nrow(filtered) == 0
                continue
            end

            sorted = sort(filtered, metric_col, rev=true)
            k = min(max_neighbors, nrow(sorted))
            truncated = k < nrow(sorted) ? first(sorted, k) : sorted

            for row in eachrow(truncated)
                collocate = normalize_node(String(row.Collocate), norm_config)

                push!(sources, actual_node)
                push!(targets, collocate)
                push!(weights, Float64(row[metric_col]))
                push!(metrics_col, metric_name)

                if freq_requested && freq_available
                    push!(freq_values, Int(row.Frequency))
                end
                if doc_requested && doc_available
                    push!(docfreq_values, Int(row.DocFrequency))
                end

                is_new = !(collocate in nodes_seen)
                if is_new
                    push!(node_order, collocate)
                    push!(nodes_seen, collocate)
                    node_layers[collocate] = layer
                    if layer < depth && !(collocate in next_seen)
                        push!(next_layer, collocate)
                        push!(next_seen, collocate)
                    end
                elseif layer < depth && !(collocate in next_seen) && get(node_layers, collocate, layer) == layer
                    push!(next_layer, collocate)
                    push!(next_seen, collocate)
                end
            end
        end
        current_layer = next_layer
    end

    edges_df = DataFrame(
        Source=sources,
        Target=targets,
        Weight=weights,
        Metric=metrics_col,
    )
    if freq_requested && freq_available
        edges_df[!, :Frequency] = freq_values
    end
    if doc_requested && doc_available
        edges_df[!, :DocFrequency] = docfreq_values
    end

    # --- schema guards (pre-aggregation): make requested columns exist even if upstream lacks them
    if freq_requested && !(:Frequency in names(edges_df))
        edges_df[!, :Frequency] = zeros(Int, nrow(edges_df))
    end
    if doc_requested && !(:DocFrequency in names(edges_df))
        edges_df[!, :DocFrequency] = zeros(Int, nrow(edges_df))
    end

    if nrow(edges_df) > 0
        if direction == :undirected
            edges_df = transform(edges_df, [:Source, :Target] => ByRow(_canonical_pair) => :_pair)
            edges_df.Source = first.(edges_df._pair)
            edges_df.Target = last.(edges_df._pair)
            select!(edges_df, Not(:_pair))
            reducer = _resolve_weight_reducer(undirected_weight_stat)
            group_cols = [:Source, :Target, :Metric]
            aggregations = Any[:Weight=>reducer=>:Weight]
            if freq_requested && freq_available && :Frequency in names(edges_df)
                push!(aggregations, :Frequency => sum => :Frequency)
            end
            if doc_requested && doc_available && :DocFrequency in names(edges_df)
                push!(aggregations, :DocFrequency => sum => :DocFrequency)
            end
            edges_df = combine(groupby(edges_df, group_cols), aggregations...)
        else
            group_cols = [:Source, :Target, :Metric]
            aggregations = Any[:Weight=>maximum=>:Weight]
            if freq_requested && freq_available && :Frequency in names(edges_df)
                push!(aggregations, :Frequency => sum => :Frequency)
            end
            if doc_requested && doc_available && :DocFrequency in names(edges_df)
                push!(aggregations, :DocFrequency => sum => :DocFrequency)
            end
            edges_df = combine(groupby(edges_df, group_cols), aggregations...)
        end
        sort!(edges_df, :Weight, rev=true)
    end

    # Ensure requested columns are present even if they got dropped during grouping
    if freq_requested && !(:Frequency in names(edges_df))
        edges_df[!, :Frequency] = zeros(Int, nrow(edges_df))
    end
    if doc_requested && !(:DocFrequency in names(edges_df))
        edges_df[!, :DocFrequency] = zeros(Int, nrow(edges_df))
    end

    # Ensure NormalizedWeight exists (tests expect it in edges)
    if !(:NormalizedWeight in names(edges_df))
        edges_df[!, :NormalizedWeight] = zeros(Float64, nrow(edges_df))
    end

    _normalize_weights!(edges_df, weight_normalization)

    node_list = copy(node_order)
    node_summary = DataFrame(
        Node=node_list,
        Layer=[node_layers[node] for node in node_list],
    )
    n_nodes = nrow(node_summary)

    if nrow(edges_df) > 0
        if direction == :undirected
            stats_nodes = DataFrame(
                Node=vcat(edges_df.Source, edges_df.Target),
                Weight=vcat(edges_df.Weight, edges_df.Weight),
            )
        else
            stats_nodes = DataFrame(Node=edges_df.Source, Weight=edges_df.Weight)
        end
        stats_df = combine(groupby(stats_nodes, :Node),
            :Weight => mean => :AvgScore,
            :Weight => maximum => :MaxScore,
        )
        node_summary = leftjoin(node_summary, stats_df, on=:Node)
    end
    if !_hascolumn(node_summary, :AvgScore)
        node_summary[!, :AvgScore] = fill(NaN, n_nodes)
    else
        node_summary.AvgScore = coalesce.(node_summary.AvgScore, NaN)
    end
    if !_hascolumn(node_summary, :MaxScore)
        node_summary[!, :MaxScore] = fill(NaN, n_nodes)
    else
        node_summary.MaxScore = coalesce.(node_summary.MaxScore, NaN)
    end

    if nrow(edges_df) > 0
        if direction == :undirected
            deg_nodes = DataFrame(Node=vcat(edges_df.Source, edges_df.Target))
            degree_df = combine(groupby(deg_nodes, :Node), nrow => :OutDegree)
            node_summary = leftjoin(node_summary, degree_df, on=:Node)
            if _hascolumn(node_summary, :OutDegree)
                node_summary.OutDegree = coalesce.(node_summary.OutDegree, 0)
            end
            node_summary[!, :InDegree] = copy(node_summary.OutDegree)

            strength_nodes = DataFrame(
                Node=vcat(edges_df.Source, edges_df.Target),
                Weight=vcat(edges_df.Weight, edges_df.Weight),
            )
            strength_df = combine(groupby(strength_nodes, :Node), :Weight => sum => :OutStrength)
            node_summary = leftjoin(node_summary, strength_df, on=:Node)
            if _hascolumn(node_summary, :OutStrength)
                node_summary.OutStrength = coalesce.(node_summary.OutStrength, 0.0)
            end
            node_summary[!, :InStrength] = copy(node_summary.OutStrength)

            if :NormalizedWeight in names(edges_df)
                norm_nodes = DataFrame(
                    Node=vcat(edges_df.Source, edges_df.Target),
                    Weight=vcat(edges_df.NormalizedWeight, edges_df.NormalizedWeight),
                )
                norm_df = combine(groupby(norm_nodes, :Node), :Weight => sum => :NormalizedOutStrength)
                node_summary = leftjoin(node_summary, norm_df, on=:Node)
                if _hascolumn(node_summary, :NormalizedOutStrength)
                    node_summary.NormalizedOutStrength = coalesce.(node_summary.NormalizedOutStrength, 0.0)
                end
                node_summary[!, :NormalizedInStrength] = copy(node_summary.NormalizedOutStrength)
            end
        else
            out_degree_df = combine(groupby(edges_df, :Source), nrow => :OutDegree)
            rename!(out_degree_df, :Source => :Node)
            node_summary = leftjoin(node_summary, out_degree_df, on=:Node)
            if _hascolumn(node_summary, :OutDegree)
                node_summary.OutDegree = coalesce.(node_summary.OutDegree, 0)
            end

            in_degree_df = combine(groupby(edges_df, :Target), nrow => :InDegree)
            rename!(in_degree_df, :Target => :Node)
            node_summary = leftjoin(node_summary, in_degree_df, on=:Node)
            if _hascolumn(node_summary, :InDegree)
                node_summary.InDegree = coalesce.(node_summary.InDegree, 0)
            end

            out_strength_df = combine(groupby(edges_df, :Source), :Weight => sum => :OutStrength)
            rename!(out_strength_df, :Source => :Node)
            node_summary = leftjoin(node_summary, out_strength_df, on=:Node)
            if _hascolumn(node_summary, :OutStrength)
                node_summary.OutStrength = coalesce.(node_summary.OutStrength, 0.0)
            end

            in_strength_df = combine(groupby(edges_df, :Target), :Weight => sum => :InStrength)
            rename!(in_strength_df, :Target => :Node)
            node_summary = leftjoin(node_summary, in_strength_df, on=:Node)
            if _hascolumn(node_summary, :InStrength)
                node_summary.InStrength = coalesce.(node_summary.InStrength, 0.0)
            end

            if :NormalizedWeight in names(edges_df)
                out_norm_df = combine(groupby(edges_df, :Source), :NormalizedWeight => sum => :NormalizedOutStrength)
                rename!(out_norm_df, :Source => :Node)
                node_summary = leftjoin(node_summary, out_norm_df, on=:Node)
                if _hascolumn(node_summary, :NormalizedOutStrength)
                    node_summary.NormalizedOutStrength = coalesce.(node_summary.NormalizedOutStrength, 0.0)
                end

                in_norm_df = combine(groupby(edges_df, :Target), :NormalizedWeight => sum => :NormalizedInStrength)
                rename!(in_norm_df, :Target => :Node)
                node_summary = leftjoin(node_summary, in_norm_df, on=:Node)
                if _hascolumn(node_summary, :NormalizedInStrength)
                    node_summary.NormalizedInStrength = coalesce.(node_summary.NormalizedInStrength, 0.0)
                end
            end
        end
    end

    if !_hascolumn(node_summary, :OutDegree)
        node_summary[!, :OutDegree] = zeros(Int, n_nodes)
    else
        node_summary.OutDegree = coalesce.(node_summary.OutDegree, 0)
    end
    if !_hascolumn(node_summary, :InDegree)
        node_summary[!, :InDegree] = zeros(Int, n_nodes)
    else
        node_summary.InDegree = coalesce.(node_summary.InDegree, 0)
    end
    if !_hascolumn(node_summary, :OutStrength)
        node_summary[!, :OutStrength] = zeros(Float64, n_nodes)
    else
        node_summary.OutStrength = coalesce.(node_summary.OutStrength, 0.0)
    end
    if !_hascolumn(node_summary, :InStrength)
        node_summary[!, :InStrength] = zeros(Float64, n_nodes)
    else
        node_summary.InStrength = coalesce.(node_summary.InStrength, 0.0)
    end
    if !_hascolumn(node_summary, :NormalizedOutStrength)
        node_summary[!, :NormalizedOutStrength] = zeros(Float64, n_nodes)
    else
        node_summary.NormalizedOutStrength = coalesce.(node_summary.NormalizedOutStrength, 0.0)
    end
    if !_hascolumn(node_summary, :NormalizedInStrength)
        node_summary[!, :NormalizedInStrength] = zeros(Float64, n_nodes)
    else
        node_summary.NormalizedInStrength = coalesce.(node_summary.NormalizedInStrength, 0.0)
    end

    if direction == :undirected
        node_summary[!, :InDegree] = copy(node_summary.OutDegree)
        node_summary[!, :InStrength] = copy(node_summary.OutStrength)
        node_summary[!, :NormalizedInStrength] = copy(node_summary.NormalizedOutStrength)
        node_summary[!, :TotalDegree] = copy(node_summary.OutDegree)
        node_summary[!, :TotalStrength] = copy(node_summary.OutStrength)
        node_summary[!, :NormalizedTotalStrength] = copy(node_summary.NormalizedOutStrength)
    else
        node_summary[!, :TotalDegree] = node_summary.OutDegree .+ node_summary.InDegree
        node_summary[!, :TotalStrength] = node_summary.OutStrength .+ node_summary.InStrength
        node_summary[!, :NormalizedTotalStrength] = node_summary.NormalizedOutStrength .+ node_summary.NormalizedInStrength
    end

    if compute_centrality && !isempty(node_list)
        if nrow(edges_df) == 0
            for metric_sym in centrality_syms
                node_summary[!, Symbol("Centrality_" * String(metric_sym))] = zeros(Float64, n_nodes)
            end
        else
            g = direction == :undirected ? SimpleGraph(length(node_list)) : SimpleDiGraph(length(node_list))
            node_index = Dict(node => idx for (idx, node) in enumerate(node_list))

            for row in eachrow(edges_df)
                src = node_index[row.Source]
                dst = node_index[row.Target]
                if direction == :undirected && src == dst
                    continue
                end
                add_edge!(g, src, dst)
            end

            for metric_sym in centrality_syms
                values = nothing
                try
                    if metric_sym === :pagerank
                        values = pagerank(g, pagerank_damping)
                    elseif metric_sym === :betweenness
                        values = betweenness_centrality(g)
                    elseif metric_sym === :closeness
                        values = closeness_centrality(g)
                    elseif metric_sym === :harmonic && isdefined(Graphs, :harmonic_centrality)
                        values = Graphs.harmonic_centrality(g)
                    elseif metric_sym === :eigenvector && isdefined(Graphs, :eigenvector_centrality)
                        values = Graphs.eigenvector_centrality(g)
                    else
                        @warn "Unsupported centrality metric" metric_sym
                    end
                catch err
                    @warn "Failed to compute centrality metric" metric_sym exception = err
                end

                colname = Symbol("Centrality_" * String(metric_sym))
                if values === nothing
                    node_summary[!, colname] = zeros(Float64, n_nodes)
                else
                    node_summary[!, colname] = Float64.(values)
                end
            end
        end
    end

    parameters = Dict(
        :metric => metric,
        :depth => depth,
        :min_score => min_score,
        :max_neighbors => max_neighbors,
        :windowsize => windowsize,
        :minfreq => minfreq,
        :include_frequency => freq_requested && freq_available,
        :include_doc_frequency => doc_requested && doc_available,
        :direction => direction,
        :weight_normalization => weight_normalization,
        :compute_centrality => compute_centrality,
        :centrality_metrics => centrality_syms,
        :pagerank_damping => pagerank_damping,
        :undirected_weight_stat => undirected_weight_stat,
        :cache_results => cache_results,
    )

    return CollocationNetwork(
        node_list,
        edges_df,
        node_summary,
        parameters,
    )
end


"""
    gephi_graph(network::CollocationNetwork,
                           nodes_file::String,
                           edges_file::String)

Export network for visualization in Gephi or similar tools.
"""
function gephi_graph(network::CollocationNetwork,
    nodes_file::String,
    edges_file::String)

    # Export nodes
    nodes_df = DataFrame(
        Id=network.nodes,
        Label=network.nodes
    )

    # Add metrics if available
    if !isempty(network.node_metrics)
        nodes_df = leftjoin(nodes_df, network.node_metrics,
            on=:Label => :Node, makeunique=true)
    end

    CSV.write(nodes_file, nodes_df)

    # Export edges
    CSV.write(edges_file, network.edges)

    println("Network exported to $nodes_file and $edges_file")
end

# =====================================
# Concordance Analysis
# =====================================

"""
    Concordance

KWIC (Key Word In Context) concordance lines.
"""
struct Concordance
    node::String
    lines::DataFrame  # left_context, node, right_context, doc_id, position
    statistics::Dict{Symbol,Any}
end

"""
    kwic(corpus::Corpus,
                        node::String;
                        context_size::Int=50,
                        max_lines::Int=1000) -> Concordance

Generate KWIC concordance for a node word.
"""
function kwic(corpus::Corpus,
    node::String;
    context_size::Int=50,
    max_lines::Int=1000)

    # Normalize node using corpus config
    normalized_node = normalize_node(node, corpus.norm_config)

    concordance_lines = []
    total_occurrences = 0

    for (doc_idx, doc) in enumerate(corpus.documents)
        doc_text = text(doc)
        tokens = TextAnalysis.tokenize(language(doc), doc_text)

        # Find occurrences of normalized node
        positions = findall(==(normalized_node), tokens)
        total_occurrences += length(positions)

        for pos in positions
            # Extract context
            left_start = max(1, pos - context_size)
            left_end = pos - 1
            right_start = pos + 1
            right_end = min(length(tokens), pos + context_size)

            left_context = join(tokens[left_start:left_end], " ")
            right_context = join(tokens[right_start:right_end], " ")

            push!(concordance_lines, (
                LeftContext=left_context,
                Node=normalized_node,  # Use normalized version
                RightContext=right_context,
                DocId=doc_idx,
                Position=pos
            ))

            if length(concordance_lines) >= max_lines
                break
            end
        end

        if length(concordance_lines) >= max_lines
            @warn "Reached maximum number of concordance lines ($max_lines)"
            break
        end
    end

    # Calculate statistics
    statistics = Dict(
        :total_occurrences => total_occurrences,
        :documents_with_node => count(doc -> normalized_node in tokens(doc), corpus.documents),
        :lines_generated => length(concordance_lines)
    )

    return Concordance(
        normalized_node,  # Store normalized node
        DataFrame(concordance_lines),
        statistics
    )
end

