# =====================================
# File: src/core/advanced_corpus.jl
# Advanced corpus analysis features
# =====================================

using StatsBase: countmap
using Statistics: mean, median, std, var, cor
using ProgressMeter: @showprogress
using SparseArrays: sparse
using Dates

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
    statistical_tests::DataFrame
    effect_sizes::DataFrame
end

# =====================================
# Temporal Analysis
# =====================================

"""
    analyze_temporal(corpus::Corpus,
                            nodes::Vector{String},
                            time_field::Symbol,
                            metric::Type{<:AssociationMetric};
                            time_bins::Int=10,
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
        results[g] = df
    end

    # 3) Summary & parameters dataframes (to satisfy the 4th/5th args)
    summary = DataFrame(
        Subcorpus=group_names,
        NumCollocates=[haskey(results, g) && !isempty(results[g]) ? nrow(results[g]) : 0 for g in group_names]
    )

    parameters = DataFrame(
        Parameter=["split_field", "node", "metric", "windowsize", "minfreq"],
        Value=[string(split_field), String(node), string(metric), string(windowsize), string(minfreq)]
    )

    # 4) Return in the expected signature
    return SubcorpusComparison(
        subcorpora,               # Dict{String,Corpus}
        String(node),             # node
        results,                  # Dict{String,DataFrame}
        summary,                  # DataFrame
        parameters                # DataFrame
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

    metric_col = Symbol(string(metric))

    for collocate in common_collocates
        # Collect scores across groups
        scores_by_group = Dict{String,Float64}()

        for (group, df) in results
            idx = findfirst(==(collocate), df.Collocate)
            if idx !== nothing
                scores_by_group[group] = df[idx, metric_col]
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

"""
    colloc_graph(corpus::Corpus,
                            seed_words::Vector{String};
                            metric::Type{<:AssociationMetric}=PMI,
                            depth::Int=2,
                            min_score::Float64=3.0,
                            max_neighbors::Int=20) -> CollocationNetwork

Build a collocation network starting from seed words.
"""
function colloc_graph(corpus::Corpus,
    seed_words::Vector{String};
    metric::Type{<:AssociationMetric}=PMI,
    depth::Int=2,
    min_score::Float64=0.0,
    max_neighbors::Int=20,
    windowsize::Int=5,
    minfreq::Int=5)

    # Get preprocessing options and normalize seed words
    prep_opts = get(corpus.metadata, "_preprocessing_options", Dict())
    normalized_seeds = [normalize_node(word, corpus.norm_config) for word in seed_words]

    nodes = Set{String}(normalized_seeds)
    edges = NamedTuple{(:Source, :Target, :Weight, :Metric),Tuple{String,String,Float64,String}}[]
    node_metrics_data = NamedTuple{(:Node, :Degree, :AvgScore, :MaxScore, :Layer),
        Tuple{String,Int,Float64,Float64,Int}}[]

    current_layer = Set(normalized_seeds)

    for layer in 1:depth
        next_layer = Set{String}()

        @showprogress desc = "Building layer $layer..." for node in current_layer
            # analyze_node will handle any further normalization
            results = analyze_node(corpus, node, metric; windowsize=windowsize, minfreq=minfreq)

            if results isa DataFrame && nrow(results) > 0
                # The node in results is already normalized
                actual_node = nrow(results) > 0 ? results[1, :Node] : node

                # keep only rows above threshold
                filtered = view(results, results.Score .>= min_score, :)
                if nrow(filtered) > 0
                    k = min(max_neighbors, nrow(filtered))
                    top_neighbors = first(filtered, k)

                    for row in eachrow(top_neighbors)
                        collocate = String(row.Collocate)
                        # Normalize the collocate as well since it might become a node
                        normalized_collocate = normalize_node(collocate, corpus.norm_config)

                        push!(edges, (Source=actual_node,
                            Target=normalized_collocate,
                            Weight=Float64(row.Score),
                            Metric=string(metric)))

                        if !(normalized_collocate in nodes)
                            push!(next_layer, normalized_collocate)
                            push!(nodes, normalized_collocate)
                        end
                    end

                    push!(node_metrics_data, (Node=actual_node,
                        Degree=nrow(top_neighbors),
                        AvgScore=mean(top_neighbors.Score),
                        MaxScore=maximum(top_neighbors.Score),
                        Layer=layer - 1))
                else
                    push!(node_metrics_data, (Node=actual_node, Degree=0, AvgScore=NaN, MaxScore=NaN, Layer=layer - 1))
                end
            else
                @info "No collocations found for node: $node"
                push!(node_metrics_data, (Node=node, Degree=0, AvgScore=NaN, MaxScore=NaN, Layer=layer - 1))
            end
        end

        current_layer = next_layer
        isempty(current_layer) && break
    end

    parameters = Dict(
        :metric => metric,
        :depth => depth,
        :min_score => min_score,
        :max_neighbors => max_neighbors,
        :windowsize => windowsize,
        :minfreq => minfreq,
    )

    return CollocationNetwork(
        collect(nodes),          # Vector{String} - all normalized
        DataFrame(edges),        # edges table with normalized nodes
        DataFrame(node_metrics_data),  # per-node metrics
        parameters
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

# =====================================
# Example Advanced Usage
# =====================================

function demonstrate_advanced_features()
    # Load corpus with metadata
    corpus = read_corpus("research_papers.csv",
        text_column=:abstract,
        metadata_columns=[:year, :field, :journal])

    # 1. Temporal Analysis
    temporal_results = analyze_temporal(
        corpus,
        ["innovation", "technology", "research"],
        :year,
        PMI,
        time_bins=5
    )

    println("Trend analysis:")
    println(first(temporal_results.trend_analysis, 10))

    # 2. Subcorpus Comparison
    field_comparison = compare_subcorpora(
        corpus,
        :field,
        "innovation",
        PMI,
        windowsize=5
    )

    println("\nStatistical tests between fields:")
    println(first(field_comparison.statistical_tests, 10))

    # 3. Keyword Extraction
    keywords = keyterms(corpus, method=:tfidf, num_keywords=30)
    println("\nTop keywords:")
    println(first(keywords, 10))

    # 4. Collocation Network
    network = colloc_graph(
        corpus,
        ["innovation", "technology"],
        metric=PMI,
        depth=2,
        min_score=3.0
    )

    gephi_graph(network, "nodes.csv", "edges.csv")

    # 5. Concordance
    concordance = kwic(corpus, "innovation", context_size=30)
    println("\nConcordance lines:")
    println(first(concordance.lines, 5))

    return (temporal_results, field_comparison, keywords, network, concordance)
end