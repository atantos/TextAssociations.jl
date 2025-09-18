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
    temporal_corpus_analysis(corpus::Corpus,
                            nodes::Vector{String},
                            time_field::Symbol,
                            metric::Type{<:AssociationMetric};
                            time_bins::Int=10,
                            windowsize::Int=5,
                            minfreq::Int=5) -> TemporalCorpusAnalysis

Analyze how word associations change over time.
"""
function temporal_corpus_analysis(corpus::Corpus,
    nodes::Vector{String},
    time_field::Symbol,
    metric::Type{<:AssociationMetric};
    time_bins::Int=10,
    windowsize::Int=5,
    minfreq::Int=5)

    # Extract time information from metadata
    time_values = []
    doc_indices = Int[]

    for (i, (key, meta)) in enumerate(corpus.metadata)
        if haskey(meta, time_field)
            push!(time_values, meta[time_field])
            push!(doc_indices, i)
        end
    end

    if isempty(time_values)
        throw(ArgumentError("No documents have the specified time field: $time_field"))
    end


    # Create time bins
    parsed_dates = nothing
    is_numeric = eltype(time_values) <: Number
    if !is_numeric
        # try to parse dates; keep nothing for unparsable
        parsed = [parse_time_value(tv) for tv in time_values]
        if all(!isnothing.(parsed))
            parsed_dates = Date.(parsed)
        end
    end

    if is_numeric
        min_time, max_time = extrema(time_values)
        bin_edges = collect(range(min_time, max_time, length=time_bins + 1))
        bin_labels = ["Period_$i" for i in 1:time_bins]
        # Create value labels directly without nested functions
        value_labels = String[]
        for x in time_values
            bin_idx = findfirst(i -> bin_edges[i] <= x < bin_edges[i+1], 1:time_bins)
            push!(value_labels, bin_labels[bin_idx])
        end
    elseif parsed_dates !== nothing
        # date binning by equal-width between min/max date
        min_time, max_time = extrema(parsed_dates)
        # avoid zero range
        if min_time == max_time
            bin_labels = ["$(min_time)"]
            value_labels = [bin_labels[1] for _ in parsed_dates]
        else
            dayspan = Dates.value(max_time - min_time)
            edges = [min_time + Dates.Day(round(Int, i * dayspan / time_bins)) for i in 0:time_bins]
            bin_edges = edges
            bin_labels = ["$(edges[i])_to_$(edges[i+1])" for i in 1:length(edges)-1]
            # Create value labels directly
            value_labels = String[]
            for x in parsed_dates
                bin_idx = findfirst(i -> bin_edges[i] <= x < bin_edges[i+1], 1:time_bins)
                push!(value_labels, bin_labels[bin_idx])
            end
        end
    else
        # categorical labels: group sorted unique values into approximately `time_bins` chunks
        sorted_times = sort(unique(string.(time_values)))
        n_unique = length(sorted_times)
        if n_unique <= time_bins
            bin_labels = sorted_times
            value_labels = string.(time_values)
        else
            step = ceil(Int, n_unique / time_bins)
            bin_labels = String[]
            # map each category directly to itself (fine-grained) for now
            value_labels = string.(time_values)
        end
    end

    # Split corpus into time periods
    results_by_period = Dict{String,MultiNodeAnalysis}()

    @showprogress desc = "Analyzing time periods..." for (period_idx, period_label) in enumerate(bin_labels)
        # Get documents for this period
        period_docs = StringDocument[]

        for (i, idx) in enumerate(doc_indices)
            if value_labels[i] == bin_labels[period_idx]
                push!(period_docs, corpus.documents[idx])
            end
        end

        if !isempty(period_docs)
            period_corpus = Corpus(period_docs)

            # Analyze this period
            period_analysis = analyze_multiple_nodes(
                period_corpus, nodes, [metric],
                windowsize=windowsize, minfreq=minfreq
            )

            results_by_period[period_label] = period_analysis
        end
    end

    # Compute trends
    trend_analysis = compute_association_trends(results_by_period, nodes, metric)

    return TemporalCorpusAnalysis(
        collect(keys(results_by_period)),
        results_by_period,
        trend_analysis,
        corpus
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
        all_collocates = Set{Symbol}()
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
    node::String,
    metric::Type{<:AssociationMetric};
    windowsize::Int=5,
    minfreq::Int=5)

    # Split corpus by field
    subcorpora = Dict{String,Corpus}()
    doc_groups = Dict{String,Vector{StringDocument}}()

    for (i, (key, meta)) in enumerate(corpus.metadata)
        if haskey(meta, split_field)
            group = string(meta[split_field])
            if !haskey(doc_groups, group)
                doc_groups[group] = StringDocument[]
            end
            push!(doc_groups[group], corpus.documents[i])
        end
    end

    # Create subcorpora
    for (group, docs) in doc_groups
        subcorpora[group] = Corpus(docs)
    end

    println("Split corpus into $(length(subcorpora)) subcorpora")

    # Analyze each subcorpus
    results = Dict{String,DataFrame}()

    @showprogress desc = "Analyzing subcorpora..." for (group, subcorpus) in subcorpora
        group_results = analyze_corpus(subcorpus, node, metric,
            windowsize=windowsize, minfreq=minfreq)
        results[group] = group_results
    end

    # Statistical comparison
    statistical_tests = perform_statistical_tests(results, metric)
    effect_sizes = calculate_effect_sizes(results, metric)

    return SubcorpusComparison(
        subcorpora,
        node,
        results,
        statistical_tests,
        effect_sizes
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
    common_collocates = Set{Symbol}()
    for (group, df) in results
        if !isempty(df)
            if isempty(common_collocates)
                common_collocates = Set(df.Collocate)
            else
                intersect!(common_collocates, Set(df.Collocate))
            end
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
    extract_keywords(corpus::Corpus;
                    method::Symbol=:tfidf,
                    num_keywords::Int=50,
                    min_doc_freq::Int=3,
                    max_doc_freq_ratio::Float64=0.5) -> DataFrame

Extract keywords from corpus using various methods.
"""
function extract_keywords(corpus::Corpus;
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
function build_document_term_matrix(documents::Vector{StringDocument},
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
    build_collocation_network(corpus::Corpus,
                            seed_words::Vector{String};
                            metric::Type{<:AssociationMetric}=PMI,
                            depth::Int=2,
                            min_score::Float64=3.0,
                            max_neighbors::Int=20) -> CollocationNetwork

Build a collocation network starting from seed words.
"""
function build_collocation_network(corpus::Corpus,
    seed_words::Vector{String};
    metric::Type{<:AssociationMetric}=PMI,
    depth::Int=2,
    min_score::Float64=3.0,
    max_neighbors::Int=20,
    windowsize::Int=5,
    minfreq::Int=5)

    nodes = Set{String}(seed_words)
    edges = NamedTuple{(:Source, :Target, :Weight, :Metric),Tuple{String,String,Float64,String}}[]
    node_metrics_data = NamedTuple{(:Node, :Degree, :AvgScore, :MaxScore, :Layer),
        Tuple{String,Int,Float64,Float64,Int}}[]

    current_layer = Set(seed_words)

    for layer in 1:depth
        next_layer = Set{String}()

        @showprogress desc = "Building layer $layer..." for node in current_layer
            results = analyze_corpus(corpus, node, metric; windowsize=windowsize, minfreq=minfreq)

            if results isa DataFrame && nrow(results) > 0
                # keep only rows above threshold
                filtered = view(results, results.Score .>= min_score, :)
                if nrow(filtered) > 0
                    k = min(max_neighbors, nrow(filtered))
                    top_neighbors = first(filtered, k)  # DataFrames.first(df, k)

                    for row in eachrow(top_neighbors)
                        collocate = String(row.Collocate)
                        push!(edges, (Source=node,
                            Target=collocate,
                            Weight=Float64(row.Score),
                            Metric=string(metric)))
                        if !(collocate in nodes)
                            push!(next_layer, collocate)
                            push!(nodes, collocate)
                        end
                    end

                    push!(node_metrics_data, (Node=node,
                        Degree=nrow(top_neighbors),
                        AvgScore=mean(top_neighbors.Score),
                        MaxScore=maximum(top_neighbors.Score),
                        Layer=layer - 1))
                else
                    push!(node_metrics_data, (Node=node, Degree=0, AvgScore=NaN, MaxScore=NaN, Layer=layer - 1))
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
        collect(nodes),          # Vector{String}
        DataFrame(edges),        # edges table
        DataFrame(node_metrics_data),  # per-node metrics
        parameters
    )
end


"""
    export_network_to_gephi(network::CollocationNetwork,
                           nodes_file::String,
                           edges_file::String)

Export network for visualization in Gephi or similar tools.
"""
function export_network_to_gephi(network::CollocationNetwork,
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
    generate_concordance(corpus::Corpus,
                        node::String;
                        context_size::Int=50,
                        max_lines::Int=1000) -> Concordance

Generate KWIC concordance for a node word.
"""
function generate_concordance(corpus::Corpus,
    node::String;
    context_size::Int=50,
    max_lines::Int=1000)

    concordance_lines = []
    total_occurrences = 0

    for (doc_idx, doc) in enumerate(corpus.documents)
        doc_text = text(doc)
        tokens = TextAnalysis.tokenize(language(doc), doc_text)

        # Find occurrences
        positions = findall(==(node), tokens)
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
                Node=node,
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
        :documents_with_node => count(doc -> node in tokens(doc), corpus.documents),
        :lines_generated => length(concordance_lines)
    )

    return Concordance(
        node,
        DataFrame(concordance_lines),
        statistics
    )
end

# =====================================
# Example Advanced Usage
# =====================================

function demonstrate_advanced_features()
    # Load corpus with metadata
    corpus = load_corpus("research_papers.csv",
        text_column=:abstract,
        metadata_columns=[:year, :field, :journal])

    # 1. Temporal Analysis
    temporal_results = temporal_corpus_analysis(
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
    keywords = extract_keywords(corpus, method=:tfidf, num_keywords=30)
    println("\nTop keywords:")
    println(first(keywords, 10))

    # 4. Collocation Network
    network = build_collocation_network(
        corpus,
        ["innovation", "technology"],
        metric=PMI,
        depth=2,
        min_score=3.0
    )

    export_network_to_gephi(network, "nodes.csv", "edges.csv")

    # 5. Concordance
    concordance = generate_concordance(corpus, "innovation", context_size=30)
    println("\nConcordance lines:")
    println(first(concordance.lines, 5))

    return (temporal_results, field_comparison, keywords, network, concordance)
end