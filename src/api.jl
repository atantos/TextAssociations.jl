# =====================================
# File: src/api.jl
# Public API functions - UPDATED to return DataFrame by default
# =====================================

# =====================================
# Core assoc_score function
# =====================================

# Trait to indicate which metrics need tokens
# Default: metrics don't need tokens
NeedsTokens(::Type{T}) where {T<:AssociationMetric} = Val(false)

# Internal: resolve metric entry point by naming convention
@inline function _resolve_metric_function(::Type{T}) where {T<:AssociationMetric}
    fname = Symbol("eval_", lowercase(String(nameof(T))))
    if !isdefined(@__MODULE__, fname)
        throw(ArgumentError("Unknown metric type $(T). Expected a function `$fname`."))
    end
    return getfield(@__MODULE__, fname)
end

"""
    assoc_score(metricType::Type{<:AssociationMetric}, x::AssociationDataFormat;
              scores_only::Bool=false,
              tokens::Union{Nothing,Vector{String}}=nothing,
              kwargs...)

Evaluate a metric on any association data format (CT or CCT).

- If the metric requires tokens (e.g., LexicalGravity), pass `tokens=...` or
  implement `assoc_tokens(::YourType)` to supply them automatically.
- Returns a DataFrame by default: [:Node, :Collocate, :Frequency, :<MetricName>].
- If `scores_only=true`, returns only the scores Vector.
"""
function assoc_score(::Type{T}, x::AssociationDataFormat;
    scores_only::Bool=false,
    tokens::Union{Nothing,Vector{String}}=nothing,
    kwargs...) where {T<:AssociationMetric}

    f = _resolve_metric_function(T)
    needs_tok = NeedsTokens(T)

    # Compute scores
    scores = if needs_tok === Val(true)
        # This metric NEEDS tokens, so get them and pass them
        toks = tokens === nothing ? assoc_tokens(x) : tokens
        toks === nothing && error("$(T) requires tokens; pass `tokens=...` or implement assoc_tokens(::$(typeof(x))).")
        f(x; tokens=toks, kwargs...)
    else
        # Do NOT pass tokens kwarg if not needed, to avoid kwarg errors on methods that don't accept it.
        f(x; kwargs...)
    end

    # Fast path
    if scores_only
        return scores
    end

    # Build standard DataFrame output
    df = assoc_df(x)
    if isempty(df)
        return DataFrame()
    end

    out = DataFrame()
    out[!, :Node] = fill(assoc_node(x), nrow(df))
    out[!, :Collocate] = String.(df.Collocate)
    # Assumes your contingency schema puts co-occurrence freq in :a
    out[!, :Frequency] = df.a
    out[!, Symbol(nameof(T))] = scores
    return out
end

"""
    assoc_score(metrics::AbstractVector{<:Type{<:AssociationMetric}},
              x::AssociationDataFormat;
              scores_only::Bool=false,
              tokens::Union{Nothing,Vector{String}}=nothing,
              kwargs...)

Evaluate multiple metrics on CT or CCT.

- Returns a DataFrame with one column per metric by default.
- If `scores_only=true`, returns Dict{String,Vector{Float64}}.
"""
function assoc_score(metrics::AbstractVector{<:Type{<:AssociationMetric}},
    x::AssociationDataFormat;
    scores_only::Bool=false,
    tokens::Union{Nothing,Vector{String}}=nothing,
    kwargs...)

    # Validate
    if !all(m -> m <: AssociationMetric, metrics)
        invalid = filter(m -> !(m <: AssociationMetric), metrics)
        throw(ArgumentError("Invalid metric types: $invalid"))
    end

    df = assoc_df(x)

    if scores_only
        scores_dict = Dict{String,Vector{Float64}}()
        for T in metrics
            scores_dict[String(nameof(T))] = assoc_score(T, x; scores_only=true, tokens=tokens, kwargs...)
        end
        return scores_dict
    else
        if isempty(df)
            return DataFrame()
        end

        out = DataFrame()
        out[!, :Node] = fill(assoc_node(x), nrow(df))
        out[!, :Collocate] = String.(df.Collocate)
        out[!, :Frequency] = df.a

        for T in metrics
            out[!, Symbol(nameof(T))] = assoc_score(T, x; scores_only=true, tokens, kwargs...)
        end
        return out
    end
end

# =====================================
# Convenience functions
# =====================================

# Convenience for `Vector{DataType}` if you keep that style elsewhere
function assoc_score(metrics::Vector{DataType},
    x::AssociationDataFormat;
    scores_only::Bool=false,
    tokens::Union{Nothing,Vector{String}}=nothing,
    kwargs...)
    return assoc_score(Vector{Type{<:AssociationMetric}}(metrics), x;
        scores_only, tokens, kwargs...)
end

# ----------------------------
# Convenience overloads from raw text (kept for parity with your earlier API)
# These build a ContingencyTable and then call the unified API.
# ----------------------------

"""
    assoc_score(metricType::Type{<:AssociationMetric},
              inputstring::AbstractString,
              node::AbstractString;
              windowsize::Int,
              minfreq::Int=5;
              scores_only::Bool=false,
              norm_config::TextNorm=TextNorm(),
              tokens::Union{Nothing,Vector{String}}=nothing,
              kwargs...)

Convenience overload to compute a metric directly from a raw string.
"""
function assoc_score(::Type{T},
    inputstring::AbstractString,
    node::AbstractString;
    windowsize::Int,
    minfreq::Int=5,
    scores_only::Bool=false,
    norm_config::TextNorm=TextNorm(),
    tokens::Union{Nothing,Vector{String}}=nothing,
    kwargs...) where {T<:AssociationMetric}

    ct = ContingencyTable(inputstring, node; windowsize, minfreq,
        norm_config)

    return assoc_score(T, ct; scores_only, tokens, kwargs...)
end

"""
    assoc_score(metrics::AbstractVector{<:Type{<:AssociationMetric}},
              inputstring::AbstractString,
              node::AbstractString;
              windowsize::Int,
              minfreq::Int=5;
              scores_only::Bool=false,
              norm_config::TextNorm=TextNorm(),
              tokens::Union{Nothing,Vector{String}}=nothing,
              kwargs...)

Convenience overload to compute multiple metrics directly from raw text.
"""
function assoc_score(metrics::AbstractVector{<:Type{<:AssociationMetric}},
    inputstring::AbstractString,
    node::AbstractString;
    windowsize::Int,
    minfreq::Int=5,
    scores_only::Bool=false,
    norm_config::TextNorm=TextNorm(),
    tokens::Union{Nothing,Vector{String}}=nothing,
    kwargs...)

    ct = ContingencyTable(inputstring, node; windowsize, minfreq,
        norm_config)

    return assoc_score(metrics, ct; scores_only, tokens, kwargs...)
end

# Keep compatibility with Vector{DataType} call sites
function assoc_score(metrics::Vector{DataType},
    inputstring::AbstractString,
    node::AbstractString;
    windowsize::Int,
    minfreq::Int=5,
    scores_only::Bool=false,
    tokens::Union{Nothing,Vector{String}}=nothing,
    kwargs...)

    return assoc_score(Vector{Type{<:AssociationMetric}}(metrics),
        inputstring, node; windowsize, minfreq,
        scores_only, tokens, kwargs...)
end


"""
    assoc_score(metricType::Type{<:AssociationMetric}, corpus::Corpus, node::AbstractString;
                windowsize::Int=5, minfreq::Int=5, kwargs...)

Evaluate a metric on a corpus - convenience method that delegates to analyze_corpus.
"""
function assoc_score(::Type{T}, corpus::Corpus, node::AbstractString;
    windowsize::Int=5,
    minfreq::Int=5,
    kwargs...) where {T<:AssociationMetric}

    # Create corpus contingency table
    cct = CorpusContingencyTable(corpus, node; windowsize, minfreq)

    # Call the unified assoc_score with the CCT
    return assoc_score(T, cct; kwargs...)
end

# Also support multiple metrics
function assoc_score(metrics::AbstractVector{<:Type{<:AssociationMetric}},
    corpus::Corpus, node::AbstractString;
    windowsize::Int=5,
    minfreq::Int=5,
    kwargs...)

    # Create corpus contingency table
    cct = CorpusContingencyTable(corpus, node, windowsize, minfreq)

    # Call the unified assoc_score with the CCT
    return assoc_score(metrics, cct; kwargs...)
end

"""
    assoc_score(metricType::Type{<:AssociationMetric}, corpus::Corpus, 
                nodes::Vector{String}; windowsize::Int=5, minfreq::Int=5, 
                top_n::Int=100, kwargs...)

Evaluate a metric on multiple nodes in a corpus.
Returns a Dict{String,DataFrame} with results for each node.
"""
function assoc_score(::Type{T}, corpus::Corpus, nodes::Vector{String};
    windowsize::Int=5,
    minfreq::Int=5,
    top_n::Int=100,
    kwargs...) where {T<:AssociationMetric}

    results = Dict{String,DataFrame}()

    @showprogress desc = "Processing nodes..." for node in nodes
        # Create corpus contingency table for this node
        cct = CorpusContingencyTable(corpus, node; windowsize, minfreq)

        # Get results for this node
        node_results = assoc_score(T, cct; kwargs...)

        # Apply top_n filtering if results exist
        if !isempty(node_results)
            # Sort by score and take top N
            score_col = Symbol(string(T))
            sort!(node_results, score_col, rev=true)
            node_results = first(node_results, min(top_n, nrow(node_results)))
        end

        # Store with normalized node as key
        results[cct.node] = node_results
    end

    return results
end

"""
    assoc_score(metrics::AbstractVector{<:Type{<:AssociationMetric}}, 
                corpus::Corpus, nodes::Vector{String}; 
                windowsize::Int=5, minfreq::Int=5, top_n::Int=100, kwargs...)

Evaluate multiple metrics on multiple nodes in a corpus.
Returns a Dict{String,DataFrame} with combined metric results for each node.
"""
function assoc_score(metrics::AbstractVector{<:Type{<:AssociationMetric}},
    corpus::Corpus, nodes::Vector{String};
    windowsize::Int=5,
    minfreq::Int=5,
    top_n::Int=100,
    kwargs...)

    results = Dict{String,DataFrame}()

    @showprogress desc = "Processing nodes..." for node in nodes
        # Create corpus contingency table for this node
        cct = CorpusContingencyTable(corpus, node; windowsize, minfreq)

        # Get results for all metrics
        node_results = assoc_score(metrics, cct; kwargs...)

        # Apply top_n filtering if results exist
        if !isempty(node_results)
            # Sort by first metric and take top N
            first_metric = Symbol(string(metrics[1]))
            sort!(node_results, first_metric, rev=true)
            node_results = first(node_results, min(top_n, nrow(node_results)))
        end

        # Store with normalized node as key
        results[cct.node] = node_results
    end

    return results
end

"""
    assoc_score(metric::Type{<:AssociationMetric}, corpus::Corpus;
                nodes::Vector{String}, windowsize::Int=5, minfreq::Int=5,
                top_n::Int=100, kwargs...)

Alternative syntax with nodes as keyword argument.
"""
function assoc_score(::Type{T}, corpus::Corpus;
    nodes::Vector{String},
    windowsize::Int=5,
    minfreq::Int=5,
    top_n::Int=100,
    kwargs...) where {T<:AssociationMetric}

    return assoc_score(T, corpus, nodes;
        windowsize=windowsize, minfreq=minfreq, top_n=top_n, kwargs...)
end