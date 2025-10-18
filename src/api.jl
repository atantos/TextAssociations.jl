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

# Adding a guardrail layer that makes assoc_score, analyze_node and token-requiring metrics behave deterministically and non-crashing

const _STATUS_OK = "ok"
const _STATUS_EMPTY = "empty"
const _STATUS_ERROR = "error"

# ------------------------------------------------------------
# Numeric promotion helper (one-time float promotion of counts)
# ------------------------------------------------------------
"""
    promote_counts(x::AssociationDataFormat, ::Type{T}=Float64) -> AssociationDataFormat

Return a copy of `x` with numeric contingency arrays promoted to floating type `T`
(default `Float64`). Non-numeric fields are preserved unchanged.
"""
function promote_counts(x::AssociationDataFormat, ::Type{T}=Float64) where {T<:AbstractFloat}
    # Replace only the standard count fields; keep everything else as-is.
    # This assumes the struct has fields a,b,c,d,m,k,N; adapt if yours differ.
    names_to_replace = (:a, :b, :c, :d, :m, :k, :N)
    pairs = Pair{Symbol,Any}[]
    for name in fieldnames(typeof(x))
        if name in names_to_replace
            push!(pairs, name => T.(getfield(x, name)))
        else
            push!(pairs, name => getfield(x, name))
        end
    end
    return AssociationDataFormat(; pairs...)
end

# Build the typed empty shell for a single metric
function _empty_result(x::AssociationDataFormat, ::Type{T};
    reason::AbstractString, error::Bool=false) where {T<:AssociationMetric}

    df = DataFrame(
        Node=String[],
        Collocate=String[],
        Frequency=Int[]
    )
    df[!, Symbol(nameof(T))] = Float64[]

    metadata!(df, "status", error ? _STATUS_ERROR : _STATUS_EMPTY, style=:note)
    metadata!(df, "message", reason, style=:note)
    metadata!(df, "node", assoc_node(x), style=:note)
    metadata!(df, "windowsize", assoc_ws(x), style=:note)
    return df
end

# Build the *typed* empty shell for multi-metric calls
function _empty_multi_result(x::AssociationDataFormat,
    metrics::AbstractVector{<:Type{<:AssociationMetric}};
    reason::AbstractString, error::Bool=false)

    df = DataFrame(
        Node=String[],
        Collocate=String[],
        Frequency=Int[]
    )
    for T in metrics
        df[!, Symbol(nameof(T))] = Float64[]
    end
    metadata!(df, "status", error ? _STATUS_ERROR : _STATUS_EMPTY, style=:note)
    metadata!(df, "message", reason, style=:note)
    metadata!(df, "node", assoc_node(x), style=:note)
    metadata!(df, "windowsize", assoc_ws(x), style=:note)
    metadata!(df, "metrics", join(string.(metrics), ", "), style=:note)
    return df
end

"""
    assoc_score(metricType::Type{T<:AssociationMetric}, x::AssociationDataFormat;
              scores_only::Bool=false,
              tokens::Union{Nothing,Vector{String}}=nothing,
              kwargs...) where {T<:AssociationMetric}

Evaluate a metric on any association data format (ContingencyTable or CorpusContingencyTable).

- If the metric requires tokens (e.g., LexicalGravity), pass `tokens=...` or
  implement `assoc_tokens(::YourType)` to supply them automatically.
- Returns a DataFrame by default: [:Node, :Collocate, :Frequency, :<MetricName>].
- If `scores_only=true`, returns only the scores Vector.
"""
function assoc_score(::Type{T}, x::AssociationDataFormat;
    scores_only::Bool=false,
    tokens::Union{Nothing,Vector{String}}=nothing,
    kwargs...) where {T<:AssociationMetric}

    # Promote numeric counts once, then delegate to the choke-point helper
    x_f = promote_counts(x, Float64)
    return _assoc_score_promoted(T, x_f; scores_only=scores_only, tokens=tokens, kwargs...)
end

# --------------------------------------------------------------------
# Internal choke point that assumes counts are already promoted to Float
# (This is the old assoc_score body, lifted unchanged into a helper.)
# --------------------------------------------------------------------
function _assoc_score_promoted(::Type{T}, x::AssociationDataFormat;
    scores_only::Bool=false,
    tokens::Union{Nothing,Vector{String}}=nothing,
    kwargs...) where {T<:AssociationMetric}

    f = _resolve_metric_function(T)
    needs_tok = NeedsTokens(T)
    df_in = assoc_df(x)

    # If there is literally no contingency data, return a typed empty result
    if isempty(df_in)
        present = assoc_node_present(x)
        reason = present === false ?
                 "Node not found in the corpus." :
                 "Node found, but no collocates met the thresholds (windowsize/minfreq/filters)."
        return scores_only ? Float64[] :
               _empty_result(x, T; reason=reason)
    end

    if needs_tok === Val(true)
        toks = tokens === nothing ? assoc_tokens(x) : tokens
        if toks === nothing
            return scores_only ? Float64[] :
                   _empty_result(x, T; reason="Metric $(T) requires tokens but none were provided or available via assoc_tokens().")
        end
        scores = try
            f(x; tokens=toks, kwargs...)
        catch err
            return scores_only ? Float64[] :
                   _empty_result(x, T; reason="Metric $(T) failed during evaluation: $(err)", error=true)
        end
        return scores_only ? scores : begin
            out = DataFrame(Node=fill(assoc_node(x), nrow(df_in)),
                Collocate=String.(df_in.Collocate),
                Frequency=df_in.a)
            out[!, Symbol(nameof(T))] = scores
            metadata!(out, "status", _STATUS_OK, style=:note)
            out
        end
    else
        scores = try
            f(x; kwargs...)
        catch err
            return scores_only ? Float64[] :
                   _empty_result(x, T; reason="Metric $(T) failed during evaluation: $(err)", error=true)
        end
        return scores_only ? scores : begin
            out = DataFrame(Node=fill(assoc_node(x), nrow(df_in)),
                Collocate=String.(df_in.Collocate),
                Frequency=df_in.a)
            out[!, Symbol(nameof(T))] = scores
            metadata!(out, "status", _STATUS_OK, style=:note)
            out
        end
    end
end

"""
    assoc_score(metrics::AbstractVector{<:Type{<:AssociationMetric}},
              x::AssociationDataFormat;
              scores_only::Bool=false,
              tokens::Union{Nothing,Vector{String}}=nothing,
              kwargs...)

Evaluate multiple metrics on ContingencyTable or CorpusContingencyTable.

- Returns a DataFrame with one column per metric by default.
- If `scores_only=true`, returns Dict{String,Vector{Float64}}.
"""
function assoc_score(metrics::AbstractVector{<:Type{<:AssociationMetric}},
    x::AssociationDataFormat;
    scores_only::Bool=false,
    tokens::Union{Nothing,Vector{String}}=nothing,
    kwargs...)

    if !all(m -> m <: AssociationMetric, metrics)
        invalid = filter(m -> !(m <: AssociationMetric), metrics)
        throw(ArgumentError("Invalid metric types: $invalid"))
    end

    df_in = assoc_df(x)
    if scores_only
        # Return Dict, empty if no rows
        return isempty(df_in) ? Dict{String,Vector{Float64}}() :
        begin
            x_f = promote_counts(x, Float64)  # promote once for all metrics
            Dict(String(nameof(T)) => _assoc_score_promoted(T, x_f; scores_only=true, tokens=tokens, kwargs...)
                 for T in metrics)
        end
    else
        if isempty(df_in)
            present = assoc_node_present(x)
            reason = present === false ?
                     "Node not found in the corpus." :
                     "Node found, but no collocates met the thresholds (windowsize/minfreq/filters)."
            return _empty_multi_result(x, metrics; reason=reason)
        end

        x_f = promote_counts(x, Float64)  # promote once

        out = DataFrame(
            Node=fill(assoc_node(x), nrow(df_in)),
            Collocate=String.(df_in.Collocate),
            Frequency=df_in.a,
        )

        for T in metrics
            # Use the promoted choke point to avoid repeated promotion
            out[!, Symbol(nameof(T))] = _assoc_score_promoted(T, x_f; scores_only=true, tokens=tokens, kwargs...)
        end
        metadata!(out, "status", _STATUS_OK, style=:note)
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
    assoc_score(::Type{T},
                inputstring::AbstractString,
                node::AbstractString;
                windowsize::Int,
                minfreq::Int=5,
                scores_only::Bool=false,
                norm_config::TextNorm=TextNorm(),
                tokens::Union{Nothing,Vector{String}}=nothing,
                kwargs...) where {T<:AssociationMetric}

Compute an association metric `T` directly from a raw string. This is a
convenience overload that (1) builds a `ContingencyTable` from
`inputstring` around `node` with the given window/threshold and
normalization settings, and (2) delegates to
`assoc_score(::Type{T}, ::AssociationDataFormat; ...)`.

# Positional arguments
- `::Type{T}`: The metric type to evaluate (e.g. `PMI`, `LLR`, `LogDice`), where `T <: AssociationMetric`.
- `inputstring`: Raw text to analyze.
- `node`: The target token whose collocates will be scored.

# Keywords
- `windowsize::Int`: Symmetric context window size (in tokens) used to
  collect co-occurrences.
- `minfreq::Int=5`: Minimum co-occurrence frequency required for a
  collocate to be kept.
- `scores_only::Bool=false`: If `true`, return only the vector of scores
  (in the same order as the returned collocates for the tabular form);
  otherwise return a `DataFrame`.
- `norm_config::TextNorm=TextNorm()`: Text normalization configuration
  applied when constructing the contingency table (e.g., casefolding,
  punctuation handling, accent normalization).
- `tokens::Union{Nothing,Vector{String}}=nothing`: Optional pre-tokenized
  sequence for metrics that require token access. If `nothing`, metrics
  that need tokens will internally derive them; metrics that do not need
  tokens ignore this.
- `kwargs...`: Additional metric-specific keywords forwarded to the
  metric evaluator (e.g., smoothing, priors).

# Returns
- If `scores_only == false` (default): a `DataFrame` with one row per
  collocate, typically including columns like `:Node`, `:Collocate`,
  frequency/count fields (e.g., `:Freq`, `:DocFrequency`), and a column
  named after the metric (e.g., `:PMI`, `:LLR`).
- If `scores_only == true`: a `Vector{Float64}` of scores aligned to the
  collocates that would appear in the `DataFrame` form.

# Notes
- This method constructs a `ContingencyTable(inputstring, node; windowsize, minfreq, norm_config)`
  and then calls `assoc_score(T, ct; scores_only, tokens, kwargs...)`.
- If the text is too small or filters are strict (`minfreq` high, small
  `windowsize`), the result may be empty. Consider lowering `minfreq`,
  increasing `windowsize`, or providing more text.
- The `node` is interpreted after normalization as specified by
  `norm_config`. Ensure `node` matches the normalized form you expect.
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

Evaluate a metric on a corpus - convenience method that delegates to analyze_node.
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
    cct = CorpusContingencyTable(corpus, node; windowsize, minfreq)

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