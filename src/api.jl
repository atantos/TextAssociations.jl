# # =====================================
# # File: src/api.jl
# # Public API functions - UPDATED WITH NODE COLUMN
# # =====================================

# """
#     evalassoc(metricType::Type{<:AssociationMetric}, data::ContingencyTable)

# Evaluate an association metric on a contingency table.

# # Arguments
# - `metricType`: The type of metric to compute
# - `data`: The contingency table

# # Returns
# Vector of association scores for each collocate.

# # Examples
# ```julia
# ct = ContingencyTable("text", "word", 5, 2)
# scores = evalassoc(PMI, ct)
# ```
# """
# function evalassoc(::Type{T}, ct::ContingencyTable) where {T<:AssociationMetric}
#     fname = Symbol("eval_", lowercase(String(nameof(T))))
#     if !isdefined(@__MODULE__, fname)
#         throw(ArgumentError("Unknown metric type: $(T). Expected a function `$fname`."))
#     end
#     func = getfield(@__MODULE__, fname)
#     return func(ct)
# end

# """
#     evalassoc(metricType::Type{<:AssociationMetric}, 
#               inputstring::AbstractString,
#               node::AbstractString, 
#               windowsize::Int, 
#               minfreq::Int=5)

# Convenience method to compute metrics directly from text.
# """
# function evalassoc(::Type{T},
#     inputstring::AbstractString,
#     node::AbstractString,
#     windowsize::Int,
#     minfreq::Int=5) where {T<:AssociationMetric}
#     ct = ContingencyTable(inputstring, node, windowsize, minfreq)
#     return evalassoc(T, ct)
# end

# """
#     evalassoc(metrics::Vector{DataType}, data::ContingencyTable)

# Evaluate multiple metrics on a contingency table.

# # Returns
# DataFrame with Node column and one column per metric.
# """
# function evalassoc(metrics::Vector{DataType}, data::ContingencyTable)
#     # Validate metrics
#     if !all(m -> m <: AssociationMetric, metrics)
#         invalid = filter(m -> !(m <: AssociationMetric), metrics)
#         throw(ArgumentError("Invalid metric types: $invalid"))
#     end

#     results = DataFrame()

#     # Get the contingency table data to get collocates
#     con_tbl = extract_cached_data(data.con_tbl)

#     if !isempty(con_tbl)
#         # Add Node column first
#         results[!, :Node] = fill(data.node, nrow(con_tbl))

#         # Add Collocate column
#         results[!, :Collocate] = con_tbl.Collocate

#         # Add Frequency column for reference
#         results[!, :Frequency] = con_tbl.a

#         # Add metric scores
#         for T in metrics
#             fname = Symbol("eval_", lowercase(String(nameof(T))))
#             if !isdefined(@__MODULE__, fname)
#                 @warn "Skipping unknown metric" metric = T expected = fname
#                 continue
#             end
#             func = getfield(@__MODULE__, fname)
#             results[!, String(nameof(T))] = func(data)
#         end
#     end

#     return results
# end

# """
#     evalassoc(metrics::Vector{DataType},
#               inputstring::AbstractString,
#               node::AbstractString,
#               windowsize::Int,
#               minfreq::Int=5)

# Convenience method to compute multiple metrics directly from text.
# Returns DataFrame with Node column and metric scores.
# """
# function evalassoc(metrics::Vector{DataType},
#     inputstring::AbstractString,
#     node::AbstractString,
#     windowsize::Int,
#     minfreq::Int=5)
#     cont_table = ContingencyTable(inputstring, node, windowsize, minfreq)
#     return evalassoc(metrics, cont_table)
# end

# # Support for AbstractVector types
# function evalassoc(metrics::AbstractVector{<:Type{<:AssociationMetric}}, data::ContingencyTable)
#     return evalassoc(Vector{DataType}(metrics), data)
# end

# function evalassoc(metrics::AbstractVector{<:Type{<:AssociationMetric}},
#     inputstring::AbstractString, node::AbstractString, windowsize::Int, minfreq::Int=5)
#     return evalassoc(Vector{DataType}(metrics), inputstring, node, windowsize, minfreq)
# end

# """
#     evalassoc_with_node(metricType::Type{<:AssociationMetric}, data::ContingencyTable)

# Evaluate a single metric and return DataFrame with Node column.
# This is a convenience function for when you want a DataFrame output even with a single metric.
# """
# function evalassoc_with_node(::Type{T}, data::ContingencyTable) where {T<:AssociationMetric}
#     fname = Symbol("eval_", lowercase(String(nameof(T))))
#     if !isdefined(@__MODULE__, fname)
#         throw(ArgumentError("Unknown metric type: $(T). Expected a function `$fname`."))
#     end

#     func = getfield(@__MODULE__, fname)
#     scores = func(data)

#     # Get the contingency table data to get collocates
#     con_tbl = extract_cached_data(data.con_tbl)

#     if !isempty(con_tbl)
#         result = DataFrame(
#             Node=fill(data.node, length(scores)),
#             Collocate=con_tbl.Collocate,
#             Frequency=con_tbl.a,
#             Score=scores
#         )
#         # Rename the Score column to the metric name
#         rename!(result, :Score => Symbol(nameof(T)))
#         return result
#     else
#         return DataFrame()
#     end
# end

# """
#     evalassoc_with_node(metricType::Type{<:AssociationMetric},
#                        inputstring::AbstractString,
#                        node::AbstractString,
#                        windowsize::Int,
#                        minfreq::Int=5)

# Convenience method to compute a single metric directly from text and return DataFrame with Node column.
# """
# function evalassoc_with_node(::Type{T},
#     inputstring::AbstractString,
#     node::AbstractString,
#     windowsize::Int,
#     minfreq::Int=5) where {T<:AssociationMetric}
#     ct = ContingencyTable(inputstring, node, windowsize, minfreq)
#     return evalassoc_with_node(T, ct)
# end
# =====================================
# File: src/api.jl
# Public API functions - UPDATED to return DataFrame by default
# =====================================

"""
    evalassoc(metricType::Type{<:AssociationMetric}, data::ContingencyTable; 
              scores_only::Bool=false)

Evaluate an association metric on a contingency table.

# Arguments
- `metricType`: The type of metric to compute
- `data`: The contingency table
- `scores_only`: If true, return only the scores vector (for performance). 
                 Default false (returns DataFrame with words)

# Returns
- By default: DataFrame with columns [Node, Collocate, Frequency, MetricName]
- If `scores_only=true`: Vector of association scores
"""
# function evalassoc(::Type{T}, ct::ContingencyTable;
#     scores_only::Bool=false) where {T<:AssociationMetric}
#     # Calculate scores
#     fname = Symbol("eval_", lowercase(String(nameof(T))))
#     if !isdefined(@__MODULE__, fname)
#         throw(ArgumentError("Unknown metric type: $(T). Expected a function `$fname`."))
#     end
#     func = getfield(@__MODULE__, fname)
#     scores = func(ct)

#     # Return based on preference
#     if scores_only
#         return scores
#     else
#         # Default: Return DataFrame with words
#         con_tbl = extract_cached_data(ct.con_tbl)

#         if isempty(scores)
#             return DataFrame()
#         end

#         # Create DataFrame with proper constructor syntax
#         df = DataFrame()
#         df[!, :Node] = fill(ct.node, length(scores))
#         df[!, :Collocate] = String.(con_tbl.Collocate)
#         df[!, :Frequency] = con_tbl.a
#         df[!, Symbol(nameof(T))] = scores

#         return df
#     end
# end

# """
#     evalassoc(metricType::Type{<:AssociationMetric}, 
#               inputstring::AbstractString,
#               node::AbstractString, 
#               windowsize::Int, 
#               minfreq::Int=5;
#               scores_only::Bool=false)

# Convenience method to compute metrics directly from text.
# By default returns DataFrame with words.
# """
# function evalassoc(::Type{T},
#     inputstring::AbstractString,
#     node::AbstractString,
#     windowsize::Int,
#     minfreq::Int=5;
#     scores_only::Bool=false) where {T<:AssociationMetric}
#     ct = ContingencyTable(inputstring, node, windowsize, minfreq)
#     return evalassoc(T, ct; scores_only=scores_only)
# end

# """
#     evalassoc(metrics::Vector{DataType}, data::ContingencyTable;
#               scores_only::Bool=false)

# Evaluate multiple metrics on a contingency table.

# # Returns
# - By default: DataFrame with Node, Collocate, Frequency, and one column per metric
# - If `scores_only=true`: Dict{String, Vector} with metric names as keys
# """
# function evalassoc(metrics::Vector{DataType}, data::ContingencyTable;
#     scores_only::Bool=false)
#     # Validate metrics
#     if !all(m -> m <: AssociationMetric, metrics)
#         invalid = filter(m -> !(m <: AssociationMetric), metrics)
#         throw(ArgumentError("Invalid metric types: $invalid"))
#     end

#     if scores_only
#         # Return Dict of score vectors for performance
#         scores_dict = Dict{String,Vector{Float64}}()
#         for T in metrics
#             fname = Symbol("eval_", lowercase(String(nameof(T))))
#             if !isdefined(@__MODULE__, fname)
#                 @warn "Skipping unknown metric" metric = T expected = fname
#                 continue
#             end
#             func = getfield(@__MODULE__, fname)
#             scores_dict[String(nameof(T))] = func(data)
#         end
#         return scores_dict
#     else
#         # Default: Return DataFrame
#         con_tbl = extract_cached_data(data.con_tbl)

#         if isempty(con_tbl)
#             return DataFrame()
#         end

#         results = DataFrame()

#         # Add identifying columns
#         results[!, :Node] = fill(data.node, nrow(con_tbl))
#         results[!, :Collocate] = String.(con_tbl.Collocate)
#         results[!, :Frequency] = con_tbl.a

#         # Add metric scores
#         for T in metrics
#             fname = Symbol("eval_", lowercase(String(nameof(T))))
#             if !isdefined(@__MODULE__, fname)
#                 @warn "Skipping unknown metric" metric = T expected = fname
#                 continue
#             end
#             func = getfield(@__MODULE__, fname)
#             results[!, Symbol(nameof(T))] = func(data)
#         end

#         return results
#     end
# end

# """
#     evalassoc(metrics::Vector{DataType},
#               inputstring::AbstractString,
#               node::AbstractString,
#               windowsize::Int,
#               minfreq::Int=5;
#               scores_only::Bool=false)

# Convenience method to compute multiple metrics directly from text.
# By default returns DataFrame with words.
# """
# function evalassoc(metrics::Vector{DataType},
#     inputstring::AbstractString,
#     node::AbstractString,
#     windowsize::Int,
#     minfreq::Int=5;
#     scores_only::Bool=false)
#     cont_table = ContingencyTable(inputstring, node, windowsize, minfreq)
#     return evalassoc(metrics, cont_table; scores_only=scores_only)
# end

# # Support for AbstractVector types
# function evalassoc(metrics::AbstractVector{<:Type{<:AssociationMetric}},
#     data::ContingencyTable;
#     scores_only::Bool=false)
#     return evalassoc(Vector{DataType}(metrics), data; scores_only=scores_only)
# end

# function evalassoc(metrics::AbstractVector{<:Type{<:AssociationMetric}},
#     inputstring::AbstractString,
#     node::AbstractString,
#     windowsize::Int,
#     minfreq::Int=5;
#     scores_only::Bool=false)
#     return evalassoc(Vector{DataType}(metrics), inputstring, node, windowsize, minfreq;
#         scores_only=scores_only)
# end
# # =====================================
# # CorpusContingencyTable overloads (no CT bridge)
# # =====================================

# """
#     evalassoc(metricType::Type{<:AssociationMetric}, cct::CorpusContingencyTable;
#               scores_only::Bool=false, kwargs...)

# Evaluate a metric directly on a CorpusContingencyTable.
# `kwargs...` are forwarded to the metric evaluator (e.g., tokens/corpus for LG).
# """
# function evalassoc(::Type{T}, cct::CorpusContingencyTable;
#     scores_only::Bool=false, kwargs...) where {T<:AssociationMetric}

#     fname = Symbol("eval_", lowercase(String(nameof(T))))
#     if !isdefined(@__MODULE__, fname)
#         throw(ArgumentError("Unknown metric type: $(T). Expected a function `$fname`."))
#     end
#     func = getfield(@__MODULE__, fname)

#     # Compute scores directly from CCT
#     scores = func(cct; kwargs...)

#     if scores_only
#         return scores
#     else
#         con_tbl = extract_cached_data(cct.aggregated_table)
#         if isempty(con_tbl)
#             return DataFrame()
#         end
#         df = DataFrame()
#         df[!, :Node] = fill(cct.node, nrow(con_tbl))
#         df[!, :Collocate] = String.(con_tbl.Collocate)
#         df[!, :Frequency] = con_tbl.a
#         df[!, Symbol(nameof(T))] = scores
#         return df
#     end
# end

# """
#     evalassoc(metrics::Vector{DataType}, cct::CorpusContingencyTable;
#               scores_only::Bool=false, kwargs...)

# Evaluate multiple metrics directly on a CorpusContingencyTable.
# """
# function evalassoc(metrics::Vector{DataType}, cct::CorpusContingencyTable;
#     scores_only::Bool=false, kwargs...)

#     if !all(m -> m <: AssociationMetric, metrics)
#         invalid = filter(m -> !(m <: AssociationMetric), metrics)
#         throw(ArgumentError("Invalid metric types: $invalid"))
#     end

#     con_tbl = extract_cached_data(cct.aggregated_table)
#     if scores_only
#         scores_dict = Dict{String,Vector{Float64}}()
#         for T in metrics
#             fname = Symbol("eval_", lowercase(String(nameof(T))))
#             if !isdefined(@__MODULE__, fname)
#                 @warn "Skipping unknown metric" metric = T expected = fname
#                 continue
#             end
#             func = getfield(@__MODULE__, fname)
#             scores_dict[String(nameof(T))] = func(cct; kwargs...)
#         end
#         return scores_dict
#     else
#         if isempty(con_tbl)
#             return DataFrame()
#         end
#         results = DataFrame()
#         results[!, :Node] = fill(cct.node, nrow(con_tbl))
#         results[!, :Collocate] = String.(con_tbl.Collocate)
#         results[!, :Frequency] = con_tbl.a

#         for T in metrics
#             fname = Symbol("eval_", lowercase(String(nameof(T))))
#             if !isdefined(@__MODULE__, fname)
#                 @warn "Skipping unknown metric" metric = T expected = fname
#                 continue
#             end
#             func = getfield(@__MODULE__, fname)
#             results[!, Symbol(nameof(T))] = func(cct; kwargs...)
#         end
#         return results
#     end
# end

# # Convenience for AbstractVector{<:Type{<:AssociationMetric}}
# function evalassoc(metrics::AbstractVector{<:Type{<:AssociationMetric}},
#     cct::CorpusContingencyTable; scores_only::Bool=false, kwargs...)
#     return evalassoc(Vector{DataType}(metrics), cct; scores_only=scores_only, kwargs...)
# end
# =====================================
# File: src/api.jl
# Public API (unified over AssociationDataFormat)
# =====================================

# Assumptions:
# - You already defined:
#     abstract type AssociationDataFormat end
#     struct ContingencyTable <: AssociationDataFormat
#     struct CorpusContingencyTable <: AssociationDataFormat
# - And you provide the following accessors (e.g., in metrics/_iface.jl):
#     assoc_df(x::AssociationDataFormat)::DataFrame
#     assoc_node(x::AssociationDataFormat)::String
#     assoc_ws(x::AssociationDataFormat)::Int
#     assoc_tokens(x::AssociationDataFormat)::Union{Nothing,Vector{String}}  # default: nothing
#
# - Each metric implements a function `eval_<lowercase(metric)>` that accepts:
#     eval_pmi(x::AssociationDataFormat)                        # count-only example
#     eval_lexicalgravity(x::AssociationDataFormat; tokens=...) # token-needed example
#
# NOTE: Count-only metrics should *not* require tokens; token-based ones should accept `tokens` kwarg.

# ----------------------------
# Small trait: which metrics need tokens?
# Extend this in your metric files as needed (defaults to false).
# Example elsewhere: NeedsTokens(::Type{LexicalGravity}) = Val(true)
# ----------------------------
NeedsTokens(::Type{T}) where {T<:AssociationMetric} = Val(false)

# ----------------------------
# Internal: resolve metric entry point by naming convention
# ----------------------------
@inline function _resolve_metric_function(::Type{T}) where {T<:AssociationMetric}
    fname = Symbol("eval_", lowercase(String(nameof(T))))
    if !isdefined(@__MODULE__, fname)
        throw(ArgumentError("Unknown metric type $(T). Expected a function `$fname`."))
    end
    return getfield(@__MODULE__, fname)
end

# ----------------------------
# Core, single-metric API over AssociationDataFormat
# ----------------------------

"""
    evalassoc(metricType::Type{<:AssociationMetric}, x::AssociationDataFormat;
              scores_only::Bool=false,
              tokens::Union{Nothing,Vector{String}}=nothing,
              kwargs...)

Evaluate a metric on any association data format (CT or CCT).

- If the metric requires tokens (e.g., LexicalGravity), pass `tokens=...` or
  implement `assoc_tokens(::YourType)` to supply them automatically.
- Returns a DataFrame by default: [:Node, :Collocate, :Frequency, :<MetricName>].
- If `scores_only=true`, returns only the scores Vector.
"""
function evalassoc(::Type{T}, x::AssociationDataFormat;
    scores_only::Bool=false,
    tokens::Union{Nothing,Vector{String}}=nothing,
    kwargs...) where {T<:AssociationMetric}

    f = _resolve_metric_function(T)
    needs_tok = NeedsTokens(T)

    # Compute scores
    scores = if needs_tok === Val(true)
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
    # (Adjust if your column name differs.)
    out[!, :Frequency] = df.a
    out[!, Symbol(nameof(T))] = scores
    return out
end

# ----------------------------
# Multi-metric API over AssociationDataFormat
# ----------------------------

"""
    evalassoc(metrics::AbstractVector{<:Type{<:AssociationMetric}},
              x::AssociationDataFormat;
              scores_only::Bool=false,
              tokens::Union{Nothing,Vector{String}}=nothing,
              kwargs...)

Evaluate multiple metrics on CT or CCT.

- Returns a DataFrame with one column per metric by default.
- If `scores_only=true`, returns Dict{String,Vector{Float64}}.
"""
function evalassoc(metrics::AbstractVector{<:Type{<:AssociationMetric}},
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
            scores_dict[String(nameof(T))] = evalassoc(T, x; scores_only=true, tokens=tokens, kwargs...)
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
            out[!, Symbol(nameof(T))] = evalassoc(T, x; scores_only=true, tokens=tokens, kwargs...)
        end
        return out
    end
end

# Convenience for `Vector{DataType}` if you keep that style elsewhere
function evalassoc(metrics::Vector{DataType},
    x::AssociationDataFormat; scores_only::Bool=false,
    tokens::Union{Nothing,Vector{String}}=nothing,
    kwargs...)
    return evalassoc(Vector{Type{<:AssociationMetric}}(metrics), x;
        scores_only=scores_only, tokens=tokens, kwargs...)
end

# ----------------------------
# Convenience overloads from raw text (kept for parity with your earlier API)
# These build a ContingencyTable and then call the unified API.
# ----------------------------

"""
    evalassoc(metricType::Type{<:AssociationMetric},
              inputstring::AbstractString,
              node::AbstractString,
              windowsize::Int,
              minfreq::Int=5;
              scores_only::Bool=false,
              tokens::Union{Nothing,Vector{String}}=nothing,
              kwargs...)

Convenience overload to compute a metric directly from a raw string.
"""
function evalassoc(::Type{T},
    inputstring::AbstractString,
    node::AbstractString,
    windowsize::Int,
    minfreq::Int=5;
    scores_only::Bool=false,
    tokens::Union{Nothing,Vector{String}}=nothing,
    strip_accents::Bool=false,
    kwargs...) where {T<:AssociationMetric}
    ct = ContingencyTable(inputstring, node, windowsize, minfreq)
    return evalassoc(T, ct; scores_only=scores_only, tokens=tokens, strip_accents=strip_accents, kwargs...)
end

"""
    evalassoc(metrics::AbstractVector{<:Type{<:AssociationMetric}},
              inputstring::AbstractString,
              node::AbstractString,
              windowsize::Int,
              minfreq::Int=5;
              scores_only::Bool=false,
              tokens::Union{Nothing,Vector{String}}=nothing,
              kwargs...)

Convenience overload to compute multiple metrics directly from raw text.
"""
function evalassoc(metrics::AbstractVector{<:Type{<:AssociationMetric}},
    inputstring::AbstractString,
    node::AbstractString,
    windowsize::Int,
    minfreq::Int=5;
    scores_only::Bool=false,
    tokens::Union{Nothing,Vector{String}}=nothing,
    kwargs...)
    ct = ContingencyTable(inputstring, node, windowsize, minfreq)
    return evalassoc(metrics, ct; scores_only=scores_only, tokens=tokens, kwargs...)
end

# Keep compatibility with Vector{DataType} call sites
function evalassoc(metrics::Vector{DataType},
    inputstring::AbstractString,
    node::AbstractString,
    windowsize::Int,
    minfreq::Int=5;
    scores_only::Bool=false,
    tokens::Union{Nothing,Vector{String}}=nothing,
    kwargs...)
    return evalassoc(Vector{Type{<:AssociationMetric}}(metrics),
        inputstring, node, windowsize, minfreq;
        scores_only=scores_only, tokens=tokens, kwargs...)
end
