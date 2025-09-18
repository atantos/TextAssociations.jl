# =====================================
# File: src/api.jl
# Public API functions - UPDATED WITH NODE COLUMN
# =====================================

"""
    evalassoc(metricType::Type{<:AssociationMetric}, data::ContingencyTable)

Evaluate an association metric on a contingency table.

# Arguments
- `metricType`: The type of metric to compute
- `data`: The contingency table

# Returns
Vector of association scores for each collocate.

# Examples
```julia
ct = ContingencyTable("text", "word", 5, 2)
scores = evalassoc(PMI, ct)
```
"""
function evalassoc(::Type{T}, ct::ContingencyTable) where {T<:AssociationMetric}
    fname = Symbol("eval_", lowercase(String(nameof(T))))
    if !isdefined(@__MODULE__, fname)
        throw(ArgumentError("Unknown metric type: $(T). Expected a function `$fname`."))
    end
    func = getfield(@__MODULE__, fname)
    return func(ct)
end

"""
    evalassoc(metricType::Type{<:AssociationMetric}, 
              inputstring::AbstractString,
              node::AbstractString, 
              windowsize::Int, 
              minfreq::Int=5)

Convenience method to compute metrics directly from text.
"""
function evalassoc(::Type{T},
    inputstring::AbstractString,
    node::AbstractString,
    windowsize::Int,
    minfreq::Int=5) where {T<:AssociationMetric}
    ct = ContingencyTable(inputstring, node, windowsize, minfreq)
    return evalassoc(T, ct)
end

"""
    evalassoc(metrics::Vector{DataType}, data::ContingencyTable)

Evaluate multiple metrics on a contingency table.

# Returns
DataFrame with Node column and one column per metric.
"""
function evalassoc(metrics::Vector{DataType}, data::ContingencyTable)
    # Validate metrics
    if !all(m -> m <: AssociationMetric, metrics)
        invalid = filter(m -> !(m <: AssociationMetric), metrics)
        throw(ArgumentError("Invalid metric types: $invalid"))
    end

    results = DataFrame()

    # Get the contingency table data to get collocates
    con_tbl = extract_cached_data(data.con_tbl)

    if !isempty(con_tbl)
        # Add Node column first
        results[!, :Node] = fill(data.node, nrow(con_tbl))

        # Add Collocate column
        results[!, :Collocate] = con_tbl.Collocate

        # Add Frequency column for reference
        results[!, :Frequency] = con_tbl.a

        # Add metric scores
        for T in metrics
            fname = Symbol("eval_", lowercase(String(nameof(T))))
            if !isdefined(@__MODULE__, fname)
                @warn "Skipping unknown metric" metric = T expected = fname
                continue
            end
            func = getfield(@__MODULE__, fname)
            results[!, String(nameof(T))] = func(data)
        end
    end

    return results
end

"""
    evalassoc(metrics::Vector{DataType},
              inputstring::AbstractString,
              node::AbstractString,
              windowsize::Int,
              minfreq::Int=5)

Convenience method to compute multiple metrics directly from text.
Returns DataFrame with Node column and metric scores.
"""
function evalassoc(metrics::Vector{DataType},
    inputstring::AbstractString,
    node::AbstractString,
    windowsize::Int,
    minfreq::Int=5)
    cont_table = ContingencyTable(inputstring, node, windowsize, minfreq)
    return evalassoc(metrics, cont_table)
end

# Support for AbstractVector types
function evalassoc(metrics::AbstractVector{<:Type{<:AssociationMetric}}, data::ContingencyTable)
    return evalassoc(Vector{DataType}(metrics), data)
end

function evalassoc(metrics::AbstractVector{<:Type{<:AssociationMetric}},
    inputstring::AbstractString, node::AbstractString, windowsize::Int, minfreq::Int=5)
    return evalassoc(Vector{DataType}(metrics), inputstring, node, windowsize, minfreq)
end

"""
    evalassoc_with_node(metricType::Type{<:AssociationMetric}, data::ContingencyTable)

Evaluate a single metric and return DataFrame with Node column.
This is a convenience function for when you want a DataFrame output even with a single metric.
"""
function evalassoc_with_node(::Type{T}, data::ContingencyTable) where {T<:AssociationMetric}
    fname = Symbol("eval_", lowercase(String(nameof(T))))
    if !isdefined(@__MODULE__, fname)
        throw(ArgumentError("Unknown metric type: $(T). Expected a function `$fname`."))
    end

    func = getfield(@__MODULE__, fname)
    scores = func(data)

    # Get the contingency table data to get collocates
    con_tbl = extract_cached_data(data.con_tbl)

    if !isempty(con_tbl)
        result = DataFrame(
            Node=fill(data.node, length(scores)),
            Collocate=con_tbl.Collocate,
            Frequency=con_tbl.a,
            Score=scores
        )
        # Rename the Score column to the metric name
        rename!(result, :Score => Symbol(nameof(T)))
        return result
    else
        return DataFrame()
    end
end

"""
    evalassoc_with_node(metricType::Type{<:AssociationMetric},
                       inputstring::AbstractString,
                       node::AbstractString,
                       windowsize::Int,
                       minfreq::Int=5)

Convenience method to compute a single metric directly from text and return DataFrame with Node column.
"""
function evalassoc_with_node(::Type{T},
    inputstring::AbstractString,
    node::AbstractString,
    windowsize::Int,
    minfreq::Int=5) where {T<:AssociationMetric}
    ct = ContingencyTable(inputstring, node, windowsize, minfreq)
    return evalassoc_with_node(T, ct)
end