# =====================================
# File: src/api.jl
# Public API functions
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
function evalassoc(metricType::Type{<:AssociationMetric}, data::ContingencyTable)
    func_name = Symbol("eval_", lowercase(string(metricType)))

    if !isdefined(@__MODULE__, func_name)
        throw(ArgumentError("Unknown metric type: $metricType"))
    end

    func = getfield(@__MODULE__, func_name)
    return func(data)
end

"""
    evalassoc(metricType::Type{<:AssociationMetric}, 
              inputstring::AbstractString,
              node::AbstractString, 
              windowsize::Int, 
              minfreq::Int=5)

Convenience method to compute metrics directly from text.
"""
function evalassoc(metricType::Type{<:AssociationMetric},
    inputstring::AbstractString,
    node::AbstractString,
    windowsize::Int,
    minfreq::Int=5)
    cont_table = ContingencyTable(inputstring, node, windowsize, minfreq)
    return evalassoc(metricType, cont_table)
end

"""
    evalassoc(metrics::Vector{DataType}, data::ContingencyTable)

Evaluate multiple metrics on a contingency table.

# Returns
DataFrame with one column per metric.
"""
function evalassoc(metrics::Vector{DataType}, data::ContingencyTable)
    # Validate metrics
    if !all(m -> m <: AssociationMetric, metrics)
        invalid = filter(m -> !(m <: AssociationMetric), metrics)
        throw(ArgumentError("Invalid metric types: $invalid"))
    end

    results = DataFrame()
    for metric in metrics
        func_name = Symbol("eval_", lowercase(string(metric)))

        if !isdefined(@__MODULE__, func_name)
            @warn "Skipping unknown metric: $metric"
            continue
        end

        func = getfield(@__MODULE__, func_name)
        results[!, string(metric)] = func(data)
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
"""
function evalassoc(metrics::Vector{DataType},
    inputstring::AbstractString,
    node::AbstractString,
    windowsize::Int,
    minfreq::Int=5)
    cont_table = ContingencyTable(inputstring, node, windowsize, minfreq)
    return evalassoc(metrics, cont_table)
end