# =====================================
# File: src/metrics/base.jl
# Base utilities for metrics
# =====================================
"""
Helper macro for extracting contingency table values.
Reduces boilerplate in metric implementations.
"""
macro extract_values(data, vars...)
    expr = quote
        con_tbl = extract_cached_data($data.con_tbl)
        isempty(con_tbl) && return Float64[]
    end

    for var in vars
        push!(expr.args, :($(esc(var)) = con_tbl.$var))
    end

    return expr
end
