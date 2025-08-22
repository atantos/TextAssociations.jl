# =====================================
# File: src/metrics/base.jl
# Base utilities for metrics
# =====================================
"""
Helper macro for extracting contingency table values.
Reduces boilerplate in metric implementations.
"""
# macro extract_values(data, vars...)
#     expr = quote
#         con_tbl = extract_cached_data($data.con_tbl)
#         isempty(con_tbl) && return Float64[]
#     end

#     for var in vars
#         push!(expr.args, :($(esc(var)) = con_tbl.$var))
#     end

#     return expr
# end
macro extract_values(data, vars...)
    # refer to the caller's `data`
    d = esc(data)

    # build assignments like: a = getproperty(con_tbl, :a)
    assigns = [:($(esc(v)) = getproperty(con_tbl, $(QuoteNode(v)))) for v in vars]

    return quote
        # create a local `con_tbl` in the caller's scope
        con_tbl = extract_cached_data($d.con_tbl)
        isempty(con_tbl) && return Float64[]
        $(assigns...)
    end
end

