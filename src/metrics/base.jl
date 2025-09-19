# =====================================
# File: src/metrics/base.jl
# Base utilities for metrics
# =====================================
"""
    @extract_values data a b c d ...

Expands to:
- `df = assoc_df(data)`
- early-return `Float64[]` if `df` is empty
- local bindings `a = df.a`, `b = df.b`, ...

Works for any `AssociationDataFormat` subtype as long as
`assoc_df(x)` returns a DataFrame with these column names.
"""
macro extract_values(data, vars...)
    d = esc(data)

    assigns = [
        :(
            $(esc(v)) = hasproperty(df, $(QuoteNode(v))) ?
                        getproperty(df, $(QuoteNode(v))) :
                        df[!, $(QuoteNode(v))]
        ) for v in vars
    ]

    return quote
        df = assoc_df($d)
        isempty(df) && return Float64[]
        $(assigns...)
    end
end
