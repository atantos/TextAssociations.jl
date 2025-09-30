# =====================================
# File: src/utils/statistical.jl
# Statistical utilities
# =====================================
"""
Safe logarithm that handles zero and negative values.
"""
log_safe(x::Real) = log(max(x, eps()))
log2_safe(x::Real) = log2(max(x, eps()))

"""
    available_metrics() -> Vector{DataType}

Returns a list of all supported association metrics.
"""
available_metrics() = METRIC_TYPES

"""
    cached_data(z::LazyProcess{T,R}) -> R

Extract data from a LazyProcess, computing it if necessary.
"""
function cached_data(z::LazyProcess{T,R}) where {T,R}
    if !z.cached_process
        z.cached_result = z.f()
        z.cached_process = true
    end
    return z.cached_result::R
end

"""
    document(input::LazyInput) -> StringDocument

Extract the document from a LazyInput wrapper.
"""
function document(input::LazyInput)
    return cached_data(input.loader)
end