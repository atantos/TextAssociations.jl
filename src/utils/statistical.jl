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
    listmetrics() -> Vector{Symbol}

Returns a list of all supported association metrics.
"""
listmetrics() = METRIC_TYPES

"""
    extract_cached_data(z::LazyProcess{T,R}) -> R

Extract data from a LazyProcess, computing it if necessary.
"""
function extract_cached_data(z::LazyProcess{T,R}) where {T,R}
    if !z.cached_process
        z.cached_result = z.f()
        z.cached_process = true
    end
    return z.cached_result::R
end

"""
    extract_document(input::LazyInput) -> StringDocument

Extract the document from a LazyInput wrapper.
"""
function extract_document(input::LazyInput)
    return extract_cached_data(input.loader)
end