# =====================================
# File: src/metrics/information_theoretic.jl
# Information-theoretic metrics
# =====================================

# Pointwise Mutual Information
function eval_pmi(data::ContingencyTable)
    @extract_values data a N k m
    log_safe.(a) .- log_safe.(N) .- (log_safe.(k) .+ log_safe.(m) .- log_safe.(N))
end

# PMI²
function eval_pmi²(data::ContingencyTable)
    @extract_values data a N k m
    2 .* log_safe.(a) .- log_safe.(N) .- (log_safe.(k) .+ log_safe.(m) .- log_safe.(N))
end

# PMI³
function eval_pmi³(data::ContingencyTable)
    @extract_values data a N k m
    3 .* log_safe.(a) .- log_safe.(N) .- (log_safe.(k) .+ log_safe.(m) .- log_safe.(N))
end

# Positive PMI
function eval_ppmi(data::ContingencyTable)
    @extract_values data a N k m
    max.(0, log_safe.(a) .- log_safe.(N) .- (log_safe.(k) .+ log_safe.(m) .- log_safe.(N)))
end

# Log Likelihood Ratio
function eval_llr(data::ContingencyTable)
    @extract_values data a b c d E₁₁ E₁₂ E₂₁ E₂₂

    observed = a .* log_safe.(a) .+ b .* log_safe.(b) .+
               c .* log_safe.(c) .+ d .* log_safe.(d)
    expected = a .* log_safe.(E₁₁) .+ b .* log_safe.(E₁₂) .+
               c .* log_safe.(E₂₁) .+ d .* log_safe.(E₂₂)

    2 * (observed .- expected)
end

# Squared LLR
function eval_llr²(data::ContingencyTable)
    llr_values = eval_llr(data)
    llr_values .^ 2
end