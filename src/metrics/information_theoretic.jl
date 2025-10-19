# =====================================
# File: src/metrics/information_theoretic.jl
# Information-theoretic metrics
# =====================================

# Pointwise Mutual Information
function eval_pmi(data::AssociationDataFormat)
    @extract_values data a N k m
    log_safe.(a) .- log_safe.(N) .- (log_safe.(k) .+ log_safe.(m) .- log_safe.(N))
end

# PMI²
function eval_pmi²(data::AssociationDataFormat)
    @extract_values data a N k m
    2 .* log_safe.(a) .- log_safe.(N) .- (log_safe.(k) .+ log_safe.(m) .- log_safe.(N))
end

# PMI³
function eval_pmi³(data::AssociationDataFormat)
    @extract_values data a N k m
    3 .* log_safe.(a) .- log_safe.(N) .- (log_safe.(k) .+ log_safe.(m) .- log_safe.(N))
end

# Positive PMI
function eval_ppmi(data::AssociationDataFormat)
    @extract_values data a N k m
    max.(0, log_safe.(a) .- log_safe.(N) .- (log_safe.(k) .+ log_safe.(m) .- log_safe.(N)))
end

# Log Likelihood Ratio
function eval_llr(data::AssociationDataFormat)
    @extract_values data a b c d E₁₁ E₁₂ E₂₁ E₂₂

    observed = a .* log_safe.(a) .+ b .* log_safe.(b) .+
               c .* log_safe.(c) .+ d .* log_safe.(d)
    expected = a .* log_safe.(E₁₁) .+ b .* log_safe.(E₁₂) .+
               c .* log_safe.(E₂₁) .+ d .* log_safe.(E₂₂)

    2 * (observed .- expected)
end

# Squared LLR
function eval_llr²(data::AssociationDataFormat)
    llr_values = eval_llr(data)
    llr_values .^ 2
end

# Log-Likelihood Ratio (G²) — Dunning (1993)
"""
    eval_g2(data::AssociationDataFormat)

Compute the Log-Likelihood Ratio statistic G² for 2×2 contingency tables.

# Arguments
- `data`: AssociationDataFormat with contingency counts.

# Definition
For observed counts `a,b,c,d` and expected counts `E₁₁,E₁₂,E₂₁,E₂₂` from the
marginals, the per-row statistic is:

G² = 2 · Σ Oᵢⱼ · ln(Oᵢⱼ / Eᵢⱼ)   (with convention 0·ln(0/E)=0)

Expected counts for a 2×2 come from row/column marginals:
- `m = a + b`, `n = c + d = N - m`
- `k = a + c`, `ℓ = b + d = N - k`
- `E₁₁ = m·k/N`, `E₁₂ = m·ℓ/N`, `E₂₁ = n·k/N`, `E₂₂ = n·ℓ/N`

# Notes
- Uses **natural log** (`ln`) as standard in G²; if you prefer log base 2, multiply by `2ln(2)` accordingly.
- Fully fused implementation (no temporaries), with **float promotion only where needed** to avoid integer overflow (e.g., `float(m) * k`).
- Guards:
  - `N` and each `Eᵢⱼ` are floored with `eps(Float64)` to prevent division by zero.
  - Terms with `Oᵢⱼ == 0` contribute 0 by convention.
- Returns `Vector{Float64}` aligned with `assoc_df(data)`.
"""
function eval_g2(data::AssociationDataFormat)
    @extract_values data a b c d m k N
    @. begin
        Nf = max(float(N), eps(Float64))
        mf = float(m)
        kf = float(k)
        n = Nf - mf
        ell = Nf - kf

        E11 = (mf * kf) / Nf
        E12 = (mf * ell) / Nf
        E21 = (n * kf) / Nf
        E22 = (n * ell) / Nf

        t11 = ifelse(a > 0, a * log(a / max(E11, eps(Float64))), 0.0)
        t12 = ifelse(b > 0, b * log(b / max(E12, eps(Float64))), 0.0)
        t21 = ifelse(c > 0, c * log(c / max(E21, eps(Float64))), 0.0)
        t22 = ifelse(d > 0, d * log(d / max(E22, eps(Float64))), 0.0)

        2.0 * (t11 + t12 + t21 + t22)
    end
end
