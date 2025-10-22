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

############################
# Bayesian-smoothed PMI (symmetric Dirichlet smoothing)
############################

"""
    eval_bpmi(data::AssociationDataFormat; λ::Float64=0.5) -> Vector{Float64}

Bayesian PMI with symmetric Dirichlet(λ) smoothing on the 2×2 table.

Uses posterior-mean plug-in probabilities:

    p̂(x,y) = (a + λ) / (N + 4λ)
    p̂(x)   = (m + 2λ) / (N + 4λ)
    p̂(y)   = (k + 2λ) / (N + 4λ)

Returns log(p̂(x,y) / (p̂(x)p̂(y))) as a vector aligned to rows of `assoc_df(data)`.

`λ=0.5` (Jeffreys) is a sensible default for sparse lexical counts.
"""
function eval_bpmi(data::AssociationDataFormat; λ::Float64=0.5)
    df = assoc_df(data)
    isempty(df) && return Float64[]

    @inbounds begin
        a = df.a
        m = df.m
        k = df.k
        N = df.N

        # Smoothed components
        N_ = N .+ 4λ
        a_ = a .+ λ
        m_ = m .+ 2λ
        k_ = k .+ 2λ

        # BPMI = log( (a_/N_) / ((m_/N_)*(k_/N_)) ) = log( a_*N_ / (m_*k_) )
        return log.((a_ .* N_) ./ (m_ .* k_)) .|> log_safe
    end
end


############################
# Bayesian-smoothed LLR (plug-in, single λ)
############################

"""
    eval_bllr(data::AssociationDataFormat; λ::Float64=0.5) -> Vector{Float64}

Bayesian-smoothed variant of the Log-Likelihood Ratio (Dunning-style)
computed *on a table smoothed with symmetric Dirichlet(λ)*.

We form a smoothed 2×2:

    a' = a + λ,  b' = b + λ,  c' = c + λ,  d' = d + λ
    m' = a'+b',  n' = c'+d',  N' = m' + n'
    p'  = (a' + c') / N'       # pooled success prob under H0
    p1' = a' / m'              # success prob row 1 (H1)
    p2' = c' / n'              # success prob row 2 (H1)

Then classic LLR on smoothed counts:

    LLR' = 2 * [ a' log(a'/(m' p')) + b' log(b'/(m'(1-p'))) +
                 c' log(c'/(n' p')) + d' log(d'/(n'(1-p'))) ]

Returns a non-negative vector of doubles.
"""
function eval_bllr(data::AssociationDataFormat; λ::Float64=0.5)
    df = assoc_df(data)
    isempty(df) && return Float64[]

    @inbounds begin
        a = df.a
        b = df.b
        c = df.c
        d = df.d
        # Smoothed cells & margins
        ap = a .+ λ
        bp = b .+ λ
        cp = c .+ λ
        dp = d .+ λ

        mp = ap .+ bp
        np = cp .+ dp
        Np = mp .+ np

        # Smoothed probabilities
        pp = (ap .+ cp) ./ Np
        one_minus_pp = 1 .- pp
        p1p = ap ./ mp
        p2p = cp ./ np

        # Helper: x * log(x / (base)) with guards
        @inline term(x, base) = (x .> 0) .* (x .* (log.(max.(x ./ max.(base, eps()), eps()))))

        # H0 terms (pooled p')
        t1 = term(ap, mp .* pp)
        t2 = term(bp, mp .* one_minus_pp)
        t3 = term(cp, np .* pp)
        t4 = term(dp, np .* one_minus_pp)

        # H1 terms (separate p1', p2') — but in LLR we subtract H0 from H1;
        # the standard compact form with counts already incorporates both by using the same 'term' pattern.
        # Since we've written the H0 version explicitly, we can compute H1 by:
        u1 = term(ap, mp .* p1p)
        u2 = term(bp, mp .* (1 .- p1p))
        u3 = term(cp, np .* p2p)
        u4 = term(dp, np .* (1 .- p2p))

        # 2 * [ (H1 sum) - (H0 sum) ]
        llr = 2 .* ((u1 .+ u2 .+ u3 .+ u4) .- (t1 .+ t2 .+ t3 .+ t4))

        # numerical floor at zero (tiny negatives from fp error)
        return max.(llr, 0.0)
    end
end

"""
    eval_bayesllr(data::AssociationDataFormat; λ::Float64=0.5) -> Vector{Float64}

Pure Bayesian evidence for association in a 2×2 table via a Bayes factor
between H₁: different proportions vs H₀: equal proportion (independence),
using symmetric Beta(λ, λ) priors. Returns `2 * log(BF₁₀)`.

For counts (a,b;c,d):
    log BF10 = log B(a+λ, b+λ) + log B(c+λ, d+λ)
               - log B(a+c+λ, b+d+λ) - log B(λ, λ)
"""
function eval_bayesllr(data::AssociationDataFormat; λ::Float64=0.5)
    df = assoc_df(data)
    isempty(df) && return Float64[]

    @inbounds begin
        a = df.a
        b = df.b
        c = df.c
        d = df.d
        lb0 = logbeta(λ, λ)

        logbf = logbeta.(a .+ λ, b .+ λ) .+
                logbeta.(c .+ λ, d .+ λ) .-
                logbeta.(a .+ c .+ λ, b .+ d .+ λ) .- lb0

        return 2 .* logbf
    end
end
