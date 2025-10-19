# =====================================
# File: src/metrics/statistical.jl
# Statistical metrics
# =====================================

# Fisher’s Exact Test — right-tailed p-value
"""
    eval_fisherright(data::AssociationDataFormat)

Compute the right-tailed Fisher’s Exact Test p-value for each 2×2 table row.

# Arguments
- `data`: AssociationDataFormat with counts.

# Definition
Given a 2×2 table with row totals `m = a + b`, `n = c + d`,
column total `k = a + c`, and `N = a + b + c + d`, the hypergeometric
pmf for observing `x` co-occurrences is:

    pmf(x) = C(m, x) * C(n, k - x) / C(N, k)

The **right-tailed** p-value is:
    
    p_right = Σ_{x = a}^{min(m, k)} pmf(x)

# Notes
- We compute `pmf(a)` once via log-space (`logfactorial`), then sum the tail using
  the stable recurrence `p(x+1) = p(x) * r(x)`, where:
  
    r(x) = [(m - x)/(x + 1)] * [(k - x)/(n - (k - x))]

- Fully vectorized: a small scalar helper is dot-called over all rows (`@.` fused).
- Returns a `Vector{Float64}` with one p-value per row.
- For ranking as an association score, a common transform is `-log10(p_right)`.
"""
function eval_fisherright(data::AssociationDataFormat)
    @extract_values data a b c d m k N

    # Scalar helper for a single row (fast, stable, no extra allocations)
    @inline function _fisherright_one(a::Int, m::Int, k::Int, N::Int)::Float64
        # Derive n; basic sanity checks (return p=1.0 if row is structurally invalid)
        n = N - m
        if N < 0 || m < 0 || n < 0 || k < 0 || k > N
            return 1.0
        end
        # Valid range for a given marginals (clamp a into feasible hypergeometric support)
        a_min = max(0, k - n)
        a_max = min(m, k)
        if a_max < a_min
            return 1.0
        end
        a0 = clamp(a, a_min, a_max)

        # pmf at a0 in log-space: log C(m, a0) + log C(n, k-a0) - log C(N, k)
        @inline logchoose(n::Integer, r::Integer) = logfactorial(n) - logfactorial(r) - logfactorial(n - r)
        logp = logchoose(m, a0) + logchoose(n, k - a0) - logchoose(N, k)
        p = exp(logp)
        s = p

        # Sum tail upwards using recurrence:
        # p(x+1) = p(x) * ((m - x)/(x + 1)) * ((k - x)/(n - (k - x)))
        x = a0
        while x < a_max && p > 0.0
            r = ((m - x) / (x + 1)) * ((k - x) / (n - (k - x)))
            p *= r
            s += p
            x += 1
        end
        return min(s, 1.0)
    end

    # Vectorized evaluation (one p_right per row)
    @. _fisherright_one(a, m, k, N)
end


# Minimum Sensitivity
"""
    eval_minsens(data::AssociationDataFormat)

Compute Minimum Sensitivity (MinSens): the minimum of the two conditional supports.

# Arguments
- `data`: AssociationDataFormat with contingency counts.

# Formula
MinSens = min(a/m, d/n)

Where:
- `a` = f(node, collocate)
- `d` = f(neither)
- `m` = f(node)
- `n` = f(not-node) = (b + d)

# Notes
- Each denominator is guarded with `eps(Float64)` to avoid divide-by-zero on degenerate rows.
- Fused implementation (no extra array copies). Returns `Vector{Float64}`.
"""
function eval_minsens(data::AssociationDataFormat)
    @extract_values data a d m n
    @. min(a / max(m, eps(Float64)), d / max(n, eps(Float64)))
end

# Chi-square
"""
    eval_chisquare(data::AssociationDataFormat)

Compute Pearson’s χ² statistic for 2×2 tables using observed and expected counts.

# Arguments
- `data`: AssociationDataFormat with contingency counts and expected cells `E₁₁, E₁₂, E₂₁, E₂₂`.

# Formula
χ² = Σ_{i,j} ( (Oᵢⱼ − Eᵢⱼ)² / Eᵢⱼ )

# Notes
- Each expected value `Eᵢⱼ` is floored with `eps(Float64)` to prevent division by zero.
- Fully fused; no temporaries beyond the output.
"""
function eval_chisquare(data::AssociationDataFormat)
    @extract_values data a b c d E₁₁ E₁₂ E₂₁ E₂₂
    e = eps(Float64)
    @. ((a - max(E₁₁, e))^2 / max(E₁₁, e)) +
       ((b - max(E₁₂, e))^2 / max(E₁₂, e)) +
       ((c - max(E₂₁, e))^2 / max(E₂₁, e)) +
       ((d - max(E₂₂, e))^2 / max(E₂₂, e))
end

# T-score
"""
    eval_tscore(data::AssociationDataFormat)

Compute t-score (simple collocational t).

# Arguments
- `data`: AssociationDataFormat with counts.

# Formula
t = (a − E) / √a,  where E = (m·k)/N

# Notes
- Uses float promotion inside the expression (`2.0`, `float(...)`) to avoid integer division/overflow.
- Denominator guarded by `eps(Float64)` via `max(a, eps)` before `√`.
- Fused; returns `Vector{Float64}`.
"""
function eval_tscore(data::AssociationDataFormat)
    @extract_values data a m k N
    @. (a - (float(m) * k) / max(float(N), eps(Float64))) / sqrt(max(a, eps(Float64)))
end

# Z-score
"""
    eval_zscore(data::AssociationDataFormat)

Compute z-score under the hypergeometric/binomial approximation.

# Arguments
- `data`: AssociationDataFormat with counts.

# Formula
E = (m·k)/N  
Var = E · (1 − m/N) · (1 − k/N)  
z = (a − E) / √Var

# Notes
- All divisions done in float; variance guarded with `eps(Float64)` under the square root.
- Fused; returns `Vector{Float64}`.
"""
function eval_zscore(data::AssociationDataFormat)
    @extract_values data a m k N
    @. begin
        E = (float(m) * k) / max(float(N), eps(Float64))
        p1 = 1.0 - float(m) / max(float(N), eps(Float64))
        p2 = 1.0 - float(k) / max(float(N), eps(Float64))
        (a - E) / sqrt(max(E * p1 * p2, eps(Float64)))
    end
end

# Phi Coefficient
"""
    eval_phicoef(data::AssociationDataFormat)

Compute the ϕ (phi) coefficient for 2×2 tables.

# Arguments
- `data`: AssociationDataFormat with counts.

# Formula
ϕ = (ad − bc) / √((a + b)(c + d)(a + c)(b + d))

# Notes
- Products computed in float to avoid integer overflow.
- Adds a tiny `eps(Float64)` under the root to avoid zero division.
- Fused; returns `Vector{Float64}`.
"""
function eval_phicoef(data::AssociationDataFormat)
    @extract_values data a b c d
    @. (float(a) * d - float(b) * c) /
       sqrt(max((a + b) * (c + d) * (a + c) * (b + d), 0.0) + eps(Float64))
end

# Cramér's V
"""
    eval_cramersv(data::AssociationDataFormat)

Compute Cramér’s V for 2×2 tables.

# Arguments
- `data`: AssociationDataFormat with counts.

# Formula
V = √(χ² / (N · (k − 1))) with k = min(r, c) = 2 ⇒ denominator reduces to N.

# Notes
- Reuses `eval_chisquare(data)`; guards N via float promotion.
- Returns `Vector{Float64}`.
"""
function eval_cramersv(data::AssociationDataFormat)
    chi2 = eval_chisquare(data)
    con_tbl = cached_data(data.con_tbl)
    N = con_tbl.N[1]  # constant across rows
    @. sqrt(chi2 / max(float(N), eps(Float64)))
end


# Tschuprow's T
"""
    eval_tschuprowt(data::AssociationDataFormat)

Compute Tschuprow’s T for 2×2 tables.

# Arguments
- `data`: AssociationDataFormat with counts.

# Formula
T = √(χ² / N)

# Notes
- Equivalent to Cramér’s V in 2×2. Uses float N and fused broadcast.
"""
function eval_tschuprowt(data::AssociationDataFormat)
    chi2 = eval_chisquare(data)
    con_tbl = cached_data(data.con_tbl)
    N = con_tbl.N[1]
    @. sqrt(chi2 / max(float(N), eps(Float64)))
end

# Contingency Coefficient
"""
    eval_contcoef(data::AssociationDataFormat)

Compute Pearson’s contingency coefficient C.

# Arguments
- `data`: AssociationDataFormat with counts.

# Formula
C = √(χ² / (χ² + N))

# Notes
- Uses float N; fused expression; returns `Vector{Float64}`.
"""
function eval_contcoef(data::AssociationDataFormat)
    chi2 = eval_chisquare(data)
    con_tbl = cached_data(data.con_tbl)
    N = con_tbl.N[1]
    @. sqrt(chi2 / (chi2 + float(N)))
end

# Piatetsky–Shapiro
"""
    eval_piatetskyshapiro(data::AssociationDataFormat)

Compute the Piatetsky–Shapiro measure.

# Arguments
- `data`: AssociationDataFormat with counts.

# Formula
PS = a/N − (k·m)/N²

# Notes
- All divisions in float; fused implementation.
"""
function eval_piatetskyshapiro(data::AssociationDataFormat)
    @extract_values data a N k m
    @. (a / max(float(N), eps(Float64))) -
       ((float(k) * m) / max(float(N) * float(N), eps(Float64)))
end

# Yule’s Ω (Omega)
"""
    eval_yuleomega(data::AssociationDataFormat)

Compute Yule’s Omega.

# Arguments
- `data`: AssociationDataFormat with counts.

# Formula
Ω = (√(ad) − √(bc)) / (√(ad) + √(bc))

# Notes
- Computes `ad`, `bc` in float; guards with `max(..., 0.0)` inside the roots
  and `max(denom, eps(Float64))` to avoid zero division.
- Fused expression; returns `Vector{Float64}`.
"""
function eval_yuleomega(data::AssociationDataFormat)
    @extract_values data a b c d
    @. (sqrt(max(float(a) * d, 0.0)) - sqrt(max(float(b) * c, 0.0))) /
       max(sqrt(max(float(a) * d, 0.0)) + sqrt(max(float(b) * c, 0.0)), eps(Float64))
end


# Yule’s Q
"""
    eval_yuleq(data::AssociationDataFormat)

Compute Yule’s Q.

# Arguments
- `data`: AssociationDataFormat with counts.

# Formula
Q = (ad − bc) / (ad + bc)

# Notes
- Uses float products to avoid overflow; denominator guarded with `eps(Float64)`.
- Fused; returns `Vector{Float64}`.
"""
function eval_yuleq(data::AssociationDataFormat)
    @extract_values data a b c d
    @. (float(a) * d - float(b) * c) /
       max(float(a) * d + float(b) * c, eps(Float64))
end