# =====================================
# File: src/metrics/statistical.jl
# Statistical metrics
# =====================================

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