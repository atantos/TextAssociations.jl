# =====================================
# File: src/metrics/similarity.jl
# Similarity metrics
# =====================================

# Dice Coefficient
"""
    eval_dice(data::AssociationDataFormat)

Compute the Dice coefficient for each collocate pair.

# Arguments
- `data`: AssociationDataFormat with contingency counts.

# Formula
Dice = 2a / (m + k)

Where:
- `a` = co-occurrence frequency
- `m` = f(node)
- `k` = f(collocate)

# Notes
- Uses a small numerical guard: `max(m + k, eps(Float64))` to avoid divide-by-zero in degenerate rows.
- Fused implementation with float literals (no extra copies). Returns `Vector{Float64}` aligned with `assoc_df(data)`.
"""
function eval_dice(data::AssociationDataFormat)
    @extract_values data a m k
    @. (2.0 * a) / max(m + k, eps(Float64))
end

"""
    eval_logdice(data::AssociationDataFormat)

Compute LogDice (base-2).

# Arguments
- `data`: AssociationDataFormat with contingency counts.

# Formula
LogDice = 14 + log₂(2a) − log₂(m + k)

Where:
- `a` = co-occurrence frequency
- `m` = f(node)
- `k` = f(collocate)

# Notes
- Uses `log2_safe` (expected to clamp/guard `log2(0)` internally).
- Fused implementation; no final clamping here (policy left to `log2_safe`).
"""
function eval_logdice(data::AssociationDataFormat)
    @extract_values data a m k
    @. 14 + log2_safe(2.0 * a) - log2_safe(m + k)
end

"""
    eval_jaccardidx(data::AssociationDataFormat)

Compute the Jaccard index (intersection over union).

# Arguments
- `data`: AssociationDataFormat with contingency counts.

# Formula
Jaccard = a / (a + b + c)

# Notes
- Guarded by `max(a + b + c, eps(Float64))`.
- Fused implementation; returns `Vector{Float64}`.
"""
function eval_jaccardidx(data::AssociationDataFormat)
    @extract_values data a b c
    @. a / max(a + b + c, eps(Float64))
end

"""
    eval_ochiaiidx(data::AssociationDataFormat)

Compute the Ochiai index (cosine on binary events).

# Arguments
- `data`: AssociationDataFormat with contingency counts.

# Formula
Ochiai = a / √(m · k)

# Notes
- Uses `sqrt(max(float(m) * k, eps(Float64)))` to avoid overflow and zero division.
- Fused implementation; returns `Vector{Float64}`.
"""
function eval_ochiaiidx(data::AssociationDataFormat)
    @extract_values data a m k
    @. a / sqrt(max(float(m) * k, eps(Float64)))
end

# Cosine Similarity
"""
    eval_cosinesim(data::AssociationDataFormat)

Compute cosine similarity for binary co-occurrence.

# Arguments
- `data`: AssociationDataFormat with contingency counts.

# Formula
CosineSim = a / √(m · k)

# Notes
- Same numeric guard as `Ochiai`. Fused; returns `Vector{Float64}`.
"""
function eval_cosinesim(data::AssociationDataFormat)
    @extract_values data a m k
    @. a / sqrt(max(float(m) * k, eps(Float64)))
end

# Overlap Coefficient
"""
    eval_overlapcoef(data::AssociationDataFormat)

Compute the Overlap (Szymkiewicz–Simpson) coefficient.

# Arguments
- `data`: AssociationDataFormat with contingency counts.

# Formula
Overlap = a / min(m, k)

# Notes
- Guarded with `max(min(m, k), eps(Float64))`. Fused; returns `Vector{Float64}`.
"""
function eval_overlapcoef(data::AssociationDataFormat)
    @extract_values data a m k
    @. a / max(min(m, k), eps(Float64))
end

# Kulczynski Similarity
"""
    eval_kulczynskisim(data::AssociationDataFormat)

Compute Kulczyński similarity (average of conditional supports).

# Arguments
- `data`: AssociationDataFormat with contingency counts.

# Formula
Kulczyński = 0.5 * (a/m + a/k)

# Notes
- Guards each denominator with `eps(Float64)`. Fused; returns `Vector{Float64}`.
"""
function eval_kulczynskisim(data::AssociationDataFormat)
    @extract_values data a m k
    @. 0.5 * (a / max(m, eps(Float64)) + a / max(k, eps(Float64)))
end

# Tanimoto Coefficient
"""
    eval_tanimotocoef(data::AssociationDataFormat)

Compute the Tanimoto coefficient (set version).

# Arguments
- `data`: AssociationDataFormat with contingency counts.

# Formula
Tanimoto = a / (m + k − a)

# Notes
- Guarded by `max(float(m) + k − a, eps(Float64))` to avoid overflow/zero division.
- Fused; returns `Vector{Float64}`.
"""
function eval_tanimotocoef(data::AssociationDataFormat)
    @extract_values data a m k
    @. a / max(float(m) + k - a, eps(Float64))
end

# Rogers-Tanimoto Coefficient
"""
    eval_rogerstanimotocoef(data::AssociationDataFormat)

Compute the Rogers–Tanimoto coefficient.

# Arguments
- `data`: AssociationDataFormat with contingency counts.

# Formula
Rogers–Tanimoto = a / (a + 2(b + c))

# Notes
- Denominator uses a float literal: `a + 2.0*(b + c)`; guarded with `eps(Float64)`.
- Fused; returns `Vector{Float64}`.
"""
function eval_rogerstanimotocoef(data::AssociationDataFormat)
    @extract_values data a b c
    @. a / max(a + 2.0 * (b + c), eps(Float64))
end

# Rogers-Tanimoto Coefficient 2
"""
    eval_rogerstanimotocoef2(data::AssociationDataFormat)

Compute the Rogers–Tanimoto coefficient variant including negative matches.

# Arguments
- `data`: AssociationDataFormat with contingency counts.

# Formula
Rogers–Tanimoto₂ = (a + d) / (a + 2(b + c) + d)

# Notes
- Uses `a + 2.0*(b + c) + d`; guarded with `eps(Float64)`. Fused result.
"""
function eval_rogerstanimotocoef2(data::AssociationDataFormat)
    @extract_values data a b c d
    @. (a + d) / max(a + 2.0 * (b + c) + d, eps(Float64))
end

# Hamman Similarity (fixed name from original HammanSim)
"""
    eval_hammansim(data::AssociationDataFormat)

Compute Hamman similarity (balanced agreements minus disagreements).

# Arguments
- `data`: AssociationDataFormat with contingency counts.

# Formula
Hamman = (a + d − b − c) / N

# Notes
- Denominator guarded: `max(float(N), eps(Float64))`. Fused; returns `Vector{Float64}`.
"""
function eval_hammansim(data::AssociationDataFormat)
    @extract_values data a b c d N
    @. ((a + d) - (b + c)) / max(float(N), eps(Float64))
end

# Hamman Similarity (variant)
"""
    eval_hammansim2(data::AssociationDataFormat)

Compute Hamman variant 2 (difference over adjusted union).

# Arguments
- `data`: AssociationDataFormat with contingency counts.

# Formula
Hamman₂ = (a − d) / (a + b + c − d)

# Notes
- Guarded denominator; fused, no extra copies.
"""
function eval_hammansim2(data::AssociationDataFormat)
    @extract_values data a b c d
    @. (a - d) / max(a + b + c - d, eps(Float64))
end

# Goodman-Kruskal Index
"""
    eval_goodmankruskalidx(data::AssociationDataFormat)

Compute the Goodman–Kruskal index on 2×2 tables.

# Arguments
- `data`: AssociationDataFormat with contingency counts.

# Formula
GK = (ad − bc) / (ad + bc)

# Notes
- Products done in float to avoid overflow: `float(a)*d`, `float(b)*c`.
- Denominator guarded; fused.
"""
function eval_goodmankruskalidx(data::AssociationDataFormat)
    @extract_values data a b c d
    @. (float(a) * d - float(b) * c) / max(float(a) * d + float(b) * c, eps(Float64))
end

# Gower Coefficient
"""
    eval_gowercoef(data::AssociationDataFormat)

Compute the (binary) Gower coefficient (presence-only form).

# Arguments
- `data`: AssociationDataFormat with contingency counts.

# Formula
Gower = a / (a + b + c)

# Notes
- Same as Jaccard in this binary setting. Guarded denominator; fused.
"""
function eval_gowercoef(data::AssociationDataFormat)
    @extract_values data a b c
    @. a / max(a + b + c, eps(Float64))
end

# Gower Coefficient (variant)
"""
    eval_gowercoef2(data::AssociationDataFormat)

Compute Gower coefficient including negative matches.

# Arguments
- `data`: AssociationDataFormat with contingency counts.

# Formula
Gower₂ = (a + d) / (a + d + 2(b + c))

# Notes
- Uses float literal in denominator; guarded; fused.
"""
function eval_gowercoef2(data::AssociationDataFormat)
    @extract_values data a b c d
    @. (a + d) / max(a + d + 2.0 * (b + c), eps(Float64))
end

# Czekanowski–Dice Coefficient
"""
    eval_czekanowskidicecoef(data::AssociationDataFormat)

Compute the Czekanowski–Dice (Sørensen–Dice) coefficient.

# Arguments
- `data`: AssociationDataFormat with contingency counts.

# Formula
CzekanowskiDice = 2a / (2a + b + c)

# Notes
- Uses float literal `2.0*a` both numerator and denominator; guarded; fused.
"""
function eval_czekanowskidicecoef(data::AssociationDataFormat)
    @extract_values data a b c
    @. (2.0 * a) / max(2.0 * a + b + c, eps(Float64))
end

# Sorgenfrey Index
"""
    eval_sorgenfreyidx(data::AssociationDataFormat)

Compute the Sorgenfrey index.

# Arguments
- `data`: AssociationDataFormat with contingency counts.

# Formula
Sorgenfrey = (2a − b − c) / (2a + b + c)

# Notes
- Uses float literal `2.0*a`; guarded denominator; fused.
"""
function eval_sorgenfreyidx(data::AssociationDataFormat)
    @extract_values data a b c
    @. (2.0 * a - b - c) / max(2.0 * a + b + c, eps(Float64))
end

# Sorgenfrey Index (variant)
"""
    eval_sorgenfreyidx2(data::AssociationDataFormat)

Compute the Sorgenfrey index variant including negative matches.

# Arguments
- `data`: AssociationDataFormat with contingency counts.

# Formula
Sorgenfrey₂ = (a + d) / (2(a + d) + b + c)

# Notes
- Denominator uses `2.0*(a + d)`; guarded; fused.
"""
function eval_sorgenfreyidx2(data::AssociationDataFormat)
    @extract_values data a b c d
    @. (a + d) / max(2.0 * (a + d) + b + c, eps(Float64))
end

# Mountford Coefficient
"""
    eval_mountfordcoef(data::AssociationDataFormat)

Compute the Mountford coefficient.

# Arguments
- `data`: AssociationDataFormat with contingency counts.

# Formula
Mountford = a / (a + 2b + 2c)

# Notes
- Uses float literals: `2.0*b`, `2.0*c`; guarded; fused.
"""
function eval_mountfordcoef(data::AssociationDataFormat)
    @extract_values data a b c
    @. a / max(a + 2.0 * b + 2.0 * c, eps(Float64))
end

# Mountford Coefficient (variant)
"""
    eval_mountfordcoef2(data::AssociationDataFormat)

Compute Mountford coefficient variant with size correction.

# Arguments
- `data`: AssociationDataFormat with contingency counts.

# Formula
Mountford₂ = (a + d) / (a + d + 2√((b + c)(k + m)))

# Notes
- Uses float inside the square root to avoid overflow: `float(b + c) * (k + m)`.
- Denominator guarded; fused.
"""
function eval_mountfordcoef2(data::AssociationDataFormat)
    @extract_values data a b c d k m
    @. (a + d) / max((a + d) + 2.0 * sqrt(max(float(b + c) * (k + m), 0.0)), eps(Float64))
end

# Sokal–Sneath Index
"""
    eval_sokalsneathidx(data::AssociationDataFormat)

Compute the Sokal–Sneath index.

# Arguments
- `data`: AssociationDataFormat with contingency counts.

# Formula
Sokal–Sneath = a / (a + 2(b + c))

# Notes
- Uses `a / max(a + 2.0*(b + c), eps(Float64))`; fused; returns `Vector{Float64}`.
"""
function eval_sokalsneathidx(data::AssociationDataFormat)
    @extract_values data a b c
    @. a / max(a + 2.0 * (b + c), eps(Float64))
end

# Sokal–Michener Coefficient
"""
    eval_sokalmichenercoef(data::AssociationDataFormat)

Compute the Sokal–Michener coefficient (simple matching).

# Arguments
- `data`: AssociationDataFormat with contingency counts.

# Formula
Sokal–Michener = (a + d) / N

# Notes
- Denominator guarded: `max(float(N), eps(Float64))`. Fused; returns `Vector{Float64}`.
"""
function eval_sokalmichenercoef(data::AssociationDataFormat)
    @extract_values data a d N
    @. (a + d) / max(float(N), eps(Float64))
end
