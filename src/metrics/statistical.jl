# =====================================
# File: src/metrics/statistical.jl
# Statistical metrics
# =====================================

# Delta Pi
function eval_deltapi(data::AssociationDataFormat)
    @extract_values data a c m n
    (a ./ max.(m, eps())) .- (c ./ max.(n, eps()))
end

# Minimum Sensitivity
function eval_minsens(data::AssociationDataFormat)
    @extract_values data a d m n
    min.(a ./ max.(m, eps()), d ./ max.(n, eps()))
end

# Chi-square
function eval_chisquare(data::AssociationDataFormat)
    @extract_values data a b c d E₁₁ E₁₂ E₂₁ E₂₂

    E₁₁ = max.(E₁₁, eps())
    E₁₂ = max.(E₁₂, eps())
    E₂₁ = max.(E₂₁, eps())
    E₂₂ = max.(E₂₂, eps())

    ((a .- E₁₁) .^ 2 ./ E₁₁) .+ ((b .- E₁₂) .^ 2 ./ E₁₂) .+
    ((c .- E₂₁) .^ 2 ./ E₂₁) .+ ((d .- E₂₂) .^ 2 ./ E₂₂)
end

# T-score
function eval_tscore(data::AssociationDataFormat)
    @extract_values data a m k N
    expected = (m .* k) ./ N
    (a .- expected) ./ sqrt.(max.(a, eps()))
end

# Z-score
function eval_zscore(data::AssociationDataFormat)
    @extract_values data a m k N
    expected = (m .* k) ./ N
    variance = expected .* (1 .- m ./ N) .* (1 .- k ./ N)
    (a .- expected) ./ sqrt.(max.(variance, eps()))
end

# Phi Coefficient
function eval_phicoef(data::AssociationDataFormat)
    @extract_values data a b c d
    num = (a .* d) .- (b .* c)
    denom = sqrt.((a .+ b) .* (c .+ d) .* (a .+ c) .* (b .+ d) .+ eps())
    num ./ denom
end

# Cramér's V
function eval_cramersv(data::AssociationDataFormat)
    chi2 = eval_chisquare(data)
    con_tbl = extract_cached_data(data.con_tbl)
    N = con_tbl.N[1]  # N is constant for all rows
    sqrt.(chi2 ./ (N * (2 - 1)))
end

# Tschuprow's T
function eval_tschuprowt(data::AssociationDataFormat)
    chi2 = eval_chisquare(data)
    con_tbl = extract_cached_data(data.con_tbl)
    N = con_tbl.N[1]
    sqrt.(chi2 ./ N)
end

# Contingency Coefficient
function eval_contcoef(data::AssociationDataFormat)
    chi2 = eval_chisquare(data)
    con_tbl = extract_cached_data(data.con_tbl)
    N = con_tbl.N[1]
    sqrt.(chi2 ./ (chi2 .+ N))
end

# Piatetsky-Shapiro
function eval_piatetskyshapiro(data::AssociationDataFormat)
    @extract_values data a N k m
    (a ./ N) .- ((k .* m) ./ (N .* N))
end

# Yule's Omega
function eval_yuleomega(data::AssociationDataFormat)
    @extract_values data a b c d
    num = sqrt.(max.(a .* d, 0)) .- sqrt.(max.(b .* c, 0))
    denom = sqrt.(a .* d) .+ sqrt.(b .* c)
    num ./ max.(denom, eps())
end

# Yule's Q
function eval_yuleq(data::AssociationDataFormat)
    @extract_values data a b c d
    num = (a .* d) .- (b .* c)
    denom = (a .* d) .+ (b .* c)
    num ./ max.(denom, eps())
end