
# log2((a / N) / ((k / N) * (m / N)))
function eval_pmi(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    log2.((con_tbl.a ./ con_tbl.N) .- log2.((con_tbl.k ./ con_tbl.N) .* (con_tbl.m ./ con_tbl.N)))
end

const pmi = eval_pmi


# log2((a^2 / N) / ((k / N) * (m / N)))
# Pointwise Mutual Information²     
function eval_pmi2(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    log2.((con_tbl.a .^ 2 ./ con_tbl.N) .- log2.((con_tbl.k ./ con_tbl.N) .* (con_tbl.m ./ con_tbl.N)))
end

const pmi² = eval_pmi2


# log2((a^3 / N) / ((k / N) * (m / N)))
# Pointwise Mutual Information³ 
function eval_pmi3(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    log2.((con_tbl.a .^ 3 ./ con_tbl.N) .- log2.((con_tbl.k ./ con_tbl.N) .* (con_tbl.m ./ con_tbl.N)))
end

const pmi³ = eval_pmi3


# max(0, log2((a / N) / ((k / N) * (m / N))))
function eval_ppmi(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    max.(0, log2.((con_tbl.a ./ con_tbl.N)) .- log2.((con_tbl.k ./ con_tbl.N) .* (con_tbl.m ./ con_tbl.N)))
end

const ppmi = eval_ppmi


# 2 * (a * log(a) - (a + b) * log(a + b) + c * log(c) - (c + d) * log(c + d))
function eval_llr(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    2 * (con_tbl.a * log(con_tbl.a) - (con_tbl.a + con_tbl.b) * log(con_tbl.a + con_tbl.b) + con_tbl.c * log(con_tbl.c) - (con_tbl.c + con_tbl.d) * log(con_tbl.c + con_tbl.d))
end

const llr = eval_llr

# deltapi
function eval_deltapi(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    (con_tbl.a ./ con_tbl.m) .- (con_tbl.c ./ con_tbl.n)
end

const deltapi = eval_deltapi
const δπ = eval_deltapi


# minimum sensitivity
function eval_min_sen(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    min_sensitivity = min(con_tbl.a ./ con_tbl.m, con_tbl.a ./ con_tbl.k)
end

const min_sen = eval_min_sen


# a + b  = m, c + d = n, a + c = k, b + d = l

# Dice
function eval_dice(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    dice = (2 .* con_tbl.a) ./ (con_tbl.m + con_tbl.k)
end

const dice = eval_dice


# Log Dice
function eval_logdice(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    log_dice = log.((2 .* con_tbl.a) ./ (con_tbl.m + con_tbl.k))
end

const log_dice = eval_logdice


# Relative Risk 
function eval_relative_risk(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    relative_risk = (con_tbl.a ./ con_tbl.m) ./ (con_tbl.c ./ con_tbl.n)
end

const rel_risk = eval_relative_risk
const rr = eval_relative_risk


# Log Relative Risk
function eval_log_relative_risk(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    log_relative_risk = log.(con_tbl.a ./ con_tbl.m) .- log.(con_tbl.c ./ con_tbl.n)
end

const log_rel_risk = eval_log_relative_risk
const lrr = eval_log_relative_risk


# Risk Difference
function eval_risk_difference(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    risk_difference = (con_tbl.a ./ con_tbl.m) .- (con_tbl.c ./ con_tbl.n)
end

const risk_diff = eval_risk_difference
const rd = eval_risk_difference


# Attributable Risk
function eval_attributable_risk(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    attributable_risk = (con_tbl.a ./ con_tbl.m) .- (con_tbl.c ./ con_tbl.n)
end

const attr_risk = eval_attributable_risk
const ar = eval_attributable_risk


# Odds Ratio
function eval_odds_ratio(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    odds_ratio = (con_tbl.a ./ con_tbl.b) ./ (con_tbl.c ./ con_tbl.d)
end

const odds_ratio = eval_odds_ratio
const or = eval_odds_ratio


# Log Odds Ratio
function eval_log_odds_ratio(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    log_odds_ratio = log.(con_tbl.a ./ con_tbl.b) .- log.(con_tbl.c ./ con_tbl.d)
end

const log_odds_ratio = eval_log_odds_ratio
const lor = eval_log_odds_ratio


# Jaccard Index
# "Jaccard", a/(a + b + c)
function eval_jaccard_index(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    jaccard_index = con_tbl.a ./ (con_tbl.a .+ con_tbl.b .+ con_tbl.c)
end

const jaccard = eval_jaccard_index


# Ochiai Index
# "Ochiai", a / sqrt((a + b) * (a + c))
function eval_ochiai_index(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    ochiai_index = con_tbl.a ./ sqrt.((con_tbl.a .+ con_tbl.b) .* (con_tbl.a .+ con_tbl.c))
end

const ochiai = eval_ochiai_index


# Piatetsky Shapiro
# "Piatetsky Shapiro", a - ((k + m) / N)
function eval_piatetsky_shapiro(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    piatetsky_shapiro = con_tbl.a .- ((con_tbl.k .* con_tbl.m) ./ con_tbl.N)
end

const piatetsky_shapiro = eval_piatetsky_shapiro


# Yule's Q
# "Yule's Q", (a * d - b * c) / (a * d + b * c)
function eval_yule_q(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    yule_q = (con_tbl.a .* con_tbl.d .- con_tbl.b .* con_tbl.c) ./ (con_tbl.a .* con_tbl.d .+ con_tbl.b .* con_tbl.c)
end

const yule_q = eval_yule_q


# Yule's Y
# "Yule's Y", sqrt((a * d) - (b * c)) / sqrt((a * d) + (b * c))
function eval_yule_y(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    yule_y = sqrt.((con_tbl.a .* con_tbl.d) .- (con_tbl.b .* con_tbl.c) ./ (con_tbl.a .* con_tbl.d) .+ (con_tbl.b .* con_tbl.c))
end

const yule_y = eval_yule_y




# (a * d - b * c) / sqrt((a + b) * (c + d) * (a + c) * (b + d))
function eval_phi_coefficient(data::ContingencyTable)
    cont_tbl = extract_cached_data(data.con_tbl)
    phi_coefficient = (cont_tbl.a .* cont_tbl.d .- cont_tbl.b .* cont_tbl.c) ./ sqrt.((cont_tbl.a .+ cont_tbl.b) .* (cont_tbl.c .+ cont_tbl.d) .* (cont_tbl.a .+ cont_tbl.c) .* (cont_tbl.b .+ cont_tbl.d))
end

const phi = eval_phi_coefficient
const φ = eval_phi_coefficient


# Cramers V
# "Cramers V", sqrt(chi2 / (N * (min(k, l) - 1)))
function eval_cramers_v(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    cramers_v = sqrt.(chi_square(data) ./ (con_tbl.N * (min.(con_tbl.k, con_tbl.l) .- 1)))
end

const cramers_v = eval_cramers_v


# Tschuprow's T
# "Tschuprow's T", sqrt(chi2 / (N * (min(k, l) - 1)))

function eval_tschuprow_t(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    tschuprow_t = sqrt.(chi_square(data) ./ (con_tbl.N .* (min.(con_tbl.k, con_tbl.l) .- 1)))
end

const tschuprow_t = eval_tschuprow_t


# Contingency Coefficient
# "Contingency Coefficient", sqrt(chi2 / (chi2 + N))
function eval_contingency_coefficient(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    contingency_coefficient = sqrt.(chi_square(data) ./ (chi_square(data) .+ con_tbl.N))
end

const contingency_coefficient = eval_contingency_coefficient


# Cosine Similarity
# "Cosine Similarity", a / sqrt((a + b) * (a + c))
function eval_cosine_similarity(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    cosine_similarity = con_tbl.a ./ sqrt.((con_tbl.a .+ con_tbl.b) .* (con_tbl.a .+ con_tbl.c))
end

const cosine_similarity = eval_cosine_similarity


# Overlap Coefficient
# "Overlap Coefficient", a / min(m, k)
function eval_overlap_coefficient(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    overlap_coefficient = con_tbl.a ./ min(con_tbl.m, con_tbl.k)
end

const overlap_coefficient = eval_overlap_coefficient


# Kulczynski Similarity
# "Kulczynski Similarity", a / ((k + m) / 2)
function eval_kulczynski_similarity(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    kulczynski_similarity = con_tbl.a ./ ((con_tbl.k .+ con_tbl.m) ./ 2)
end

const kulczynski_similarity = eval_kulczynski_similarity


# Tanimoto Coefficient
# "Tanimoto Coefficient", a / (k + m - a)
function eval_tanimoto_coefficient(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    tanimoto_coefficient = con_tbl.a ./ (con_tbl.k .+ con_tbl.m .- con_tbl.a)
end

const tanimoto_coefficient = eval_tanimoto_coefficient


# Hamann Similarity
# "Hamann Similarity", (a + d - b - c) / N
function eval_hamann_similarity(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    hamann_similarity = (con_tbl.a .+ con_tbl.d .- con_tbl.b .- con_tbl.c) ./ con_tbl.N
end

const hamann_similarity = eval_hamann_similarity


# Goodman-Kruskal Index
# "Goodman-Kruskal Index", (a * d - b * c) / (a * d + b * c)
function eval_goodman_kruskal_index(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    goodman_kruskal_index = (con_tbl.a .* con_tbl.d .- con_tbl.b .* con_tbl.c) ./ (con_tbl.a .* con_tbl.d .+ con_tbl.b .* con_tbl.c)
end

const goodman_kruskal_index = eval_goodman_kruskal_index


# Gower's Coefficient
# "Gower's Coefficient", (a + d) / (a + d + 2 * (b + c))
function eval_gower_coefficient(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    gower_coefficient = (con_tbl.a .+ con_tbl.d) ./ (con_tbl.a .+ con_tbl.d .+ 2 .* (con_tbl.b .+ con_tbl.c))
end

const gower_coefficient = eval_gower_coefficient


# Czekanowski-Dice Coefficient
# "Czekanowski-Dice Coefficient", 2 * a / (2 * a + b + c)
function eval_czekanowski_dice_coefficient(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    czekanowski_dice_coefficient = 2 .* con_tbl.a ./ (2 .* con_tbl.a .+ con_tbl.b .+ con_tbl.c)
end

const czekanowski_dice_coefficient = eval_czekanowski_dice_coefficient


# Ochiai Coefficient
# "Ochiai Coefficient", a / sqrt((k + m) * (k + c))
function eval_ochiai_coefficient(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    ochiai_coefficient = con_tbl.a ./ sqrt.((con_tbl.k .+ con_tbl.m) .* (con_tbl.k .+ con_tbl.c))
end

const ochiai_coefficient = eval_ochiai_coefficient


# Sorgenfrey Index
# "Sorgenfrey Index", (a + d) / (2 * (a + d) + b + c)
function eval_sorgenfrey_index(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    sorgenfrey_index = (con_tbl.a .+ con_tbl.d) ./ (2 .* (con_tbl.a .+ con_tbl.d) .+ con_tbl.b .+ con_tbl.c)
end

const sorgenfrey_index = eval_sorgenfrey_index


# Mountford's Coefficient
# "Mountford's Coefficient", (a + d) / (a + d + 2 * sqrt((b + c) * (k + m)))
function eval_mountford_coefficient(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    mountford_coefficient = (con_tbl.a .+ con_tbl.d) ./ (con_tbl.a .+ con_tbl.d .+ 2 .* sqrt.((con_tbl.b .+ con_tbl.c) .* (con_tbl.k .+ con_tbl.m)))
end

const mountford_coefficient = eval_mountford_coefficient


# Sokal-Sneath Index
# "Sokal-Sneath Index", a / (a + 2 *
function eval_sokal_sneath_index(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    sokal_sneath_index = con_tbl.a ./ (con_tbl.a .+ 2 .* (con_tbl.b .+ con_tbl.c))
end

const sokal_sneath_index = eval_sokal_sneath_index


# Rogers-Tanimoto Coefficient
# "Rogers-Tanimoto Coefficient", a / (a + 2 * (b + c))
function eval_rogers_tanimoto_coefficient(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    rogers_tanimoto_coefficient = con_tbl.a ./ (con_tbl.a .+ 2 .* (con_tbl.b .+ con_tbl.c))
end

const rogers_tanimoto_coefficient = eval_rogers_tanimoto_coefficient


# Sokal-Michener Coefficient
# "Sokal-Michener Coefficient", (a + d) / (a + d + 2 * (
function eval_sokal_michener_coefficient(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    sokal_michener_coefficient = (con_tbl.a .+ con_tbl.d) ./ (con_tbl.a .+ con_tbl.d .+ 2 .* (con_tbl.b .+ con_tbl.c))
end

const sokal_michener = eval_sokal_michener_coefficient


# Gravity G Index
# "Gravity G Index", (a * d) / (b * c)



# =============================================================================

# Fisher's Exact Test
function eval_fisher_exact_test(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    fisher_exact_test = (con_tbl.a * con_tbl.d) / (con_tbl.b * con_tbl.c)
end

const fisher = eval_fisher_exact_test


# Chi Square
function eval_chi_square(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    chi_square = (con_tbl.a * con_tbl.d - con_tbl.b * con_tbl.c)^2 / ((con_tbl.a + con_tbl.b) * (con_tbl.c + con_tbl.d) * (con_tbl.a + con_tbl.c) * (con_tbl.b + con_tbl.d))
end

const chi_square = eval_chi_square
const χ² = eval_chi_square


# Poisson
function eval_poisson(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    poisson = (con_tbl.a .- con_tbl.m) .^ 2 ./ con_tbl.m
end

const poisson = eval_poisson

# Define a general _evaluate function for the AssociationMetric type

# Define specific implementations of _evaluate for each metric type
# _evaluate(metric::PMI, data::ContingencyTable; node=nothing, kwargs...) = eval_pmi(data, node; kwargs...)
# _evaluate(metric::PMI2, data::ContingencyTable; node=nothing, kwargs...) = eval_pmi2(data, node; kwargs...)
# _evaluate(metric::PMI3, data::ContingencyTable; node=nothing, kwargs...) = eval_pmi3(data, node; kwargs...)
# _evaluate(metric::PPMI, data::ContingencyTable; node=nothing, kwargs...) = eval_ppmi(data, node; kwargs...)
# _evaluate(metric::LLR, data::ContingencyTable; node=nothing, kwargs...) = eval_llr(data, node; kwargs...)
# _evaluate(metric::Dice, data::ContingencyTable; node=nothing, kwargs...) = eval_dice(data, node; kwargs...)
# _evaluate(metric::LogDice, data::ContingencyTable; node=nothing, kwargs...) = eval_log_dice(data, node; kwargs...)
# _evaluate(metric::DeltaP, data::ContingencyTable; node=nothing, kwargs...) = eval_deltapi(data, node; kwargs...)
# _evaluate(metric::DeltaP, data::ContingencyTable; node=nothing, kwargs...) = eval_deltapi(data, node; kwargs...)
# _evaluate(metric::DeltaP, data::ContingencyTable; node=nothing, kwargs...) = eval_deltapi(data, node; kwargs...)
# _evaluate(metric::DeltaP, data::ContingencyTable; node=nothing, kwargs...) = eval_deltapi(data, node; kwargs...)
# _evaluate(metric::DeltaP, data::ContingencyTable; node=nothing, kwargs...) = eval_deltapi(data, node; kwargs...)
# _evaluate(metric::DeltaP, data::ContingencyTable; node=nothing, kwargs...) = eval_deltapi(data, node; kwargs...)


# for M in (metrics...)
#     @eval @inline (metric::$M)(data, node; kwargs...) = _evaluate(metric, data; node=node, kwargs...)
# end

metrics = (PMI, PMI2, PMI3, PPMI, LLR, Dice, LogDice, DeltaPi) # Assuming these are defined types

for M in metrics
    @eval begin
        @inline function (::$M)(data::ContingencyTable)
            eval_fn_name = Symbol("eval_", lowercase(string($M)))
            eval_fn = getfield(StringAssociations, eval_fn_name)
            invoke(eval_fn, Tuple{typeof(data)}, data)
        end
    end
end


# DeltaP
# ChiSquare
# FisherExactTest
# LogDice
# JaccardIndex
# DiceCoefficient
# PhiCoefficient
# CramersV
# TschuprowT
# ContingencyCoefficient
# CosineSimilarity
# OverlapCoefficient