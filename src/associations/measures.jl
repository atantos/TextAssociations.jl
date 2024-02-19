
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
function eval_minsensitivity(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    min_sensitivity = min(con_tbl.a ./ con_tbl.m, con_tbl.a ./ con_tbl.k)
end

const minsen = eval_minsensitivity


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
    logdice = log.((2 .* con_tbl.a) ./ (con_tbl.m + con_tbl.k))
end

const logdice = eval_logdice


# Relative Risk 
function eval_relrisk(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    relrisk = (con_tbl.a ./ con_tbl.m) ./ (con_tbl.c ./ con_tbl.n)
end

const rel_risk = eval_relrisk
const rr = eval_relrisk


# Log Relative Risk
function eval_logrelrisk(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    logrelrisk = log.(con_tbl.a ./ con_tbl.m) .- log.(con_tbl.c ./ con_tbl.n)
end

const log_rel_risk = eval_logrelrisk
const lrr = eval_logrelrisk


# Risk Difference
function eval_riskdiff(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    riskdiff = (con_tbl.a ./ con_tbl.m) .- (con_tbl.c ./ con_tbl.n)
end

const riskdiff = eval_riskdiff
const rd = eval_riskdiff


# Attributable Risk
function eval_attrrisk(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    attrrisk = (con_tbl.a ./ con_tbl.m) .- (con_tbl.c ./ con_tbl.n)
end

const attr_risk = eval_attrrisk
const ar = eval_attrrisk


# Odds Ratio
function eval_oddsratio(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    oddsratio = (con_tbl.a ./ con_tbl.b) ./ (con_tbl.c ./ con_tbl.d)
end

const odds_ratio = eval_oddsratio
const or = eval_oddsratio


# Log Odds Ratio
function eval_logoddsratio(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    logoddsratio = log.(con_tbl.a ./ con_tbl.b) .- log.(con_tbl.c ./ con_tbl.d)
end

const logoddsratio = eval_logoddsratio
const lor = eval_logoddsratio


# Jaccard Index
# "Jaccard", a/(a + b + c)
function eval_jaccardindex(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    jaccardindex = con_tbl.a ./ (con_tbl.a .+ con_tbl.b .+ con_tbl.c)
end

const jaccard = eval_jaccardindex


# Ochiai Index
# "Ochiai", a / sqrt((a + b) * (a + c))
function eval_ochiaiindex(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    ochiaiindex = con_tbl.a ./ sqrt.((con_tbl.a .+ con_tbl.b) .* (con_tbl.a .+ con_tbl.c))
end

const ochiai = eval_ochiaiindex

# Ochiai Coefficient
# "Ochiai Coefficient", a / sqrt((k + m) * (k + c))
function eval_ochiaicoef(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    ochiaicoef = con_tbl.a ./ sqrt.((con_tbl.k .+ con_tbl.m) .* (con_tbl.k .+ con_tbl.c))
end

const ochiai_ochiaicoef = eval_ochiaicoef

# Piatetsky Shapiro
# "Piatetsky Shapiro", a - ((k + m) / N)
function eval_piatetskyshapiro(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    piatetskyshapiro = con_tbl.a .- ((con_tbl.k .* con_tbl.m) ./ con_tbl.N)
end

const piatetskyshapiro = eval_piatetskyshapiro


# Yule's Q
# "Yule's Q", (a * d - b * c) / (a * d + b * c)
function eval_yuleq(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    yuleq = (con_tbl.a .* con_tbl.d .- con_tbl.b .* con_tbl.c) ./ (con_tbl.a .* con_tbl.d .+ con_tbl.b .* con_tbl.c)
end

const yuleq = eval_yuleq


# Yule's Y
# "Yule's Y", sqrt((a * d) - (b * c)) / sqrt((a * d) + (b * c))
function eval_yuley(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    yuley = sqrt.((con_tbl.a .* con_tbl.d) .- (con_tbl.b .* con_tbl.c) ./ (con_tbl.a .* con_tbl.d) .+ (con_tbl.b .* con_tbl.c))
end

const yuley = eval_yuley

# Phi Coefficient
# (a * d - b * c) / sqrt((a + b) * (c + d) * (a + c) * (b + d))
function eval_phicoef(data::ContingencyTable)
    cont_tbl = extract_cached_data(data.con_tbl)
    phicoef = (cont_tbl.a .* cont_tbl.d .- cont_tbl.b .* cont_tbl.c) ./ sqrt.((cont_tbl.a .+ cont_tbl.b) .* (cont_tbl.c .+ cont_tbl.d) .* (cont_tbl.a .+ cont_tbl.c) .* (cont_tbl.b .+ cont_tbl.d))
end

const phi = eval_phicoef
const φ = eval_phicoef


# Cramers V
# "Cramers V", sqrt(chi2 / (N * (min(k, l) - 1)))
function eval_cramersv(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    cramersv = sqrt.(chi_square(data) ./ (con_tbl.N * (min.(con_tbl.k, con_tbl.l) .- 1)))
end

const cramersv = eval_cramersv


# Tschuprow's T
# "Tschuprow's T", sqrt(chi2 / (N * (min(k, l) - 1)))

function eval_tschuprowt(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    tschuprowt = sqrt.(chi_square(data) ./ (con_tbl.N .* (min.(con_tbl.k, con_tbl.l) .- 1)))
end

const tschuprowt = eval_tschuprowt


# Contingency Coefficient
# "Contingency Coefficient", sqrt(chi2 / (chi2 + N))
function eval_contcoef(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    contcoef = sqrt.(chi_square(data) ./ (chi_square(data) .+ con_tbl.N))
end

const contcoef = eval_contcoef


# Cosine Similarity
# "Cosine Similarity", a / sqrt((a + b) * (a + c))
function eval_cosinesim(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    cosinesim = con_tbl.a ./ sqrt.((con_tbl.a .+ con_tbl.b) .* (con_tbl.a .+ con_tbl.c))
end

const cosinesim = eval_cosinesim


# Overlap Coefficient
# "Overlap Coefficient", a / min(m, k)
function eval_overlapcoef(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    overlapcoef = con_tbl.a ./ min(con_tbl.m, con_tbl.k)
end

const overlapcoef = eval_overlapcoef


# Kulczynski Similarity
# "Kulczynski Similarity", a / ((k + m) / 2)
function eval_kulczynskisim(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    kulczynskisim = con_tbl.a ./ ((con_tbl.k .+ con_tbl.m) ./ 2)
end

const kulczynskisim = eval_kulczynskisim


# Tanimoto Coefficient
# "Tanimoto Coefficient", a / (k + m - a)
function eval_tanimotocoef(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    tanimotocoef = con_tbl.a ./ (con_tbl.k .+ con_tbl.m .- con_tbl.a)
end

const tanimotocoef = eval_tanimotocoef


# Hamann Similarity
# "Hamann Similarity", (a + d - b - c) / N
function eval_hamannsim(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    hamannsim = (con_tbl.a .+ con_tbl.d .- con_tbl.b .- con_tbl.c) ./ con_tbl.N
end

const hamannsim = eval_hamannsim

# Goodman-Kruskal Index
# "Goodman-Kruskal Index", (a * d - b * c) / (a * d + b * c)
function eval_goodmankruskalindex(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    goodmankruskalindex = (con_tbl.a .* con_tbl.d .- con_tbl.b .* con_tbl.c) ./ (con_tbl.a .* con_tbl.d .+ con_tbl.b .* con_tbl.c)
end

const goodmankruskalindex = eval_goodmankruskalindex


# Gower's Coefficient
# "Gower's Coefficient", (a + d) / (a + d + 2 * (b + c))
function eval_gowercoef(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    gowercoef = (con_tbl.a .+ con_tbl.d) ./ (con_tbl.a .+ con_tbl.d .+ 2 .* (con_tbl.b .+ con_tbl.c))
end

const gowercoef = eval_gowercoef


# Czekanowski-Dice Coefficient
# "Czekanowski-Dice Coefficient", 2 * a / (2 * a + b + c)
function eval_czekanowskidicecoef(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    czekanowskidicecoef = 2 .* con_tbl.a ./ (2 .* con_tbl.a .+ con_tbl.b .+ con_tbl.c)
end

const czekanowskidicecoef = eval_czekanowskidicecoef

# Sorgenfrey Index
# "Sorgenfrey Index", (a + d) / (2 * (a + d) + b + c)
function eval_sorgenfreyindex(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    sorgenfreyindex = (con_tbl.a .+ con_tbl.d) ./ (2 .* (con_tbl.a .+ con_tbl.d) .+ con_tbl.b .+ con_tbl.c)
end

const sorgenfreyindex = eval_sorgenfreyindex


# Mountford's Coefficient
# "Mountford's Coefficient", (a + d) / (a + d + 2 * sqrt((b + c) * (k + m)))
function eval_mountfordcoef(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    mountfordcoef = (con_tbl.a .+ con_tbl.d) ./ (con_tbl.a .+ con_tbl.d .+ 2 .* sqrt.((con_tbl.b .+ con_tbl.c) .* (con_tbl.k .+ con_tbl.m)))
end

const mountfordcoef = eval_mountfordcoef

# Sokal-Sneath Index
# "Sokal-Sneath Index", a / (a + 2 *
function eval_sokalsneathindex(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    sokalsneathindex = con_tbl.a ./ (con_tbl.a .+ 2 .* (con_tbl.b .+ con_tbl.c))
end

const sokalsneathindex = eval_sokalsneathindex


# Rogers-Tanimoto Coefficient
# "Rogers-Tanimoto Coefficient", a / (a + 2 * (b + c))
function eval_rogerstanimotocoef(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    rogerstanimotocoef = con_tbl.a ./ (con_tbl.a .+ 2 .* (con_tbl.b .+ con_tbl.c))
end

const rogerstanimotocoef = eval_rogerstanimotocoef


# Sokal-Michener Coefficient
# "Sokal-Michener Coefficient", (a + d) / (a + d + 2 * (
function eval_sokalmichenercoef(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    sokalmichenercoef = (con_tbl.a .+ con_tbl.d) ./ (con_tbl.a .+ con_tbl.d .+ 2 .* (con_tbl.b .+ con_tbl.c))
end

const sokalmichenercoef = eval_sokalmichenercoef


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

# Define a general evaluate function for the AssociationMetric type

metrics = (PMI, PMI2, PMI3, PPMI, LLR, DeltaPi, Dice, LogDice, RelRisk, LogRelRisk, RiskDiff, AttrRisk, OddsRatio, LogRatio, LogOddsRatio, JaccardIndex, OchiaiIndex, OchiaiCoef, PiatetskyShapiro, YuleQ, YuleY, PhiCoef, CramersV, TschuprowT, ContCoef, CosineSim, OverlapCoef, KulczynskiSim, TanimotoCoef, GoodmanKruskalIndex, GowerCoef, CzekanowskiDiceCoef, SorgenfreyIndex, MountfordCoef, SokalSneathIndex, RogersTanimotoCoef, SokalmMchenerCoef, Tscore, Zscore, ChiSquare, FisherExactTest)

for M in metrics
    eval_fn_symbol = Symbol("eval_", lowercase(string(M)))
    @eval begin
        @inline function evalassoc(::$M, data::ContingencyTable)
            invoke($(eval_fn_symbol), Tuple{ContingencyTable}, data)
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