
# log2((a / N) / ((k / N) * (m / N)))
function eval_pmi(data::ContingencyTable)
    con_tbl = TextAssociations.extract_cached_data(data.con_tbl)
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

# Classic LLR: 2 * (a * log(a / E11) + b * log(b / E12) + c * log(c / E21) + d * log(d / E22))
function eval_llr(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    2 * (con_tbl.a .* (log.(con_tbl.a) .- log.(con_tbl.E₁₁)) .+ con_tbl.b .* (log.(con_tbl.b) .- log.(con_tbl.E₁₂)) .+ con_tbl.c .* (log.(con_tbl.c) .- log.(con_tbl.E₂₁)) .+ con_tbl.d .* (log.(con_tbl.d) .- log.(con_tbl.E₂₂)))
end

const llr = eval_llr

# LLR2: 2 * (a * log(a) - (a + b) * log(a + b) + c * log(c) - (c + d) * log(c + d))
function eval_llr2(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # implementation
end

const llr2 = eval_llr2

# squared LLR: 
function eval_llr²(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    2 * (con_tbl.a .* (log.(con_tbl.a) .- log.(con_tbl.E₁₁)) .+ con_tbl.b .* (log.(con_tbl.b) .- log.(con_tbl.E₁₂)) .+ con_tbl.c .* (log.(con_tbl.c) .- log.(con_tbl.E₂₁)) .+ con_tbl.d .* (log.(con_tbl.d) .- log.(con_tbl.E₂₂)))
end

const llr² = eval_llr²

# deltapi: \Delta \pi = \frac{a}{a + b} - \frac{c}{c + d}
function eval_deltapi(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    (con_tbl.a ./ con_tbl.m) .- (con_tbl.c ./ con_tbl.n)
end

const deltapi = eval_deltapi
const δπ = eval_deltapi


# minimum sensitivity: \text{Min. Sensitivity} = \min\left(\frac{a}{a + b}, \frac{d}{c + d}\right)
function eval_minsensitivity(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    min(con_tbl.a ./ con_tbl.m, con_tbl.d ./ con_tbl.n)
end

const minsen = eval_minsensitivity


# a + b  = m, c + d = n, a + c = k, b + d = l

# Dice
function eval_dice(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    (2 .* con_tbl.a) ./ (con_tbl.m + con_tbl.k)
end

const dice = eval_dice


# Log Dice: \text{Log Dice} = 14 + \log_2\left(\frac{2a}{2a + b + c}\right)
function eval_logdice(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    14 .+ log2.((2 .* con_tbl.a) ./ (con_tbl.m + con_tbl.k))
end

const logdice = eval_logdice


# Relative Risk: \text{Relative Risk} = \frac{\frac{a}{a + b}}{\frac{c}{c + d}} 
function eval_relrisk(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    (con_tbl.a ./ con_tbl.m) ./ (con_tbl.c ./ con_tbl.n)
end

const relrisk = eval_relrisk
const rr = eval_relrisk


# Log Relative Risk: \text{Log Relative Risk} = \log\left(\frac{\frac{a}{a + b}}{\frac{c}{c + d}}\right)
function eval_logrelrisk(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    log.(con_tbl.a ./ con_tbl.m) .- log.(con_tbl.c ./ con_tbl.n)
end

const logrelrisk = eval_logrelrisk
const lrr = eval_logrelrisk


# Risk Difference: \frac{a}{a + b} - \frac{c}{c + d}
function eval_riskdiff(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    (con_tbl.a ./ con_tbl.m) .- (con_tbl.c ./ con_tbl.n)
end

const riskdiff = eval_riskdiff
const rd = eval_riskdiff


# Attributable Risk: \frac{a}{a + b} - \frac{c}{c + d}
function eval_attrrisk(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    (con_tbl.a ./ con_tbl.m) .- (con_tbl.c ./ con_tbl.n)
end

const atrisk = eval_attrrisk
const ar = eval_attrrisk


# Odds Ratio: \frac{a \cdot d}{b \cdot c}
function eval_oddsratio(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # oddsratio = (con_tbl.a ./ con_tbl.b) ./ (con_tbl.c ./ con_tbl.d)
    (con_tbl.a .* con_tbl.d) ./ (con_tbl.b .* con_tbl.c)
end

const oddsratio = eval_oddsratio
const or = eval_oddsratio


# Log Odds Ratio: \log\left(\frac{a \cdot d}{b \cdot c}\right)
function eval_logoddsratio(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # logoddsratio = log.(con_tbl.a ./ con_tbl.b) .- log.(con_tbl.c ./ con_tbl.d)
    log.(con_tbl.a .* con_tbl.d) .- log.(con_tbl.b .* con_tbl.c)
end

const logoddsratio = eval_logoddsratio
const lor = eval_logoddsratio


# Jaccard Index: \frac{a}{a + b + c}
# "Jaccard", a/(a + b + c)
function eval_jaccardindex(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    con_tbl.a ./ (con_tbl.a .+ con_tbl.b .+ con_tbl.c)
end

const jaccard = eval_jaccardindex


# Ochiai Index
# "Ochiai", a / sqrt((a + b) * (a + c))
function eval_ochiaiindex(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    con_tbl.a ./ sqrt.((con_tbl.a .+ con_tbl.b) .* (con_tbl.a .+ con_tbl.c))
end

const ochiai = eval_ochiaiindex

# Piatetsky Shapiro
# "Piatetsky Shapiro", \frac{a}{n} - \frac{(a + b)(a + c)}{n^2}
function eval_piatetskyshapiro(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    con_tbl.a .- ((con_tbl.k .* con_tbl.m) ./ con_tbl.N)
end

const piatetskyshapiro = eval_piatetskyshapiro


# Yule's Omega (ω) Coefficient
# "Yule's Omega", sqrt((a * d) - (b * c)) / sqrt((a * d) + (b * c))
function eval_yuleomega(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    (sqrt.((con_tbl.a .* con_tbl.d)) .- sqrt.(con_tbl.b .* con_tbl.c)) ./ (sqrt.(con_tbl.a .* con_tbl.d) .+ sqrt.(con_tbl.b .* con_tbl.c))
end

const yuleomega = eval_yuleomega

# Yule's Q  Coefficient
# "Yule's Q", sqrt((a * d) - (b * c)) / sqrt((a * d) + (b * c)) , \frac{a \cdot d - b \cdot c}{a \cdot d + b \cdot c}
function eval_yuleq(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    ((con_tbl.a .* con_tbl.d) .- (con_tbl.b .* con_tbl.c)) ./ ((con_tbl.a .* con_tbl.d) .+ (con_tbl.b .* con_tbl.c))
end

const yuleq = eval_yuleq

# Phi Coefficient
# (a * d - b * c) / sqrt((a + b) * (c + d) * (a + c) * (b + d))
function eval_phicoef(data::ContingencyTable)
    cont_tbl = extract_cached_data(data.con_tbl)
    (cont_tbl.a .* cont_tbl.d .- cont_tbl.b .* cont_tbl.c) ./ sqrt.((cont_tbl.a .+ cont_tbl.b) .* (cont_tbl.c .+ cont_tbl.d) .* (cont_tbl.a .+ cont_tbl.c) .* (cont_tbl.b .+ cont_tbl.d))
end

const phi = eval_phicoef
const φ = eval_phicoef


# Cramers V
# "Cramers V", sqrt(chi2 / (N * (min(k, l) - 1))), \sqrt{\frac{\phi^2}{\min(1, 1)}} = \sqrt{\phi^2} = \|\phi\|
function eval_cramersv(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    sqrt.(chi_square(data) ./ (con_tbl.N * (min.(con_tbl.k, con_tbl.l) .- 1)))
end

const cramersv = eval_cramersv


# Tschuprow's T
# "Tschuprow's T", sqrt(chi2 / (N * (min(k, l) - 1))), \sqrt{\frac{\chi^2}{n \cdot \sqrt{(k - 1)(r - 1)}}}

function eval_tschuprowt(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    sqrt.(chi_square(data) ./ (con_tbl.N .* (min.(con_tbl.k, con_tbl.l) .- 1)))
end

const tschuprowt = eval_tschuprowt


# Contingency Coefficient
# "Contingency Coefficient", sqrt(chi2 / (chi2 + N)), \sqrt{\frac{\chi^2}{\chi^2 + n}}
function eval_contcoef(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    sqrt.(chi_square(data) ./ (chi_square(data) .+ con_tbl.N))
end

const contcoef = eval_contcoef


# Cosine Similarity
# "Cosine Similarity", a / sqrt((a + b) * (a + c)), \frac{a}{\sqrt{(a + b)(a + c)}}
function eval_cosinesim(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    con_tbl.a ./ sqrt.((con_tbl.a .+ con_tbl.b) .* (con_tbl.a .+ con_tbl.c))
end

const cosinesim = eval_cosinesim


# Overlap Coefficient
# "Overlap Coefficient", a / min(m, k), \frac{a}{\min(a + b, a + c)}
function eval_overlapcoef(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    con_tbl.a ./ min(con_tbl.m, con_tbl.k)
end

const overlapcoef = eval_overlapcoef


# Kulczynski Similarity
# "Kulczynski Similarity", a / ((k + m) / 2), \frac{a}{a + b} + \frac{a}{a + c}
function eval_kulczynskisim(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    con_tbl.a ./ ((con_tbl.k .+ con_tbl.m) ./ 2)
end

const kulczynskisim = eval_kulczynskisim

# Tanimoto Coefficient
# "Tanimoto Coefficient", a / (k + m - a), \frac{a}{a + b + c}
function eval_tanimotocoef(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    con_tbl.a ./ (con_tbl.k .+ con_tbl.m .- con_tbl.a)
end

const tanimotocoef = eval_tanimotocoef

# Rogers-Tanimoto Coefficient  (traditional)
# "Rogers-Tanimoto Coefficient", \frac{a}{a + 2(b + c)}
function eval_rogerstanimotocoef(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)a
    # TODO
end

const rogerstanimotocoef = eval_rogerstanimotocoef


# Rogers-Tanimoto Coefficient (incorporates the frequency of occurrences where neither of the events occurs, making it more inclusive in certain scenarios)
# "Rogers-Tanimoto Coefficient", (a + d) / (a + 2 * (b + c) + d)
function eval_rogerstanimotocoef2(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)a
    (con_tbl.a .+ con_tbl.d) ./ (con_tbl.a .+ 2 .* (con_tbl.b .+ con_tbl.c) + con_tbl.d)
end

const rogerstanimotocoef2 = eval_rogerstanimotocoef2

# Hamann Similarity
# "Hamann Similarity", \frac{a + d - b - c}{N}
function eval_hamannsim(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    (con_tbl.a .+ con_tbl.d .- con_tbl.b .- con_tbl.c) ./ con_tbl.N
end

const hamannsim = eval_hamannsim

# Hamann Similarity 2
# "Hamann Similarity", \frac{a - d}{a + b + c - d}
function eval_hamannsim2(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # TODO
end

const hamannsim2 = eval_hamannsim2

# Goodman-Kruskal Index
# "Goodman-Kruskal Index", (a * d - b * c) / (a * d + b * c)
function eval_goodmankruskalindex(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    (con_tbl.a .* con_tbl.d .- con_tbl.b .* con_tbl.c) ./ (con_tbl.a .* con_tbl.d .+ con_tbl.b .* con_tbl.c)
end

const goodmankruskalidx = eval_goodmankruskalindex


# Gower's Coefficient (traditional) \frac{a}{a + b + c}
# "Gower's Coefficient", 
function eval_gowercoef(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # TODO
end

const gowercoef = eval_gowercoef

# Gower's Coefficient, \frac{a + d}{a + d + 2(b + c)}
# "Gower's Coefficient", (a + d) / (a + d + 2 * (b + c))
function eval_gowercoef2(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    (con_tbl.a .+ con_tbl.d) ./ (con_tbl.a .+ con_tbl.d .+ 2 .* (con_tbl.b .+ con_tbl.c))
end

const gowercoef2 = eval_gowercoef2


# Czekanowski-Dice Coefficient, \frac{2a}{2a + b + c}
# "Czekanowski-Dice Coefficient", 2 * a / (2 * a + b + c)
function eval_czekanowskidicecoef(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    czekanowskidicecoef = 2 .* con_tbl.a ./ (2 .* con_tbl.a .+ con_tbl.b .+ con_tbl.c)
end

const czekanowskidicecoef = eval_czekanowskidicecoef

# Sorgenfrey Index (traditional) \frac{2a - b - c}{2a + b + c}
# "Sorgenfrey Index",
function eval_sorgenfreyindex(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # TODO
end

const sorgenfreydx = eval_sorgenfreyindex

# Sorgenfrey Index
# "Sorgenfrey Index", (a + d) / (2 * (a + d) + b + c)
function eval_sorgenfreyindex2(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    (con_tbl.a .+ con_tbl.d) ./ (2 .* (con_tbl.a .+ con_tbl.d) .+ con_tbl.b .+ con_tbl.c)
end

const sorgenfreyidx2 = eval_sorgenfreyindex2

# Mountford's Coefficient (traditional) \frac{a}{a + 2b + 2c}
# "Mountford's Coefficient", 
function eval_mountfordcoef(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # TODO
end

const mountfordcoef = eval_mountfordcoef

# Mountford's Coefficient 2 (alternative)
# "Mountford's Coefficient", (a + d) / (a + d + 2 * sqrt((b + c) * (k + m)))
function eval_mountfordcoef2(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    (con_tbl.a .+ con_tbl.d) ./ (con_tbl.a .+ con_tbl.d .+ 2 .* sqrt.((con_tbl.b .+ con_tbl.c) .* (con_tbl.k .+ con_tbl.m)))
end

const mountfordcoef2 = eval_mountfordcoef2

# Sokal-Sneath Index, \frac{a}{a + 2b + 2c}
# "Sokal-Sneath Index", a / (a + 2 * (b + c))
function eval_sokalsneathindex(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    con_tbl.a ./ (con_tbl.a .+ 2 .* (con_tbl.b .+ con_tbl.c))
end

const sokalsneathidx = eval_sokalsneathindex

# Sokal-Michener Coefficient
# "Sokal-Michener Coefficient", DONE
function eval_sokalmichenercoef(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    sokalmichenercoef = (con_tbl.a .+ con_tbl.d) ./ con_tbl.N
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

function eval_jointprob(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # implementation
end

const jointprob = eval_jointprob

function eval_conditionalprob(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # implementation
end

const conditionalprob = eval_conditionalprob

function eval_revconditionalprob(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # implementation
end

const revconditionalprob = eval_revconditionalprob

function eval_mutualdep(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # implementation
end

const mutualdep = eval_mutualdep

function eval_logfreqmd(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # implementation
end

const logfreqmd = eval_logfreqmd

function eval_normexpectation(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # implementation
end

const normexpectation = eval_normexpectation

function eval_mutualexpectation(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # implementation
end

const mutualexpectation = eval_mutualexpectation

function eval_salience(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # implementation
end

const salience = eval_salience

function eval_pearsonchi2(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # implementation
end

const pearsonchi2 = eval_pearsonchi2

function eval_ttest(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # implementation
end

const ttest = eval_ttest

function eval_zscore(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # implementation
end

const zscore = eval_zscore

function eval_klosgen(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # implementation
end

const klosgen = eval_klosgen

function eval_russellrao(data::ContingencyTable) # DONE
    con_tbl = extract_cached_data(data.con_tbl)
    con_tbl.a ./ con_tbl.N
end

const russellrao = eval_russellrao

function eval_kulczynsky1(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # implementation
end

const kulczynsky1 = eval_kulczynsky1

function eval_driverkroeber(data::ContingencyTable) # DONE
    con_tbl = extract_cached_data(data.con_tbl)
    driverkroeber = con_tbl.a ./ sqrt.((con_tbl.a .+ con_tbl.b) .* (con_tbl.a .+ con_tbl.c))
end

const driverkroeber = eval_driverkroeber

function eval_pearsoncor(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # implementation
end

const pearsoncor = eval_pearsoncor

function eval_baroniurbani(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # implementation
end

const baroniurbani = eval_baroniurbani

function eval_braunblanquet(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # implementation
end

const braunblanquet = eval_braunblanquet

function eval_simpsonidx(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # implementation
end

const simpsonidx = eval_simpsonidx

function eval_michaelidx(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # implementation
end

const michaelidx = eval_michaelidx

function eval_fageridx(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # implementation
end

const fageridx = eval_fageridx

function eval_unisubtypes(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # implementation
end

const unisubtypes = eval_unisubtypes

function eval_ucost(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # implementation
end

const ucost = eval_ucost

function eval_rcost(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # implementation
end

const rcost = eval_rcost

function eval_scost(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # implementation
end

const scost = eval_scost

function eval_tcombcost(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # implementation
end

const tcombcost = eval_tcombcost

function eval_jmeasure(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # implementation
end

const jmeasure = eval_jmeasure

function eval_giniidx(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # implementation
end

const giniidx = eval_giniidx

function eval_confmeasure(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # implementation
end

const confmeasure = eval_confmeasure

function eval_laplace(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # implementation
end

const laplace = eval_laplace

function eval_convictmeasure(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # implementation
end

const convictmeasure = eval_convictmeasure

function eval_certaintymeasure(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # implementation
end

const certaintymeasure = eval_certaintymeasure

function eval_addedvaluemeasure(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # implementation
end

const addedvaluemeasure = eval_addedvaluemeasure

function eval_collectivestrength(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # implementation
end

const collectivestrength = eval_collectivestrength

function eval_cohenskappa(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # implementation
end

const cohenskappa = eval_cohenskappa
const kappa = eval_cohenskappa

# Joint Probability: joint_probability
# Conditional Probability: conditional_probability
# Reverse Conditional Probability: reverse_conditional_probability
# Mutual Dependency (MD): mutual_dependency
# Log Frequency Biased MD: log_freq_biased_md
# Normalized Expectation: normalized_expectation
# Mutual Expectation: mutual_expectation
# Salience: salience_measure
# Pearson's Chi-Squared Test: pearson_chi2_test
# T Test: t_test
# Z Score: z_score
# Squared Log Likelihood Ratio: squared_log_likelihood_ratio
# Klosgen: klosgen_measure
# Russell-Rao: russell_rao
# Sokal-Michener: sokal_michener
# First Kulczynsky: first_kulczynsky
# Yule's Omega (ω): yules_omega
# Driver-Kroeber: driver_kroeber
# Pearson (for correlation): pearson_correlation
# Baroni-Urbani: baroni_urbani
# Braun-Blanquet: braun_blanquet
# Simpson: simpson_index
# Michael: michael_index
# Fager: fager_index
# Unigram Subtuples: unigram_subtuples
# U Cost: u_cost
# S Cost: s_cost
# R Cost: r_cost
# T Combined Cost: t_combined_cost
# Kappa: kappa
# J-Measure: j_measure
# Gini Index: gini_index
# Confidence: confidence_measure
# Laplace: laplace_measure
# Conviction: conviction_measure
# Certainty: certainty_measure
# Added Value: added_value_measure
# Collective Strength: collective_strength
# ========================================

# check if the following are included in the package
# Poisson Significance Measure: poisson_significance_measure
# Hamann: hamann

# Define a general evaluate function for the AssociationMetric type

# metrics = (PMI, PMI2, PMI3, PPMI, LLR, DeltaPi, Dice, LogDice, RelRisk, LogRelRisk, RiskDiff, AttrRisk, OddsRatio, LogRatio, LogOddsRatio, JaccardIndex, OchiaiIndex, OchiaiCoef, PiatetskyShapiro, YuleQ, YuleY, PhiCoef, CramersV, TschuprowT, ContCoef, CosineSim, OverlapCoef, KulczynskiSim, TanimotoCoef, GoodmanKruskalIndex, GowerCoef, CzekanowskiDiceCoef, SorgenfreyIndex, MountfordCoef, SokalSneathIndex, RogersTanimotoCoef, SokalmMchenerCoef, Tscore, Zscore, ChiSquare, FisherExactTest)

# for M in metrics
#     eval_fn_symbol = Symbol("eval_", lowercase(string(M)))
#     @eval begin
#         @inline function evalassoc(::Type{$M}, data::ContingencyTable)
#             invoke($(eval_fn_symbol), Tuple{ContingencyTable}, data)
#         end
#     end
# end
"""
    evalassoc(metricType::Type{<:AssociationMetric}, data::ContingencyTable)

Evaluate an association metric based on the provided metric type and a contingency table. This function dynamically dispatches the calculation to the appropriate function determined by `metricType`.

# Usage

```julia-doc
result = evalassoc(MetricType, data)
```

Replace `MetricType` with the desired association metric type (e.g., `PMI`, `Dice`) and data with your contingency table.

# Examples
**PMI (Pointwise Mutual Information)**:

```julia-doc
result = evalassoc(PMI, data)
```

**Dice Coefficient**:

```julia-doc
result = evalassoc(Dice, data)
```

You can see all supported metrics through `listmetrics()`.

# Further Reading

For detailed mathematical definitions and discussions on each metric, refer to our documentation site.

# Tips

Ensure your data is a `ContingencyTable` instance object. To create one, use the `ContingencyTable` constructor. 
"""
function evalassoc(metricType::Type{<:AssociationMetric}, data::ContingencyTable)
    func_name = Symbol("eval_", lowercase(string(metricType)))  # Construct function name
    func = getfield(@__MODULE__, func_name)  # Get the function from the current module
    return func(data)  # Call the function
end


# OverlapCoefficient

# List all available statistical assoiation measures from Pecina's paper

# 1. Joint probability
# 2. Conditional probability 
# 3. Reverse conditional probability

# 5. Mutual dependency $(M D)
# 6. Log frequency biased $M D$
# 7. Normalized expectation
# 8. Mutual expectation
# 9. Salience 
# 10. Pearson's chi^2 test

# 12. t test
# 13. z score
# 14. Poison significance measure 

# 16. Squared log likelihood ratio 

# 23. First Kulczynsky

# 32. Pearson 
# 33. Baroni-Urbani
# 34. Braun-Blanquet
# 35. Simpson 
# 36. Michael 

# 38. Fager 
# 39. Unigram subtuples 
# 40. $U$ cost 
# 41. $S$ cost 
# 42. $R$ cost 
# 43. $T$ combined cost

# 45. Kappa
# 45. J-measure

# 47. Gini index
# 48. Confidence
# 49. Laplace
# 50. Conviction

# 52. Certainity
# 53. Added value
# 54. Collective
# 55. Klosgen

# DONE or INCLUDED in the package

# 28. Yule's $\omega$ DONE
# 30. Driver-Kroeber DONE
# 18. Sokal-Michener DONE
# 17. Russel-Rao DONE
# 20. Hamann DONE
# 4. Pointwise mutual information DONE
# 15. Log likelihood ratio DONE
# 22. Jaccard DONE
# 44. Phi coefficient DONE
# 51. Piatersky-Shapiro DONE
# 27. Odds ratio DONE
# 29. Yule's $Q$ DONE
# 31. Fifth Sokal-Sneath INCLUDED
# 11. Fisher's exact test INCLUDED
# 19. Rogers-Tanimoto DONE
# 21. Third Sokal-Sneath INCLUDED
# 24. Second Sokal-Sneath INCLUDED
# 25. Second Kulczynski INCLUDED
# 26. Fourth Sokal-Sneath INCLUDED
# 37. Mountford INCLUDED
# 46. Gower INCLUDED


