# =====================================
# File: src/metrics/measures.jl
# Additional metrics (PMI, LLR, etc.)
# =====================================

# log2((a / N) / ((k / N) * (m / N)))
# Pointwise Mutual Information
function eval_pmi(data::AssociationDataFormat)
    @extract_values data a N k m
    # PMI = log( (a/N) / ((k/N)*(m/N)) ) = log(a) - log(N) - (log(k) + log(m) - log(N))
    log_safe.(a) .- log_safe.(N) .- (log_safe.(k) .+ log_safe.(m) .- log_safe.(N))
end

const pmi = eval_pmi


# log2((a^2 / N) / ((k / N) * (m / N)))
# Pointwise Mutual Information²
function eval_pmi²(data::AssociationDataFormat)
    @extract_values data a N k m
    2 .* log_safe.(a) .- log_safe.(N) .- (log_safe.(k) .+ log_safe.(m) .- log_safe.(N))
end

const pmi² = eval_pmi²


# log2((a^3 / N) / ((k / N) * (m / N)))
# Pointwise Mutual Information³
function eval_pmi³(data::AssociationDataFormat)
    @extract_values data a N k m
    3 .* log_safe.(a) .- log_safe.(N) .- (log_safe.(k) .+ log_safe.(m) .- log_safe.(N))
end

const pmi³ = eval_pmi³

function eval_ppmi(data::AssociationDataFormat)
    @extract_values data a N k m
    max.(0, log_safe.(a) .- log_safe.(N) .- (log_safe.(k) .+ log_safe.(m) .- log_safe.(N)))
end

const ppmi = eval_ppmi

# Classic LLR: 2 * (a * log(a / E11) + b * log(b / E12) + c * log(c / E21) + d * log(d / E22))
# LLR=2×[(alog(a)+blog(b)+clog(c)+dlog(d))−(alog(E11)+blog(E12​)+clog(E21)+dlog(E22))]
function eval_llr(data::AssociationDataFormat)
    @extract_values data a b c d E₁₁ E₁₂ E₂₁ E₂₂

    # Compute the terms
    observed_term = a .* log_safe.(a) .+ b .* log_safe.(b) .+ c .* log_safe.(c) .+ d .* log_safe.(d)
    expected_term = a .* log_safe.(E₁₁) .+ b .* log_safe.(E₁₂) .+ c .* log_safe.(E₂₁) .+ d .* log_safe.(E₂₂)

    # LLR formula
    llr = 2 * (observed_term .- expected_term)
    return llr
end

const llr = eval_llr

# LLR2: 2 * (a * log(a) - (a + b) * log(a + b) + c * log(c) - (c + d) * log(c + d))
# LLR2=2(a⋅log(a)−(a+b)⋅log(a+b)+c⋅log(c)−(c+d)⋅log(c+d))
function eval_llr²(data::AssociationDataFormat)
    # con_tbl = cached_data(data.con_tbl)
    # Extract individual components for readability
    # a, b, c, d = con_tbl.a, con_tbl.b, con_tbl.c, con_tbl.d
    # E₁₁, E₁₂, E₂₁, E₂₂ = con_tbl.E₁₁, con_tbl.E₁₂, con_tbl.E₂₁, con_tbl.E₂₂
    @extract_values data a b c d E₁₁ E₁₂ E₂₁ E₂₂

    # Compute the observed and expected terms using log_safe
    observed_term = a .* log_safe.(a) .+ b .* log_safe.(b) .+ c .* log_safe.(c) .+ d .* log_safe.(d)
    expected_term = a .* log_safe.(E₁₁) .+ b .* log_safe.(E₁₂) .+ c .* log_safe.(E₂₁) .+ d .* log_safe.(E₂₂)

    # Compute the squared LLR
    2 * (observed_term .- expected_term) .^ 2
end

const llr² = eval_llr²

# deltapi: \Delta \pi = \frac{a}{a + b} - \frac{c}{c + d}
function eval_deltapi(data::AssociationDataFormat)
    # con_tbl = cached_data(data.con_tbl)
    # a, c, m, n = con_tbl.a, con_tbl.c, con_tbl.m, con_tbl.n
    @extract_values data a c m n

    # Avoid division by zero by ensuring denominators are never zero
    denom1 = m .+ eps()
    denom2 = n .+ eps()

    (a ./ denom1) .- (c ./ denom2)
end

const deltapi = eval_deltapi
const δπ = eval_deltapi


# minimum sensitivity: \text{Min. Sensitivity} = \min\left(\frac{a}{a + b}, \frac{d}{c + d}\right)
function eval_minsensitivity(data::AssociationDataFormat)
    # con_tbl = cached_data(data.con_tbl)
    # a, d, m, n = con_tbl.a, con_tbl.d, con_tbl.m, con_tbl.n
    @extract_values data a d m n

    # Avoid division by zero by ensuring denominators are never zero
    denom1 = m .+ eps()
    denom2 = n .+ eps()

    # Compute sensitivities
    sensitivity1 = a ./ denom1
    sensitivity2 = d ./ denom2

    # Compute the minimum sensitivity
    min.(sensitivity1, sensitivity2)
end

const minsen = eval_minsensitivity


# a + b  = m, c + d = n, a + c = k, b + d = l

# Dice f Co-occurrence based word association
function eval_dice(data::AssociationDataFormat)
    # Extract individual components for readability and performance
    # con_tbl = cached_data(data.con_tbl)
    # a, m, k = con_tbl.a, con_tbl.m, con_tbl.k
    @extract_values data a m k

    # Avoid division by zero by ensuring the denominator is never zero
    denom = m .+ k .+ eps()

    # Compute Dice coefficient
    (2 .* a) ./ denom
end

const dice = eval_dice


# Log Dice: \text{Log Dice} = 14 + \log_2\left(\frac{2a}{2a + b + c}\right)
function eval_logdice(data::AssociationDataFormat)
    # con_tbl = cached_data(data.con_tbl)
    # a, m, k = con_tbl.a, con_tbl.m, con_tbl.k
    @extract_values data a m k

    # Compute Log Dice using logarithmic properties
    14 .+ log2_safe.(2 .* a) .- log2_safe.(m .+ k)
end

const logdice = eval_logdice


# Relative Risk: \text{Relative Risk} = \frac{\frac{a}{a + b}}{\frac{c}{c + d}}
#  https://www.ncbi.nlm.nih.gov/books/NBK430824/figure/article-28324.image.f1/
function eval_relrisk(data::AssociationDataFormat)
    # con_tbl = cached_data(data.con_tbl)
    # a, c, m, n = con_tbl.a, con_tbl.c, con_tbl.m, con_tbl.n
    @extract_values data a c m n

    # Avoid division by zero
    max.((a .* n) ./ (c .* m), eps())  # Ensure no division leads to invalid values
end

const relrisk = eval_relrisk
const rr = eval_relrisk


# Log Relative Risk: \text{Log Relative Risk} = \log\left(\frac{\frac{a}{a + b}}{\frac{c}{c + d}}\right)
function eval_logrelrisk(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    # Extract individual components
    a, c, m, n = con_tbl.a, con_tbl.c, con_tbl.m, con_tbl.n

    # Use log_safe for stability
    log_safe.(a) .- log_safe.(m) .- log_safe.(c) .+ log_safe.(n)
end

const logrelrisk = eval_logrelrisk
const lrr = eval_logrelrisk


# Risk Difference: \frac{a}{a + b} - \frac{c}{c + d}
# incidence proportion difference 46.6.2 Incidence proportion difference, https://www.r4epi.com/measures-of-association
function eval_riskdiff(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    # Extract individual components
    a, c, m, n = con_tbl.a, con_tbl.c, con_tbl.m, con_tbl.n

    # Ensure stability using max.(_, eps()) to avoid division by zero
    (a ./ max.(m, eps())) .- (c ./ max.(n, eps()))
end

const riskdiff = eval_riskdiff
const rd = eval_riskdiff


# Attributable Risk: \frac{a}{a + b} - \frac{c}{c + d}
function eval_attrrisk(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    # Extract individual components
    a, c, m, n = con_tbl.a, con_tbl.c, con_tbl.m, con_tbl.n

    # Ensure stability using max.(_, eps()) to avoid division by zero
    (a ./ max.(m, eps())) .- (c ./ max.(n, eps()))
end

const atrisk = eval_attrrisk
const ar = eval_attrrisk


# Odds Ratio: \frac{a \cdot d}{b \cdot c}
function eval_oddsratio(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    # Extract individual components
    a, b, c, d = con_tbl.a, con_tbl.b, con_tbl.c, con_tbl.d

    # Ensure stability using max.(_, eps()) to avoid division by zero
    (a .* d) ./ max.(b .* c, eps())
end

const oddsratio = eval_oddsratio
const or = eval_oddsratio


# Log Odds Ratio: \log\left(\frac{a \cdot d}{b \cdot c}\right)
function eval_logoddsratio(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    # Extract individual components
    a, b, c, d = con_tbl.a, con_tbl.b, con_tbl.c, con_tbl.d

    # Compute log odds ratio using log laws and log_safe
    log_safe.(a) .+ log_safe.(d) .- log_safe.(b) .- log_safe.(c)
end

const logoddsratio = eval_logoddsratio
const lor = eval_logoddsratio


# Jaccard Index (for 2×2): a / (k + m - a) == a / (a + b + c)
function eval_jaccardidx(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    a, k, m = con_tbl.a, con_tbl.k, con_tbl.m
    a ./ max.(k .+ m .- a, eps())
end

const jaccardidx = eval_jaccardidx


# Ochiai Index
# "Ochiai", a / sqrt((a + b) * (a + c))
function eval_ochiaiidx(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    # Extract individual components
    a, m, k = con_tbl.a, con_tbl.m, con_tbl.k

    # Compute Ochiai Index
    denominator = sqrt.(max.(m .* k, eps()))
    a ./ denominator
end

const ochiaiidx = eval_ochiaiidx

# Piatetsky Shapiro
# "Piatetsky Shapiro", \frac{a}{n} - \frac{(a + b)(a + c)}{n^2}
function eval_piatetskyshapiro(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    # Extract individual components
    a, N, k, m = con_tbl.a, con_tbl.N, con_tbl.k, con_tbl.m

    # Compute Piatetsky-Shapiro metric
    (a ./ N) .- ((k .* m) ./ (N .* N))
end

const piatetskyshapiro = eval_piatetskyshapiro


# Yule's Omega (ω) Coefficient
# "Yule's Omega", sqrt((a * d) - (b * c)) / sqrt((a * d) + (b * c))
function eval_yuleomega(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    # Extract individual components
    a, b, c, d = con_tbl.a, con_tbl.b, con_tbl.c, con_tbl.d

    # Compute numerator and denominator for Yule's Omega
    numerator = sqrt.((a .* d) .- (b .* c))
    denominator = sqrt.((a .* d) .+ (b .* c))

    # Avoid division by zero by ensuring denominator is always nonzero
    numerator ./ max.(denominator, eps())
end

const yuleomega = eval_yuleomega

# Yule's Q  Coefficient
# "Yule's Q", (a * d) - (b * c)) / (a * d) + (b * c), \frac{a \cdot d - b \cdot c}{a \cdot d + b \cdot c}
function eval_yuleq(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    # Extract individual components
    a, b, c, d = con_tbl.a, con_tbl.b, con_tbl.c, con_tbl.d

    # Compute numerator and denominator for Yule's Q
    numerator = (a .* d) .- (b .* c)
    denominator = (a .* d) .+ (b .* c)

    # Avoid division by zero by ensuring denominator is always nonzero
    numerator ./ max.(denominator, eps())
end

const yuleq = eval_yuleq

# Phi Coefficient
# (a * d - b * c) / sqrt((a + b) * (c + d) * (a + c) * (b + d))
function eval_phicoef(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    a, b, c, d = con_tbl.a, con_tbl.b, con_tbl.c, con_tbl.d
    num = (a .* d) .- (b .* c)
    denom = sqrt.((a .+ b) .* (c .+ d) .* (a .+ c) .* (b .+ d) .+ eps())
    num ./ denom
end

const phi = eval_phicoef
const φ = eval_phicoef


# Cramers V
# "Cramers V", sqrt(chi2 / (N * (min(k, l) - 1))), \sqrt{\frac{\phi^2}{\min(1, 1)}} = \sqrt{\phi^2} = \|\phi\| using the standard formula and not the determinant of the contringency table
function eval_cramersv(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)

    # Extract relevant variables
    a, b, c, d = con_tbl.a, con_tbl.b, con_tbl.c, con_tbl.d
    N = con_tbl.N
    E₁₁, E₁₂, E₂₁, E₂₂ = con_tbl.E₁₁, con_tbl.E₁₂, con_tbl.E₂₁, con_tbl.E₂₂

    # Avoid division by zero by replacing E with eps()
    E₁₁, E₁₂, E₂₁, E₂₂ = max.(E₁₁, eps()), max.(E₁₂, eps()), max.(E₂₁, eps()), max.(E₂₂, eps())

    # Compute chi-squared statistic
    chi2 = ((a .- E₁₁) .^ 2 ./ E₁₁) .+
           ((b .- E₁₂) .^ 2 ./ E₁₂) .+
           ((c .- E₂₁) .^ 2 ./ E₂₁) .+
           ((d .- E₂₂) .^ 2 ./ E₂₂)

    # Compute Cramér's V (k - 1 = 1 for 2x2 tables)
    return sqrt.(chi2 ./ (N .* (2 - 1)))
end


const cramersv = eval_cramersv


# Tschuprow's T
function eval_tschuprowt(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)

    # Extract relevant variables
    a, b, c, d = con_tbl.a, con_tbl.b, con_tbl.c, con_tbl.d
    N = con_tbl.N
    E₁₁, E₁₂, E₂₁, E₂₂ = con_tbl.E₁₁, con_tbl.E₁₂, con_tbl.E₂₁, con_tbl.E₂₂

    # Avoid division by zero for expected values
    E₁₁, E₁₂, E₂₁, E₂₂ = max.(E₁₁, eps()), max.(E₁₂, eps()), max.(E₂₁, eps()), max.(E₂₂, eps())

    # Compute chi-squared statistic
    chi2 = ((a .- E₁₁) .^ 2 ./ E₁₁) .+
           ((b .- E₁₂) .^ 2 ./ E₁₂) .+
           ((c .- E₂₁) .^ 2 ./ E₂₁) .+
           ((d .- E₂₂) .^ 2 ./ E₂₂)

    # Compute Tschuprow’s T (simplified as k = r = 2)
    return sqrt.(chi2 ./ N)
end

const tschuprowt = eval_tschuprowt


# Contingency Coefficient
# "Contingency Coefficient", sqrt(chi2 / (chi2 + N)), \sqrt{\frac{\chi^2}{\chi^2 + n}}
function eval_contcoef(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)

    # Extract relevant variables
    a, b, c, d = con_tbl.a, con_tbl.b, con_tbl.c, con_tbl.d
    N = con_tbl.N
    E₁₁, E₁₂, E₂₁, E₂₂ = con_tbl.E₁₁, con_tbl.E₁₂, con_tbl.E₂₁, con_tbl.E₂₂

    # Avoid division by zero for expected values
    E₁₁, E₁₂, E₂₁, E₂₂ = max.(E₁₁, eps()), max.(E₁₂, eps()), max.(E₂₁, eps()), max.(E₂₂, eps())

    # Compute chi-squared statistic
    chi2 = ((a .- E₁₁) .^ 2 ./ E₁₁) .+
           ((b .- E₁₂) .^ 2 ./ E₁₂) .+
           ((c .- E₂₁) .^ 2 ./ E₂₁) .+
           ((d .- E₂₂) .^ 2 ./ E₂₂)

    # Compute Contingency Coefficient
    return sqrt.(chi2 ./ (chi2 .+ N))
end

const contcoef = eval_contcoef


# Cosine Similarity
# "Cosine Similarity", a / sqrt((a + b) * (a + c)), \frac{a}{\sqrt{(a + b)(a + c)}}
function eval_cosinesim(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)

    # Extract relevant variables
    a, m, k = con_tbl.a, con_tbl.m, con_tbl.k

    # Compute Cosine Similarity
    a ./ sqrt.(m .* k)
end

const cosinesim = eval_cosinesim


# Overlap Coefficient
# "Overlap Coefficient", a / min(m, k), \frac{a}{\min(a + b, a + c)}
function eval_overlapcoef(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)

    # Extract relevant variables
    a, m, k = con_tbl.a, con_tbl.m, con_tbl.k

    # Compute Overlap Coefficient
    a ./ min.(m, k)
end

const overlapcoef = eval_overlapcoef


# Kulczynski Similarity
# "Kulczynski Similarity", a / ((k + m) / 2), \frac{a}{a + b} + \frac{a}{a + c}
function eval_kulczynskisim(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)

    # Extract relevant variables
    a, m, k = con_tbl.a, con_tbl.m, con_tbl.k

    # Compute Kulczynski Similarity
    a ./ ((m .+ k) ./ 2)
end

const kulczynskisim = eval_kulczynskisim

# Tanimoto Coefficient == Jaccard for binary data: a / (k + m - a)
function eval_tanimotocoef(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    con_tbl.a ./ max.(con_tbl.k .+ con_tbl.m .- con_tbl.a, eps())
end

const tanimotocoef = eval_tanimotocoef

# ======================================== from that point on I need to check the implementation (newer chatgpt session with the measures from the screenshots in the metrics folder)

# 1. Joint Probability
function eval_jointprob(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    a, N = con_tbl.a, con_tbl.N
    a ./ N
end

# 2. Conditional Probability
function eval_condprob(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    a, m = con_tbl.a, con_tbl.m
    a ./ m
end

# 3. Reverse Conditional Probability
function eval_reversecondprob(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    a, k = con_tbl.a, con_tbl.k
    a ./ k
end

# 5. Mutual Dependency
# function eval_mutualdependency(data::AssociationDataFormat)
#     con_tbl = cached_data(data.con_tbl)
#     a, N, k, m = con_tbl.a, con_tbl.N, con_tbl.k, con_tbl.m
#     (a ./ N) .^ 2 ./ ((k ./ N) .* (m ./ N))
# end

# # 6. Log Frequency Biased MD
# function eval_logfreqbiasmd(data::AssociationDataFormat)
#     con_tbl = cached_data(data.con_tbl)
#     a, N, k, m = con_tbl.a, con_tbl.N, con_tbl.k, con_tbl.m
#     log_safe.((a ./ N) .^ 2 ./ ((k ./ N) .* (m ./ N))) .+ log_safe.(a ./ N)
# end

# 7. Normalized Expectation
function eval_normalizedexpectation(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    a, N, k, m = con_tbl.a, con_tbl.N, con_tbl.k, con_tbl.m
    (2 .* (a ./ N)) ./ ((m ./ N) .+ (k ./ N))
end

# 8. Mutual Expectation
function eval_mutualexpectation(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    a, N, k, m = con_tbl.a, con_tbl.N, con_tbl.k, con_tbl.m
    (2 .* (a ./ N)) .* (a ./ N) ./ ((m ./ N) .+ (k ./ N))
end

# 9. Salience
function eval_salience(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    a, N, k, m = con_tbl.a, con_tbl.N, con_tbl.k, con_tbl.m
    (a ./ N) .^ 2 ./ ((m ./ N) .* (k ./ N)) .* log_safe.(a ./ N)
end

# 10. Pearson’s Chi² Test
function eval_chi2(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    a, b, c, d, N = con_tbl.a, con_tbl.b, con_tbl.c, con_tbl.d, con_tbl.N
    expected = (a .+ b) .* (a .+ c) ./ max.(N, eps())
    (a .- expected) .^ 2 ./ max.(expected, eps())
end

# 11. Fisher’s Exact Test
# Requires specialized statistical libraries for exact calculation

# 12. t-Test
function eval_ttest(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    a, c, N = con_tbl.a, con_tbl.c, con_tbl.N
    numerator = (a ./ N) .- (c ./ N)
    denominator = sqrt.((a ./ N) .* (1 .- (a ./ N)) ./ N)
    numerator ./ denominator
end

# 13. z-Score
function eval_zscore(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    a, c, N = con_tbl.a, con_tbl.c, con_tbl.N
    numerator = (a ./ N) .- (c ./ N)
    denominator = sqrt.((a ./ N) .* (1 .- a ./ N) ./ N)
    numerator ./ denominator
end

# 14. Poisson Significance Measure
function eval_poissonsig(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    a, m, k, N = con_tbl.a, con_tbl.m, con_tbl.k, con_tbl.N
    f_obs = a
    f_exp = (m .* k) ./ N
    f_obs .* log_safe.(f_obs ./ f_exp) .+ f_exp
end

# # 15. Log Likelihood Ratio
# function eval_llr(data::AssociationDataFormat)
#     con_tbl = cached_data(data.con_tbl)
#     a, b, c, d = con_tbl.a, con_tbl.b, con_tbl.c, con_tbl.d
#     E₁₁, E₁₂, E₂₁, E₂₂ = con_tbl.E₁₁, con_tbl.E₁₂, con_tbl.E₂₁, con_tbl.E₂₂
#     observed_term = a .* log_safe.(a) .+ b .* log_safe.(b) .+ c .* log_safe.(c) .+ d .* log_safe.(d)
#     expected_term = a .* log_safe.(E₁₁) .+ b .* log_safe.(E₁₂) .+ c .* log_safe.(E₂₁) .+ d .* log_safe.(E₂₂)
#     2 * (observed_term .- expected_term)
# end

# # 16. Squared Log Likelihood Ratio
# function eval_llr_squared(data::AssociationDataFormat)
#     eval_llr(data) .^ 2
# end

# Association Coefficients (Examples Below)

# 17. Russel-Rao
function eval_russelrao(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    a, b, c, d = con_tbl.a, con_tbl.b, con_tbl.c, con_tbl.d
    a ./ (a .+ b .+ c .+ d)
end

# 18. Sokal-Michener
function eval_sokalmichener(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    a, d, b, c = con_tbl.a, con_tbl.d, con_tbl.b, con_tbl.c
    (a .+ d) ./ (a .+ b .+ c .+ d)
end

# 19. Rogers-Tanimoto
# function eval_rogerstanimoto(data::AssociationDataFormat)
#     con_tbl = cached_data(data.con_tbl)
#     a, d, b, c = con_tbl.a, con_tbl.d, con_tbl.b, con_tbl.c
#     (a .+ d) ./ (a .+ 2 .* (b .+ c) .+ d)
# end

# # 20. Hamann
# function eval_hamann(data::AssociationDataFormat)
#     con_tbl = cached_data(data.con_tbl)
#     a, d, b, c = con_tbl.a, con_tbl.d, con_tbl.b, con_tbl.c
#     ((a .+ d) .- (b .+ c)) ./ (a .+ b .+ c .+ d)
# end

# # 22. Jaccard
# function eval_jaccard(data::AssociationDataFormat)
#     con_tbl = cached_data(data.con_tbl)
#     a, b, c = con_tbl.a, con_tbl.b, con_tbl.c
#     a ./ (a .+ b .+ c)
# end

# 21. Third Sokal-Sneath
function eval_third_sokalsneath(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    b, c, a, d = con_tbl.b, con_tbl.c, con_tbl.a, con_tbl.d
    (b .+ c) ./ (a .+ d)
end

# 23. First Kulczynski
function eval_first_kulczynski(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    a, b, c = con_tbl.a, con_tbl.b, con_tbl.c
    a ./ (b .+ c)
end

# 24. Second Sokal-Sneath
function eval_second_sokalsneath(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    a, b, c = con_tbl.a, con_tbl.b, con_tbl.c
    a ./ (a .+ 2 .* (b .+ c))
end

# 25. Second Kulczynski
function eval_second_kulczynski(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    a, b, c = con_tbl.a, con_tbl.b, con_tbl.c
    0.5 .* ((a ./ (a .+ b)) .+ (a ./ (a .+ c)))
end

# 26. Fourth Sokal-Sneath
function eval_fourth_sokalsneath(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    a, b, c, d = con_tbl.a, con_tbl.b, con_tbl.c, con_tbl.d
    0.25 .* ((a ./ (a .+ b)) .+ (a ./ (a .+ c)) .+ (d ./ (d .+ b)) .+ (d ./ (d .+ c)))
end
# 27. Odds Ratio
# This is already implemented earlier as eval_oddsratio. No further action needed.

# 28. Yule's Ω
# This is already implemented earlier as eval_yuleomega. No further action needed.

# 29. Yule's Q
# This is already implemented earlier as eval_yuleq. No further action needed.

# 30. Driver-Kroeber
function eval_driverkroeber(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    a, b, c, d = con_tbl.a, con_tbl.b, con_tbl.c, con_tbl.d
    a ./ sqrt.((a .+ b) .* (a .+ c))
end

# 31. Fifth Sokal-Sneath
function eval_fifth_sokalsneath(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    a, b, c, d = con_tbl.a, con_tbl.b, con_tbl.c, con_tbl.d
    (a .* d) ./ sqrt.((a .+ b) .* (a .+ c) .* (d .+ b) .* (d .+ c))
end

# 32. Pearson
# This has already been implemented as eval_cramersv due to the connection between the two.

# 33. Baroni-Urbani
function eval_baroniurbani(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    a, d, b, c = con_tbl.a, con_tbl.d, con_tbl.b, con_tbl.c
    (a .+ sqrt.(a .* d)) ./ (a .+ b .+ c .+ sqrt.(a .* d))
end

# 34. Braun-Blanquet
function eval_braunblanquet(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    a, b, c = con_tbl.a, con_tbl.b, con_tbl.c
    a ./ max.(a .+ b, a .+ c)
end

# 35. Simpson
function eval_simpson(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    a, b, c = con_tbl.a, con_tbl.b, con_tbl.c
    a ./ min.(a .+ b, a .+ c)
end

# 36. Michael
function eval_michael(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    a, b, c, d = con_tbl.a, con_tbl.b, con_tbl.c, con_tbl.d
    (4 .* (a .* d .- b .* c)) ./ ((a .+ d) .^ 2 .+ (b .+ c) .^ 2)
end

# 37. Mountford
function eval_mountford(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    a, b, c = con_tbl.a, con_tbl.b, con_tbl.c
    (2 .* a) ./ (2 .* b .* c .+ a .* b .+ a .* c)
end

# 38. Fager
function eval_fager(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    a, b, c = con_tbl.a, con_tbl.b, con_tbl.c
    (a ./ sqrt.((a .+ b) .* (a .+ c))) .- (0.5 .* max.(b, c))
end

# %% LaTeX formulas for association measures and metrics

# % 1. Joint Probability
# P(xy) = \frac{a}{N}

# % 2. Conditional Probability
# P(y|x) = \frac{a}{m}

# % 3. Reverse Conditional Probability
# P(x|y) = \frac{a}{k}

# % 4. Pointwise Mutual Information (PMI)
# PMI = \log \frac{P(xy)}{P(x)P(y)} = \log(a) + \log(N) - \log(k) - \log(m)

# % 5. Mutual Dependency
# MD = \frac{P(xy)^2}{P(x)P(y)} = \frac{a^2}{k \cdot m}

# % 6. Log Frequency Biased MD
# LFMD = \log \frac{P(xy)^2}{P(x)P(y)} + \log P(xy) = \log(a^2) + \log(N) - \log(k) - \log(m)

# % 7. Normalized Expectation
# NE = \frac{2 \cdot P(xy)}{P(x) + P(y)} = \frac{2 \cdot a}{k + m}

# % 8. Mutual Expectation
# ME = \frac{2 \cdot P(xy)^2}{P(x) + P(y)} = \frac{2 \cdot a^2}{k + m}

# % 9. Salience
# Salience = \frac{P(xy)^2}{P(x)P(y)} \cdot \log P(xy) = \frac{a^2}{k \cdot m} \cdot \log(a)

# % 10. Pearson's \chi^2 Test
# \chi^2 = \frac{(a - E_{11})^2}{E_{11}}, \text{ where } E_{11} = \frac{(a + b)(a + c)}{N}

# % 12. t-Test
# t = \frac{P(xy) - P(x)P(y)}{\sqrt{P(xy)(1 - P(xy)) / N}} = \frac{a - \frac{k \cdot m}{N}}{\sqrt{a(1 - \frac{a}{N}) / N}}

# % 13. z-Score
# z = \frac{P(xy) - P(x)P(y)}{\sqrt{P(xy)(1 - P(xy)) / N}} = \frac{a - \frac{k \cdot m}{N}}{\sqrt{a(1 - \frac{a}{N}) / N}}

# % 14. Poisson Significance Measure
# PSM = f_{obs} \log \frac{f_{obs}}{f_{exp}} + f_{exp}, \text{ where } f_{obs} = a \text{ and } f_{exp} = \frac{k \cdot m}{N}

# % 15. Log Likelihood Ratio
# LLR = 2 \cdot \left(a \log \frac{a}{E_{11}} + b \log \frac{b}{E_{12}} + c \log \frac{c}{E_{21}} + d \log \frac{d}{E_{22}}\right)

# % 16. Squared Log Likelihood Ratio
# LLR^2 = \left(LLR\right)^2

# % 17. Russel-Rao
# RR = \frac{a}{a + b + c + d}

# % 18. Sokal-Michener
# SM = \frac{a + d}{a + b + c + d}

# % 19. Rogers-Tanimoto
# RT = \frac{a + d}{a + 2(b + c) + d}

# % 20. Hamann
# H = \frac{(a + d) - (b + c)}{a + b + c + d}

# % 22. Jaccard
# J = \frac{a}{a + b + c}

# % 24. Second Sokal-Sneath
# SS2 = \frac{a}{a + 2(b + c)}

# % 25. Second Kulczynski
# K2 = \frac{1}{2} \left(\frac{a}{a + b} + \frac{a}{a + c}\right)

# % 26. Fourth Sokal-Sneath
# SS4 = \frac{1}{4} \left(\frac{a}{a + b} + \frac{a}{a + c} + \frac{d}{d + b} + \frac{d}{d + c}\right)

# % 30. Driver-Kroeber
# DK = \frac{a}{\sqrt{(a + b)(a + c)}}

# % 31. Fifth Sokal-Sneath
# SS5 = \frac{ad}{\sqrt{(a + b)(a + c)(d + b)(d + c)}}

# % 33. Baroni-Urbani
# BU = \frac{a + \sqrt{ad}}{a + b + c + \sqrt{ad}}

# % 34. Braun-Blanquet
# BB = \frac{a}{\max(a + b, a + c)}

# % 35. Simpson
# S = \frac{a}{\min(a + b, a + c)}

# % 36. Michael
# M = \frac{4(ad - bc)}{(a + d)^2 + (b + c)^2}

# % 37. Mountford
# MN = \frac{2a}{2bc + ab + ac}

# % 38. Fager
# F = \frac{a}{\sqrt{(a + b)(a + c)}} - \frac{1}{2} \max(b, c)


# ======================================== from that point on i need to check the implementation (old chatgpt session)

# Rogers-Tanimoto Coefficient  (traditional)
# "Rogers-Tanimoto Coefficient", \frac{a}{a + 2(b + c)}
function eval_rogerstanimotocoef(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    con_tbl.a ./ (con_tbl.a .+ 2 .* (con_tbl.b .+ con_tbl.c))
end

const rogerstanimotocoef = eval_rogerstanimotocoef


# Rogers-Tanimoto Coefficient (incorporates the frequency of occurrences where neither of the events occurs, making it more inclusive in certain scenarios)
# "Rogers-Tanimoto Coefficient", (a + d) / (a + 2 * (b + c) + d)
function eval_rogerstanimotocoef2(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    (con_tbl.a .+ con_tbl.d) ./ (con_tbl.a .+ 2 .* (con_tbl.b .+ con_tbl.c) .+ con_tbl.d)
end

const rogerstanimotocoef2 = eval_rogerstanimotocoef2

# Hamann Similarity
# "Hamann Similarity", \frac{a + d - b - c}{N}
function eval_hamannsim(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    (con_tbl.a .+ con_tbl.d .- con_tbl.b .- con_tbl.c) ./ con_tbl.N
end

const hamannsim = eval_hamannsim

# Hamann Similarity 2
# "Hamann Similarity", \frac{a - d}{a + b + c - d}
function eval_hamannsim2(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    (con_tbl.a .- con_tbl.d) ./ (con_tbl.a .+ con_tbl.b .+ con_tbl.c .- con_tbl.d)
end

const hamannsim2 = eval_hamannsim2

# Goodman-Kruskal Index
# "Goodman-Kruskal Index", (a * d - b * c) / (a * d + b * c)
function eval_goodmankruskalidx(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    (con_tbl.a .* con_tbl.d .- con_tbl.b .* con_tbl.c) ./ (con_tbl.a .* con_tbl.d .+ con_tbl.b .* con_tbl.c)
end

const goodmankruskalidx = eval_goodmankruskalidx


# Gower's Coefficient (traditional) \frac{a}{a + b + c}
# "Gower's Coefficient",
function eval_gowercoef(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    con_tbl.a ./ (con_tbl.a .+ con_tbl.b .+ con_tbl.c)
end

const gowercoef = eval_gowercoef

# Gower's Coefficient, \frac{a + d}{a + d + 2(b + c)}
# "Gower's Coefficient", (a + d) / (a + d + 2 * (b + c))
function eval_gowercoef2(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    (con_tbl.a .+ con_tbl.d) ./ (con_tbl.a .+ con_tbl.d .+ 2 .* (con_tbl.b .+ con_tbl.c))
end

const gowercoef2 = eval_gowercoef2

# Czekanowski-Dice Coefficient, \frac{2a}{2a + b + c}
# "Czekanowski-Dice Coefficient", 2 * a / (2 * a + b + c)
function eval_czekanowskidicecoef(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    2 * con_tbl.a ./ (2 .* con_tbl.a .+ con_tbl.b .+ con_tbl.c)
end

const czekanowskidicecoef = eval_czekanowskidicecoef

# Sorgenfrey Index (traditional) \frac{2a - b - c}{2a + b + c}
# "Sorgenfrey Index",
function eval_sorgenfreyidx(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    # \frac{2a - b - c}{2a + b + c}
    (2 * con_tbl.a .- con_tbl.b .- con_tbl.c) ./ (2 .* con_tbl.a .+ con_tbl.b .+ con_tbl.c)
end

const sorgenfreyidx = eval_sorgenfreyidx

# Sorgenfrey Index
# "Sorgenfrey Index", (a + d) / (2 * (a + d) + b + c)
function eval_sorgenfreyidx2(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    (con_tbl.a .+ con_tbl.d) ./ (2 .* (con_tbl.a .+ con_tbl.d) .+ con_tbl.b .+ con_tbl.c)
end

const sorgenfreyidx2 = eval_sorgenfreyidx2

# Mountford's Coefficient (traditional) \frac{a}{a + 2b + 2c}
# "Mountford's Coefficient",
function eval_mountfordcoef(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    # \frac{a}{a + 2b + 2c}
    con_tbl.a ./ (con_tbl.a .+ 2 .* (con_tbl.b .+ con_tbl.c))
end

const mountfordcoef = eval_mountfordcoef

# Mountford's Coefficient 2 (alternative)
# "Mountford's Coefficient", (a + d) / (a + d + 2 * sqrt((b + c) * (k + m)))
function eval_mountfordcoef2(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    (con_tbl.a .+ con_tbl.d) ./ (con_tbl.a .+ con_tbl.d .+ 2 .* sqrt.((con_tbl.b .+ con_tbl.c) .* (con_tbl.k .+ con_tbl.m)))
end

const mountfordcoef2 = eval_mountfordcoef2

# Sokal-Sneath Index, \frac{a}{a + 2b + 2c}
# "Sokal-Sneath Index", a / (a + 2 * (b + c))
function eval_sokalsneathidx(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    con_tbl.a ./ (con_tbl.a .+ 2 * (con_tbl.b .+ con_tbl.c))
end

const sokalsneathidx = eval_sokalsneathidx

# Sokal-Michener Coefficient
# "Sokal-Michener Coefficient", DONE \frac{a + d}{a + b + c + d}
function eval_sokalmichenercoef(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    (con_tbl.a .+ con_tbl.d) ./ con_tbl.N
end

const sokalmichenercoef = eval_sokalmichenercoef


# Gravity G Index
# "Gravity G Index", (a * d) / (b * c)
function eval_lexicalgravity(data::AssociationDataFormat)
    con_tbl = cached_data(data.con_tbl)
    # log((f(w1,w2)*n(w1)/f(w1))) + log((f(w1,w2)*n'(w2)/f(w2)))
    # calculate the n'(w2) and f(w2) for each word in the context window

    # Retrieve the raw input string lazily
    inputstring = cached_data(data.input_ref.loader)

    f_w1_w2 = con_tbl.a
    n_w1 = nrow(con_tbl)
    f_w1 = sum(con_tbl.a)
    n_w2 = find_prior_words(inputstring, con_tbl.Collocate, con_tbl.windowsize)
    f_w2 = count_substrings(inputstring, string.(" ", con_tbl.Collocate, " "))
    log.(f_w1_w2 .* n_w1 / f_w1) .+ log.(f_w1_w2 .* n_w2 ./ f_w2)

end

const lexicalgravity = eval_lexicalgravity


"""
    assoc_score(metricType::Type{<:AssociationMetric}, cont_tbl::AssociationDataFormat)

Evaluate an association metric based on the provided metric type and a contingency table. This function dynamically dispatches the calculation to the appropriate function determined by `metricType`.

# Arguments
- `metrics::Array{<:AssociationMetric}`: An array of association metric types to evaluate.
- `data::AssociationDataFormat`: The contingency table data on which to evaluate the metrics. To create one, use the `AssociationDataFormat` constructor.

# Returns
- A Vector of numerical values where each value represents the association metric score of the node word picked when creating the AssociationDataFormat with each of the co-occurring words in the window length picked when creating the AssociationDataFormat.

# Usage

```julia-doc
result = assoc_score(MetricType, cont_tbl)
```

Replace `MetricType` with the desired association metric type (e.g., `PMI`, `Dice`) and cont_tbl with your contingency table. You can see all supported metrics through `listmetrics()`.

# Examples
**PMI (Pointwise Mutual Information)**:

```julia-doc
result = assoc_score(PMI, cont_tbl)
```

**Dice Coefficient**:

```julia-doc
result = assoc_score(Dice, cont_tbl)
```

# Further Reading

For detailed mathematical definitions and discussion on each metric, refer to our documentation site.
"""
function assoc_score(metricType::Type{<:AssociationMetric}, input::Union{AssociationDataFormat,AbstractString}; node::AbstractString="", windowsize::Int=0, minfreq::Int=5)

    # Convert raw input to AssociationDataFormat if necessary
    if input isa AbstractString
        input = AssociationDataFormat(input, node, windowsize, minfreq)
    end

    func_name = Symbol("eval_", lowercase(string(metricType)))  # Construct function name
    func = getfield(@__MODULE__, func_name)  # Get the function from the current module
    return func(input), input # Call the function
end

"""
    assoc_score(metrics::Array{<:AssociationMetric}, cont_tbl::AssociationDataFormat)

Evaluate an array of association metrics on the given contingency table.

# Arguments
- `metrics::Array{<:AssociationMetric}`: An array of association metric types to evaluate.
- `data::AssociationDataFormat`: The contingency table data on which to evaluate the metrics.

# Returns
- A DataFrame where each column represents an evaluation result for a corresponding metric.

# Usage

```julia-doc
result = assoc_score([MetricType1, MetricType2, MetricType3, ...], cont_tbl)
```

Replace `MetricType\$` with the desired association metric types (e.g., `PMI`, `Dice`) and cont_tbl with your contingency table. You can see all supported metrics through `listmetrics()`.


# Examples
**PMI (Pointwise Mutual Information)** and **Dice Coefficient** returned within two columns of the DataFrame `result` below.

```julia-doc
assoc_score(Metrics([PMI, Dice]), cont_to)

n×2 DataFrame
 Row │ PMI  Dice
     │ Float64   Float64
─────┼─────────────────
   1 │ 0.2      0.4
   2 | 0.3      0.3
   3 | 0.1      0.5
   4 | 0.7      0.6
```
"""
function assoc_score(metrics::Vector{DataType}, input::Union{AssociationDataFormat,AbstractString};
    node::AbstractString="", windowsize::Int=0, minfreq::Int=5)
    # Validate that all elements are subtypes of AssociationMetric
    if !all(metric -> metric <: AssociationMetric, metrics)
        throw(ArgumentError("All metrics must be subtypes of AssociationMetric. Found: $metrics"))
    end

    # Convert raw input to AssociationDataFormat if necessary
    if input isa AbstractString
        input = AssociationDataFormat(input, node, windowsize, minfreq)
    end

    results_df = DataFrame()
    for metric in metrics
        func_name = Symbol("eval_", lowercase(string(metric)))  # Construct function name
        func = getfield(@__MODULE__, func_name)  # Get the function from the current module
        result = func(input)  # Call the function and store the result
        results_df[!, string(metric)] = result  # Add the result to the DataFrame as a column
    end
    return results_df, input
end

# Define a const that will result in a DataFrame with all metrics
const ALL_METRICS = [PMI, PMI², PMI³, PPMI, LLR, LLR², DeltaPi, MinSens, Dice, LogDice, RelRisk, LogRelRisk, RiskDiff, AttrRisk, OddsRatio, LogRatio, LogOddsRatio, JaccardIdx, OchiaiIdx, PiatetskyShapiro, YuleOmega, YuleQ, PhiCoef, CramersV, TschuprowT, ContCoef, CosineSim, OverlapCoef, KulczynskiSim, TanimotoCoef, RogersTanimotoCoef, RogersTanimotoCoef2, HammanSim, HammanSim2, GoodmanKruskalIdx, GowerCoef, GowerCoef2, CzekanowskiDiceCoef, SorgenfreyIdx, SorgenfreyIdx2, MountfordCoef, MountfordCoef2, SokalSneathIdx, SokalMichenerCoef, Tscore, Zscore, ChiSquare, FisherExactTest, CohensKappa]

# templatic way of creating dynamically the wrapper functions that serve as unified API functions
for metric in ALL_METRICS
    @eval begin
        eval_func_name = Symbol("eval_", lowercase(string($metric)))

        # Define the unified API function
        $(Symbol("eval_", lowercase(string(metric))))(
            input::Union{AssociationDataFormat,AbstractString},
            node::AbstractString="",
            windowsize::Int=0,
            minfreq::Int=5
        ) = assoc_score($metric, input, node=node, windowsize=windowsize, minfreq=minfreq)
    end
end


# OverlapCoefficient

# Docstring values for the template placeholders
metric_templates = Dict(
    :PMI => (
        description="Compute Pointwise Mutual Information (PMI) for a given contingency table.",
        formula="\\text{PMI}(a, b) = \\log_2\\left(\\frac{P(a, b)}{P(a)P(b)}\\right)",
        usage="eval_pmi(cont_tbl)"
    ),
    :PPMI => (
        description="Compute Positive Pointwise Mutual Information (PPMI) for a given contingency table.",
        formula="\\text{PPMI}(a, b) = \\max(0, \\text{PMI}(a, b))",
        usage="eval_ppmi(cont_tbl)"
    ),
    :Dice => (
        description="Compute the Dice Coefficient for a given contingency table.",
        formula="\\text{Dice} = \\frac{2a}{m + k}",
        usage="eval_dice(cont_tbl)"
    ),
    :LogDice => (
        description="Compute Log Dice for a given contingency table.",
        formula="\\text{LogDice} = 14 + \\log_2\\left(\\frac{2a}{m + k}\\right)",
        usage="eval_logdice(cont_tbl)"
    ),
    :RelRisk => (
        description="Compute the Relative Risk (RR) for a given contingency table.",
        formula="\\text{RR} = \\frac{\\frac{a}{m}}{\\frac{c}{n}}",
        usage="eval_relrisk(cont_tbl)"
    ),
    :LogRelRisk => (
        description="Compute the Log Relative Risk for a given contingency table.",
        formula="\\log(\\text{RR}) = \\log\\left(\\frac{a}{m}\\right) - \\log\\left(\\frac{c}{n}\\right)",
        usage="eval_logrelrisk(cont_tbl)"
    ),
    :RiskDiff => (
        description="Compute the Risk Difference for a given contingency table.",
        formula="\\text{Risk Difference} = \\frac{a}{m} - \\frac{c}{n}",
        usage="eval_riskdiff(cont_tbl)"
    ),
    :AttrRisk => (
        description="Compute the Attributable Risk for a given contingency table.",
        formula="\\text{Attributable Risk} = \\frac{a}{m} - \\frac{c}{n}",
        usage="eval_attrrisk(cont_tbl)"
    ),
    :OddsRatio => (
        description="Compute the Odds Ratio for a given contingency table.",
        formula="\\text{OR} = \\frac{a \\cdot d}{b \\cdot c}",
        usage="eval_oddsratio(cont_tbl)"
    ),
    :LogOddsRatio => (
        description="Compute the Log Odds Ratio for a given contingency table.",
        formula="\\log(\\text{OR}) = \\log\\left(\\frac{a \\cdot d}{b \\cdot c}\\right)",
        usage="eval_logoddsratio(cont_tbl)"
    ),
    :JaccardIdx => (
        description="Compute the Jaccard Index for a given contingency table.",
        formula="\\text{Jaccard Index} = \\frac{a}{a + b + c}",
        usage="eval_jaccardindex(cont_tbl)"
    ),
    :OchiaiIdx => (
        description="Compute the Ochiai Index for a given contingency table.",
        formula="\\text{Ochiai Index} = \\frac{a}{\\sqrt{(a + b)(a + c)}}",
        usage="eval_ochiaiindex(cont_tbl)"
    )
    # Add more metrics as needed...
)


# function for generating the docstrings
function generate_docstring(metric::Symbol)
    # Extract short name from qualified name (e.g., TextAssociations.PMI => PMI)
    metric_symbol = metric isa Symbol ? metric : Symbol(split(string(metric), ".")[end])
    template = metric_templates[metric_symbol]
    """
    $(template.description)

    # Arguments
    - `data::AssociationDataFormat`: The contingency table to evaluate.

    # Returns
    - `Array`: An array of $(string(metric)) scores.

    # Formula
    ``$(template.formula)``

    # Usage
    ```julia
    $(template.usage)
    ```
    """
end

# attaching docstrings for all the measures to the function definitions
for metric in keys(metric_templates)
    @eval begin
        # Attach the generated docstring
        @doc generate_docstring(Symbol(split(string($metric), ".")[end])) $(Symbol("eval_", lowercase(string(metric))))
    end
end
