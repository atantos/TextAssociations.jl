
# log2((a / N) / ((k / N) * (m / N)))
function eval_pmi(data::ContingencyTable)
    con_tbl = TextAssociations.extract_cached_data(data.con_tbl)
    log2.((con_tbl.a ./ con_tbl.N) .- log2.((con_tbl.k ./ con_tbl.N) .* (con_tbl.m ./ con_tbl.N)))
end

const pmi = eval_pmi


# log2((a^2 / N) / ((k / N) * (m / N)))
# Pointwise Mutual Information²     
function eval_pmi²(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    log2.((con_tbl.a .^ 2 ./ con_tbl.N) .- log2.((con_tbl.k ./ con_tbl.N) .* (con_tbl.m ./ con_tbl.N)))
end

const pmi² = eval_pmi²


# log2((a^3 / N) / ((k / N) * (m / N)))
# Pointwise Mutual Information³ 
function eval_pmi³(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    log2.((con_tbl.a .^ 3 ./ con_tbl.N) .- log2.((con_tbl.k ./ con_tbl.N) .* (con_tbl.m ./ con_tbl.N)))
end

const pmi³ = eval_pmi³


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

# Dice f Co-occurrence based word association
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
#  https://www.ncbi.nlm.nih.gov/books/NBK430824/figure/article-28324.image.f1/
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
# incidence proportion difference 46.6.2 Incidence proportion difference, https://www.r4epi.com/measures-of-association
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
    con_tbl.a ./ (con_tbl.m .+ con_tbl.c)
end

const jaccard = eval_jaccardindex


# Ochiai Index
# "Ochiai", a / sqrt((a + b) * (a + c))
function eval_ochiaiindex(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    con_tbl.a ./ sqrt.((con_tbl.m) .* (con_tbl.k))
end

const ochiai = eval_ochiaiindex

# Piatetsky Shapiro
# "Piatetsky Shapiro", \frac{a}{n} - \frac{(a + b)(a + c)}{n^2}
function eval_piatetskyshapiro(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    (con_tbl.a ./ con_tbl.N) .- ((con_tbl.k .* con_tbl.m) ./ con_tbl.N²)
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
    con_tbl.a ./ (con_tbl.a .+ 2(con_tbl.b .+ con_tbl.c))
end

const rogerstanimotocoef = eval_rogerstanimotocoef


# Rogers-Tanimoto Coefficient (incorporates the frequency of occurrences where neither of the events occurs, making it more inclusive in certain scenarios)
# "Rogers-Tanimoto Coefficient", (a + d) / (a + 2 * (b + c) + d)
function eval_rogerstanimotocoef2(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)a
    (con_tbl.a .+ con_tbl.d) ./ (con_tbl.a .+ 2(con_tbl.b .+ con_tbl.c) + con_tbl.d)
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
    (con_tbl.a .- con_tbl.d) ./ (con_tbl.a .+ con_tbl.b .+ con_tbl.c .- con_tbl.d)
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
    con_tbl.a ./ (con_tbl.a .+ con_tbl.b .+ con_tbl.c)
end

const gowercoef = eval_gowercoef

# Gower's Coefficient, \frac{a + d}{a + d + 2(b + c)}
# "Gower's Coefficient", (a + d) / (a + d + 2 * (b + c))
function eval_gowercoef2(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    (con_tbl.a .+ con_tbl.d) ./ (con_tbl.a .+ con_tbl.d .+ 2(con_tbl.b .+ con_tbl.c))
end

const gowercoef2 = eval_gowercoef2

# Czekanowski-Dice Coefficient, \frac{2a}{2a + b + c}
# "Czekanowski-Dice Coefficient", 2 * a / (2 * a + b + c)
function eval_czekanowskidicecoef(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    2 * con_tbl.a ./ (2 * con_tbl.a .+ con_tbl.b .+ con_tbl.c)
end

const czekanowskidicecoef = eval_czekanowskidicecoef

# Sorgenfrey Index (traditional) \frac{2a - b - c}{2a + b + c}
# "Sorgenfrey Index",
function eval_sorgenfreyindex(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # \frac{2a - b - c}{2a + b + c}
    (2 * con_tbl.a .- con_tbl.b .- con_tbl.c) ./ (2 * con_tbl.a .+ con_tbl.b .+ con_tbl.c)
end

const sorgenfreyidx = eval_sorgenfreyindex

# Sorgenfrey Index
# "Sorgenfrey Index", (a + d) / (2 * (a + d) + b + c)
function eval_sorgenfreyindex2(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    (con_tbl.a .+ con_tbl.d) ./ (2 * (con_tbl.a .+ con_tbl.d) .+ con_tbl.b .+ con_tbl.c)
end

const sorgenfreyidx2 = eval_sorgenfreyindex2

# Mountford's Coefficient (traditional) \frac{a}{a + 2b + 2c}
# "Mountford's Coefficient", 
function eval_mountfordcoef(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # \frac{a}{a + 2b + 2c}
    con_tbl.a ./ (con_tbl.a .+ 2 * (con_tbl.b .+ con_tbl.c))
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
    con_tbl.a ./ (con_tbl.a .+ 2 * (con_tbl.b .+ con_tbl.c))
end

const sokalsneathidx = eval_sokalsneathindex

# Sokal-Michener Coefficient
# "Sokal-Michener Coefficient", DONE \frac{a + d}{a + b + c + d}
function eval_sokalmichenercoef(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    (con_tbl.a .+ con_tbl.d) ./ con_tbl.N
end

const sokalmichenercoef = eval_sokalmichenercoef


# Gravity G Index
# "Gravity G Index", (a * d) / (b * c)
function eval_lexicalgravity(data::ContingencyTable)
    con_tbl = extract_cached_data(data.con_tbl)
    # log((f(w1,w2)*n(w1)/f(w1))) + log((f(w1,w2)*n'(w2)/f(w2)))
    # calculate the n'(w2) and f(w2) for each word in the context window

    f_w1_w2 = con_tbl.a
    n_w1 = nrow(con_tbl)
    f_w1 = sum(con_tbl.a)
    n_w2 = find_prior_words(data.prepared_string, con_tbl.Collocate, con_tbl.windowsize)
    f_w2 = count_substrings(data.prepared_string, string.(" ", con_tbl.Collocate, " "))
    log.(f_w1_w2 .* n_w1 / f_w1) .+ log.(f_w1_w2 .* n_w2 ./ f_w2)

end

const lexicalgravity = eval_lexicalgravity


"""
    evalassoc(metricType::Type{<:AssociationMetric}, cont_tbl::ContingencyTable)

Evaluate an association metric based on the provided metric type and a contingency table. This function dynamically dispatches the calculation to the appropriate function determined by `metricType`.

# Arguments
- `metrics::Array{<:AssociationMetric}`: An array of association metric types to evaluate.
- `data::ContingencyTable`: The contingency table data on which to evaluate the metrics. To create one, use the `ContingencyTable` constructor. 

# Returns
- A Vector of numerical values where each value represents the association metric score of the node word picked when creating the ContingencyTable with each of the co-occurring words in the window length picked when creating the ContingencyTable. 

# Usage

```julia-doc
result = evalassoc(MetricType, cont_tbl)
```

Replace `MetricType` with the desired association metric type (e.g., `PMI`, `Dice`) and cont_tbl with your contingency table. You can see all supported metrics through `listmetrics()`.

# Examples
**PMI (Pointwise Mutual Information)**:

```julia-doc
result = evalassoc(PMI, cont_tbl)
```

**Dice Coefficient**:

```julia-doc
result = evalassoc(Dice, cont_tbl)
```

# Further Reading

For detailed mathematical definitions and discussion on each metric, refer to our documentation site.

    evalassoc(metrics::Array{<:AssociationMetric}, cont_tbl::ContingencyTable)

Evaluate an array of association metrics on the given contingency table.

# Arguments
- `metrics::Array{<:AssociationMetric}`: An array of association metric types to evaluate.
- `data::ContingencyTable`: The contingency table data on which to evaluate the metrics.

# Returns
- A DataFrame where each column represents an evaluation result for a corresponding metric.

# Usage

```julia-doc
result = evalassoc([MetricType1, MetricType2, MetricType3, ...], cont_tbl)
```

Replace `MetricType\$` with the desired association metric types (e.g., `PMI`, `Dice`) and cont_tbl with your contingency table. You can see all supported metrics through `listmetrics()`.


# Examples
**PMI (Pointwise Mutual Information)** and **Dice Coefficient** returned within two columns of the DataFrame `result` below.

```julia-doc
evalassoc(Metrics([PMI, Dice]), cont_to)

n×2 DataFrame
 Row │ PMI  Dice  
     │ Float64   Float64 
─────┼─────────────────
   1 │ 0.2		0.4 		
   2 | 0.3		0.3 		
   3 | 0.1 		0.5 		
   4 | 0.7		0.6		
```

"""
function evalassoc(metricType::Type{<:AssociationMetric}, data::ContingencyTable)
    func_name = Symbol("eval_", lowercase(string(metricType)))  # Construct function name
    func = getfield(@__MODULE__, func_name)  # Get the function from the current module
    return func(data)  # Call the function
end

function evalassoc(metrics::Vector{DataType}, data::ContingencyTable)
    # Validate that all elements are subtypes of AssociationMetric
    if !all(metric -> metric <: AssociationMetric, metrics)
        throw(ArgumentError("All metrics must be subtypes of AssociationMetric. Found: $metrics"))
    end

    results_df = DataFrame()
    for metric in metrics
        func_name = Symbol("eval_", lowercase(string(metric)))  # Construct function name
        func = getfield(@__MODULE__, func_name)  # Get the function from the current module
        result = func(data)  # Call the function and store the result
        results_df[!, string(metric)] = result  # Add the result to the DataFrame as a column
    end
    return results_df
end

# Define a const that will result in a DataFrame with all metrics
const ALL_METRICS = [PMI, PMI², PMI³, PPMI, LLR, LLR2, LLR², DeltaPi, MinSens, Dice, LogDice, RelRisk, LogRelRisk, RiskDiff, AttrRisk, OddsRatio, LogRatio, LogOddsRatio, JaccardIdx, OchiaiIdx, PiatetskyShapiro, YuleOmega, YuleQ, PhiCoef, CramersV, TschuprowT, ContCoef, CosineSim, OverlapCoef, KulczynskiSim, TanimotoCoef, RogersTanimotoCoef, RogersTanimotoCoef2, HammanSim, HammanSim2, GoodmanKruskalIdx, GowerCoef, GowerCoef2, CzekanowskiDiceCoef, SorgenfreyIdx, SorgenfreyIdx2, MountfordCoef, MountfordCoef2, SokalSneathIdx, SokalMichenerCoef, Tscore, Zscore, ChiSquare, FisherExactTest, CohensKappa]

# OverlapCoefficient

