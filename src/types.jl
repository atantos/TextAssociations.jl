abstract type AssociationMetric <: SemiMetric end

abstract type AssociationDataFormat end

# List of metric names
metric_names = [:PMI, :PMI², :PMI³, :PPMI, :LLR, :LLR2, :LLR², :DeltaPi, :MinSens, :Dice, :LogDice, :RelRisk, :LogRelRisk, :RiskDiff, :AttrRisk, :OddsRatio, :LogRatio, :LogOddsRatio, :JaccardIdx, :OchiaiIdx, :PiatetskyShapiro, :YuleOmega, :YuleQ, :PhiCoef, :CramersV, :TschuprowT, :ContCoef, :CosineSim, :OverlapCoef, :KulczynskiSim, :TanimotoCoef, :RogersTanimotoCoef, :RogersTanimotoCoef2, :HammanSim, :HammanSim2, :GoodmanKruskalIdx, :GowerCoef, :GowerCoef2, :CzekanowskiDiceCoef, :SorgenfreyIdx, :SorgenfreyIdx2, :MountfordCoef, :MountfordCoef2, :SokalSneathIdx, :SokalMichenerCoef, :Tscore, :Zscore, :ChiSquare, :FisherExactTest, :CohensKappa]

# Programmatically define an abstract type for each metric
for name in metric_names
    @eval abstract type $name <: AssociationMetric end
end


"""
    PMI(inputstring)

Calculate the Pointwise Mutual Information (PMI) score based on the following formula: log2(O₁₁/(E₁₁*span))

```math
\text{log}_{2}\frac{O_{11}}{E_{11} \ast S}
```

```jldoc

```

"""
# struct PMI <: AssociationMetric end

# Caching the text preprocessing for co-occurrence matrix and the countmap() for the cells of the table that will be counting https://discourse.julialang.org/t/best-practice-approach-for-caching-data-in-objects/20419?u=alex_tantos
mutable struct LazyProcess{T}
    f::Function
    cached_result::Union{Nothing,T}
    cached_process::Bool
    LazyProcess{T}(f) where {T} = new{T}(f, nothing, false)
end

struct ContingencyTable <: AssociationDataFormat
    con_tbl::LazyProcess{DataFrame}
    node::AbstractString
    windowsize::Int
    minfreq::Int64

    function ContingencyTable(inputstring::AbstractString, node::AbstractString, windowsize::Int, minfreq::Int64=5; auto_prep::Bool=true)
        prepared_string = auto_prep ? prepstring(inputstring) : inputstring
        con_tbl = LazyProcess{DataFrame}(() -> conttbl(prepared_string, node, windowsize, minfreq))

        new(con_tbl, node, windowsize, minfreq)
    end
end

#################################################################################
#
#  Help functions for navigating the CachedData type
#
#  extract_cached_coom(m::CachedData) = extract_cached_data(m.coo_matrix)
#  extract_cached_freq(m::CachedData) = extract_cached_data(m.unigram_frequencies)
#  extract_cached_paircounts(cached_data::CachedData) = extract_cached_data(cached_data.paircounts)
#  get_N(m::CachedData) = extract_cached_data(m.unigram_frequencies)
#  browse_coom_pairs(coo::CooMatrix, term1::AbstractString, term2::AbstractString)
# 
#################################################################################

"""
    extract_cached_data(z::LazyProcess{T}) 

Intermediate function for the `LazyProcess` type. If the `cached_process` is `false` then the function `f()` will be called and the result will be stored in `cached_result` and `cached_process` will be set to `true`. If `cached_process` is `true` then the `cached_result` will be returned (which is either a Cooccurrence Matrix or a Dictionary with words as keys and frequencies as their values).

# Example
```julia-repl
julia> doc = StringDocument("This is a text about an apple. There are many texts about apples.");
       z = CachedData(doc, 2, :default)
       extract_cached_data(z)
2.0
```
"""
function extract_cached_data(z::LazyProcess{T}) where {T}
    z.cached_process || (z.cached_result = z.f(); z.cached_process = true)
    z.cached_result::T
end
