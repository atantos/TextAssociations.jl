# =====================================
# File: src/types.jl
# Type definitions
# =====================================
"""
Abstract type for all association metrics.
"""
abstract type AssociationMetric <: SemiMetric end

"""
Abstract type for data formats used in association computations.
"""
abstract type AssociationDataFormat end

# Define metric types programmatically
const METRIC_TYPES = [
    :PMI, :PMI², :PMI³, :PPMI,
    :LLR, :LLR²,
    :DeltaPi, :MinSens,
    :Dice, :LogDice,
    :RelRisk, :LogRelRisk, :RiskDiff, :AttrRisk,
    :OddsRatio, :LogOddsRatio,
    :JaccardIdx, :OchiaiIdx,
    :PiatetskyShapiro,
    :YuleOmega, :YuleQ,
    :PhiCoef, :CramersV, :TschuprowT, :ContCoef,
    :CosineSim, :OverlapCoef, :KulczynskiSim,
    :TanimotoCoef, :RogersTanimotoCoef, :RogersTanimotoCoef2,
    :HammanSim, :HammanSim2,
    :GoodmanKruskalIdx,
    :GowerCoef, :GowerCoef2,
    :CzekanowskiDiceCoef,
    :SorgenfreyIdx, :SorgenfreyIdx2,
    :MountfordCoef, :MountfordCoef2,
    :SokalSneathIdx, :SokalMichenerCoef,
    :LexicalGravity,
    :Tscore, :Zscore, :ChiSquare
]

# Generate abstract types for each metric
for metric in METRIC_TYPES
    @eval abstract type $metric <: AssociationMetric end
end

"""
    LazyProcess{T,R}

Lazy evaluation wrapper for deferred computations.
Stores a function that computes a result when first needed and caches it.
"""
mutable struct LazyProcess{T,R}
    f::T
    cached_result::Union{Nothing,R}
    cached_process::Bool

    LazyProcess(f::T) where {T} = new{T,DataFrame}(f, nothing, false)
    LazyProcess(f::T, ::Type{R}) where {T,R} = new{T,R}(f, nothing, false)
end

"""
    LazyInput

Wrapper for lazily storing and accessing the processed input document.
This is used by metrics like Lexical Gravity that need access to the 
original text beyond just the contingency table.
"""
struct LazyInput
    loader::LazyProcess{F,StringDocument} where {F}

    function LazyInput(input_doc::StringDocument)
        # Store a reference to the document lazily
        lazy_proc = LazyProcess(() -> input_doc, StringDocument)
        return new(lazy_proc)
    end
end
