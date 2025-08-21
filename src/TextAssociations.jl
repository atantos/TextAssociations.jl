# =====================================
# File: src/TextAssociations.jl
# Main module file
# =====================================
module TextAssociations

using Distances: Distances, SemiMetric, Metric, evaluate, result_type
using Memoize
using StatsBase
using WordTokenizers
using DataStructures
using SparseArrays
using LinearAlgebra
using DataFrames
using FreqTables
using Chain
using TextAnalysis

# Include source files in logical order
include("types.jl")
include("utils/text_processing.jl")
include("utils/statistical.jl")
include("utils/text_analysis.jl")
include("core/contingency_table.jl")
include("metrics/base.jl")
include("metrics/information_theoretic.jl")
include("metrics/statistical.jl")
include("metrics/similarity.jl")
include("metrics/epidemiological.jl")
include("metrics/lexical_gravity.jl")
include("api.jl")

# Exports
export
    # Core types
    AssociationMetric,
    AssociationDataFormat,
    ContingencyTable,
    LazyProcess,
    LazyInput,

    # Main API functions
    evalassoc,
    listmetrics,
    extract_cached_data,
    extract_document,

    # Utility functions
    prepstring,
    createvocab,
    conttbl,

    # All metric types
    PMI, PMI², PMI³, PPMI,
    LLR, LLR²,
    DeltaPi, MinSens,
    Dice, LogDice,
    RelRisk, LogRelRisk, RiskDiff, AttrRisk,
    OddsRatio, LogOddsRatio,
    JaccardIdx, OchiaiIdx,
    PiatetskyShapiro,
    YuleOmega, YuleQ,
    PhiCoef, CramersV, TschuprowT, ContCoef,
    CosineSim, OverlapCoef, KulczynskiSim,
    TanimotoCoef, RogersTanimotoCoef, RogersTanimotoCoef2,
    HammanSim, HammanSim2,
    GoodmanKruskalIdx,
    GowerCoef, GowerCoef2,
    CzekanowskiDiceCoef,
    SorgenfreyIdx, SorgenfreyIdx2,
    MountfordCoef, MountfordCoef2,
    SokalSneathIdx, SokalMichenerCoef,
    LexicalGravity,
    ChiSquare, Tscore, Zscore

end
