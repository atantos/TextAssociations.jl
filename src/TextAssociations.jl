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


# Exports
export
    # generic function
    evalassoc,
    listmetrics,

    # generic types/functions
    AssociationMetric,
    AssociationDataFormat,
    Metrics,
    ContingencyTable,

    # Constants
    ALL_METRICS,

    # Metric Types
    PMI,
    PMI²,
    PMI³,
    PPMI,
    LLR,
    LLR2,
    LLR²,
    DeltaPi,
    MinSens,
    Dice,
    LogDice,
    RelRisk,
    LogRelRisk,
    RiskDiff,
    AttrRisk,
    OddsRatio,
    LogRatio,
    LogOddsRatio,
    JaccardIdx,
    OchiaiIdx,
    PiatetskyShapiro,
    YuleOmega,
    YuleQ,
    PhiCoef,
    CramersV,
    TschuprowT,
    ContCoef,
    CosineSim,
    OverlapCoef,
    KulczynskiSim,
    TanimotoCoef,
    RogersTanimotoCoef,
    RogersTanimotoCoef2,
    HammanSim,
    HammanSim2,
    GoodmanKruskalIdx,
    GowerCoef,
    GowerCoef2,
    CzekanowskiDiceCoef,
    SorgenfreyIdx,
    SorgenfreyIdx2,
    MountfordCoef,
    MountfordCoef2,
    SokalSneathIdx,
    SokalMichenerCoef,
    Tscore,
    Zscore,
    ChiSquare,
    FisherExactTest,
    CohensKappa,

    # Utility Functions
    tostringvector,
    prepstring,
    createvocab,
    conttbl


include("types.jl")
include("utils.jl")
include("associations/measures.jl")

function Base.show(io::IO, con_tbl::ContingencyTable)
    println(io, "ContingencyTable instance with:")
    println(io, "* Node word: $(con_tbl.node)")
    println(io, "* Window size: $(con_tbl.windowsize)")
    println(io, "* Minimum collocation frequency: $(con_tbl.minfreq)")
end

end
