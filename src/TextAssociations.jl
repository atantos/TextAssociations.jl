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
    ContingencyTable,

    # Metric Types
    PMI,
    PMI2,
    PMI3,
    PPMI,
    LLR,
    DeltaPi,
    Dice,
    LogDice,
    RelRisk,
    LogRelRisk,
    RiskDiff,
    AttrRisk,
    OddsRatio,
    LogRatio,
    LogOddsRatio,
    JaccardIndex,
    OchiaiIndex,
    OchiaiCoef,
    PiatetskyShapiro,
    YuleQ,
    YuleY,
    PhiCoef,
    CramersV,
    TschuprowT,
    ContCoef,
    CosineSim,
    OverlapCoef,
    KulczynskiSim,
    TanimotoCoef,
    GoodmanKruskalIndex,
    GowerCoef,
    CzekanowskiDiceCoef,
    SorgenfreyIndex,
    MountfordCoef,
    SokalSneathIndex,
    RogersTanimotoCoef,
    SokalmMchenerCoef,
    Tscore,
    Zscore,
    ChiSquare,
    FisherExactTest,

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
