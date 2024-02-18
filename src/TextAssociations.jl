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


# Write your package code here.

# Exports
export
    # generic functions
    evaluate,
    log_likelihood_ratio,

    # generic types/functions
    AssociationMetric,
    AssociationDataFormat,
    ContingencyTable,

    # specific measure types
    PMI,
    PMI2,
    PMI3,
    PPMI,
    LogLikelihoodRatio,
    MutualInformation,
    TScore,
    ZScore,
    DeltaP,
    ChiSquare,
    FisherExactTest,
    LogDice,
    LogRatio,
    JaccardIndex,
    OchiaiIndex,
    PiatetskyShapiro,
    YuleQ,
    YuleY,
    DiceCoefficient,
    PhiCoefficient,
    CramersV,
    TschuprowT,
    ContingencyCoefficient,
    UncertaintyCoefficient,
    CosineSimilarity,
    OverlapCoefficient,
    KulczynskiSimilarity,

    # convenience functions
    createcoom

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
