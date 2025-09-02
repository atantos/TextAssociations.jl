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
using CSV
using JSON
using ProgressMeter
using Distributed
using HypothesisTests
using SQLite
using Dates
using Random

# Include source files in logical order
include("types.jl")
include("utils/io.jl")
include("utils/text_processing.jl")
include("utils/statistical.jl")
include("utils/text_analysis.jl")
include("core/contingency_table.jl")
include("core/corpus_analysis.jl")
include("core/advanced_corpus.jl")
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
    # Corpus-related types
    Corpus,
    CorpusContingencyTable,
    MultiNodeAnalysis,
    # Advanced features
    TemporalCorpusAnalysis,
    SubcorpusComparison,
    CollocationNetwork,
    Concordance,

    # Main API functions
    evalassoc,
    listmetrics,
    extract_cached_data,
    extract_document,
    # corpus-related API functions
    load_corpus, load_corpus_from_dataframe,
    analyze_corpus, analyze_multiple_nodes,
    corpus_statistics, export_results,
    batch_process_corpus, stream_corpus_analysis,
    # Advanced features
    temporal_corpus_analysis, compare_subcorpora,
    extract_keywords, build_collocation_network,
    export_network_to_gephi, generate_concordance

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
