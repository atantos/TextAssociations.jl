# =====================================
# File: src/TextAssociations.jl
# Main module file
# =====================================
module TextAssociations

using Distances: Distances, SemiMetric, Metric, evaluate, result_type
using DataFrames
using DataStructures: OrderedDict
using TextAnalysis
using SparseArrays: SparseMatrixCSC
using Chain
using ProgressMeter
using Unicode

# === Types & core ===
include("types.jl")

# === Utils ===
include("utils/io.jl")
include("utils/text_processing.jl")
include("utils/statistical.jl")
include("utils/text_analysis.jl")

# === Core data structures ===
include("core/contingency_table.jl")
include("core/corpus_analysis.jl")
include("core/advanced_corpus.jl")

# === Metrics (iface first, then the umbrella) ===
include("metrics/_iface.jl")          # abstract API & fallbacks
include("metrics/metrics.jl")         # umbrella that pulls all metric impls

# === Public API last ===
include("api.jl")


# Exports
export
    # Core types
    AssociationMetric,
    AssociationDataFormat,
    ContingencyTable,
    LazyProcess,
    LazyInput,
    TextNorm,
    # Corpus-related types
    StringDocument, # from TextAnalysis
    Corpus,
    CorpusContingencyTable,
    MultiNodeAnalysis,
    # Advanced features
    TemporalCorpusAnalysis,
    SubcorpusComparison,
    CollocationNetwork,
    Concordance,

    # Main API functions
    assoc_score,
    available_metrics,
    cached_data,
    document,
    # corpus-related API functions
    text, tokens, # from TextAnalysis
    read_corpus, read_corpus_df,
    analyze_node, analyze_nodes,
    corpus_stats, vocab_coverage,
    coverage_summary, token_distribution,
    write_results,
    batch_process_corpus, stream_corpus_analysis,
    # Advanced features
    analyze_temporal, compare_subcorpora,
    keyterms, colloc_graph,
    gephi_graph, kwic,

    # Utility functions
    normalize_node, prep_string,
    strip_diacritics, build_vocab,
    cont_table,

    # All metric types
    PMI, PMI², PMI³, PPMI,
    LLR, LLR²,
    BPMI, BLLR, FisherRight,
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
    ChiSquare, Tscore, Zscore,
    DeltaPiLeft, DeltaPiRight,
    BayesLLR,

    # DataFrame utilities
    write_results,
    analysis_metric,
    analysis_summary,
    filter_scores,
    export_with_metadata,
    load_with_metadata

end
