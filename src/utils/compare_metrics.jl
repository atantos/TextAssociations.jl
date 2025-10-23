# =====================================
# File: src/utils/compare_metrics.jl
# Metric-comparison diagnostics
# =====================================

using DataFrames
using Statistics
using StatsBase

# Try Kendall τ via HypothesisTests if present (optional)
const _HAS_HT = try
    @eval import HypothesisTests
    true
catch
    false
end

"""
    MetricComparison

Container for compare_metrics() outputs. Fields may be empty DataFrames if a diagnostic
was disabled or early-exit triggered.

Fields:
- `scores::DataFrame`      — the input scores table (subset/sampled if requested)
- `rank_corr::DataFrame`   — Spearman (and Kendall if available) rank correlations
- `topk_overlap::DataFrame`— Top-K overlap / Jaccard per pair × K
- `rank_diffs::DataFrame`  — Per-collocate Δrank for a selected pair (default: PMI vs LLR)
- `by_band_corr::DataFrame`— Frequency-banded Spearman correlations
- `metadata::Dict{Symbol,Any}` — run parameters and early-exit notes
"""
Base.@kwdef mutable struct MetricComparison
    scores::DataFrame = DataFrame()
    rank_corr::DataFrame = DataFrame()
    topk_overlap::DataFrame = DataFrame()
    rank_diffs::DataFrame = DataFrame()
    by_band_corr::DataFrame = DataFrame()
    metadata::Dict{Symbol,Any} = Dict{Symbol,Any}()
end

# ---- Utilities ---------------------------------------------------------------

# Detect metric columns if user didn't pass explicit list for a scores DataFrame
function _metric_syms_from_df(df::DataFrame)
    # Heuristic: numeric columns except known non-metric fields
    excludes = Set([:Node, :Collocate, :Frequency, :DocFrequency, :Score])
    syms = Symbol[]
    for n in names(df)
        n in excludes && continue
        col = df[!, n]
        eltype(col) <: Real || continue
        push!(syms, n)
    end
    syms
end

# Build "best rank = 1" rank vectors for each metric (descending scores)
function _metric_ranks(df::DataFrame, metrics::Vector{Symbol})
    r = DataFrame(Collocate = hasproperty(df, :Collocate) ? df.Collocate : 1:nrow(df))
    for m in metrics
        r[!, Symbol(string(m)*"_rank")] = rank(-df[!, m]; tied = :average)
    end
    r
end

# Spearman (rank Pearson) for all metric pairs
function _spearman_corr(df::DataFrame, metrics::Vector{Symbol})
    R = _metric_ranks(df, metrics)
    pairs = [(metrics[i], metrics[j]) for i in 1:length(metrics) for j in i:length(metrics)]
    rows = Vector{NamedTuple{(:Metric1,:Metric2,:Spearman),Tuple{String,String,Float64}}}()
    for (m1, m2) in pairs
        x = R[!, Symbol(String(m1)*"_rank")]
        y = R[!, Symbol(String(m2)*"_rank")]
        ρ = cor(x, y)
        push!(rows, (String(m1), String(m2), ρ))
    end
    DataFrame(rows)
end

# Kendall τ (optional if HypothesisTests is installed)
function _kendall_corr(df::DataFrame, metrics::Vector{Symbol})
    !_HAS_HT && return DataFrame(Metric1=String[], Metric2=String[], KendallTau=Float64[])
    R = _metric_ranks(df, metrics)
    pairs = [(metrics[i], metrics[j]) for i in 1:length(metrics) for j in i:length(metrics)]
    rows = Vector{NamedTuple{(:Metric1,:Metric2,:KendallTau),Tuple{String,String,Float64}}}()
    for (m1, m2) in pairs
        x = R[!, Symbol(String(m1)*"_rank")]
        y = R[!, Symbol(String(m2)*"_rank")]
        τ = HypothesisTests.kendalltau(x, y).τ
        push!(rows, (String(m1), String(m2), τ))
    end
    DataFrame(rows)
end

# Top-K overlap (intersection count) and Jaccard per pair × K
function _topk_overlap(df::DataFrame, metrics::Vector{Symbol}; ks::Vector{Int}=[10,25,50])
    # Precompute index orderings
    ord = Dict{Symbol,Vector{Int}}()
    for m in metrics
        ord[m] = sortperm(df[!, m]; rev=true)  # high→low
    end
    rows = NamedTuple[]
    for K in ks
        for i in 1:length(metrics)-1, j in i+1:length(metrics)
            a = Set(ord[metrics[i]][1:min(K, nrow(df))])
            b = Set(ord[metrics[j]][1:min(K, nrow(df))])
            inter = length(intersect(a, b))
            union_ = length(union(a, b))
            jacc = union_ == 0 ? 0.0 : inter / union_
            push!(rows, (K=K, Pair="$(metrics[i]) vs $(metrics[j])", Overlap=inter, Jaccard=jacc))
        end
    end
    DataFrame(rows)
end

# Per-collocate Δrank for a specified pair (defaults to first two metrics)
function _rank_deltas(df::DataFrame, metrics::Vector{Symbol}; pair::Union{Nothing,Tuple{Symbol,Symbol}}=nothing)
    chosen = pair === nothing ? (metrics[1], metrics[2]) : pair
    R = _metric_ranks(df, metrics)
    base = DataFrame(Collocate = hasproperty(df, :Collocate) ? df.Collocate : 1:nrow(df))
    base[!, :Δrank] = R[!, Symbol(String(chosen[1]) * "_rank")] .- R[!, Symbol(String(chosen[2]) * "_rank")]
    base[!, :Metric1] = String(chosen[1])
    base[!, :Metric2] = String(chosen[2])
    base
end

# Frequency-banded Spearman correlations
function _banded_spearman(df::DataFrame, metrics::Vector{Symbol};
                          bands = [(0,5)=>"rare", (6,20)=>"mid", (21,typemax(Int))=>"high")
    hasproperty(df, :Frequency) || return DataFrame(Band=String[], Pair=String[], Spearman=Float64[], N=Int[])
    R = _metric_ranks(df, metrics)
    rows = NamedTuple[]
    for (range, label) in bands
        mask = (df.Frequency .>= first(range)) .& (df.Frequency .<= last(range))
        n = sum(mask)
        n == 0 && continue
        for i in 1:length(metrics)-1, j in i+1:length(metrics)
            x = R[mask, Symbol(String(metrics[i]) * "_rank")]
            y = R[mask, Symbol(String(metrics[j]) * "_rank")]
            ρ = length(x) >= 2 ? cor(x, y) : NaN
            push!(rows, (Band=label, Pair="$(metrics[i]) vs $(metrics[j])", Spearman=ρ, N=n))
        end
    end
    DataFrame(rows)
end

# Optionally sample collocates (stratified by frequency bands) for speed
function _maybe_sample(df::DataFrame; sample_fraction::Float64=1.0, seed::Int=0)
    (0.0 < sample_fraction < 1.0) || return df
    rng = seed == 0 ? Random.GLOBAL_RNG : MersenneTwister(seed)

    if hasproperty(df, :Frequency)
        # simple 3-band stratified sample
        bands = [(0,5)=>"rare", (6,20)=>"mid", (21,typemax(Int))=>"high"]
        parts = DataFrame[]
        for (range, _) in bands
            mask = (df.Frequency .>= first(range)) .& (df.Frequency .<= last(range))
            sub = view(df, mask, :)
            k = max(1, round(Int, sample_fraction * nrow(sub)))
            nrow(sub) == 0 ? push!(parts, DataFrame()) :
                push!(parts, sub[rand(rng, 1:nrow(sub), k; replace=false), :])
        end
        return vcat(parts...; cols=:union)
    else
        k = max(1, round(Int, sample_fraction * nrow(df)))
        return df[rand(rng, 1:nrow(df), k; replace=false), :]
    end
end

# ---- Public API --------------------------------------------------------------

"""
    compare_metrics(ct::AssociationDataFormat,
                    metrics::Vector{DataType};
                    ks = [10,25,50],
                    diagnostics = (:corr, :topk, :deltas, :bands),
                    delta_pair::Union{Nothing,Tuple{Symbol,Symbol}}=nothing,
                    bands = [(0,5)=>"rare", (6,20)=>"mid", (21,typemax(Int))=>"high"],
                    early_exit_threshold = 0.95,
                    sample_fraction::Float64 = 1.0,
                    seed::Int = 0) -> MetricComparison

Run cross-metric diagnostics on the same contingency-table results.

- If `ct` is provided, we call `assoc_score(metrics, ct)` to obtain a scores DataFrame.
- Overload exists for `compare_metrics(df::DataFrame, ...)` when you already computed scores.

Early-exit: if the *minimum* pairwise Spearman among metrics ≥ `early_exit_threshold`,
we return only correlations (and metadata noting early-exit).
"""
function compare_metrics(ct::AssociationDataFormat,
                         metrics::Vector{DataType};
                         ks = [10,25,50],
                         diagnostics = (:corr, :topk, :deltas, :bands),
                         delta_pair::Union{Nothing,Tuple{Symbol,Symbol}}=nothing,
                         bands = [(0,5)=>"rare", (6,20)=>"mid", (21,typemax(Int))=>"high"],
                         early_exit_threshold = 0.95,
                         sample_fraction::Float64 = 1.0,
                         seed::Int = 0)
    df = assoc_score(metrics, ct)
    return compare_metrics(df, nothing;
        ks, diagnostics, delta_pair, bands, early_exit_threshold, sample_fraction, seed)
end

function compare_metrics(df::DataFrame,
                         metrics::Union{Nothing,Vector{Symbol}}=nothing;
                         ks = [10,25,50],
                         diagnostics = (:corr, :topk, :deltas, :bands),
                         delta_pair::Union{Nothing,Tuple{Symbol,Symbol}}=nothing,
                         bands = [(0,5)=>"rare", (6,20)=>"mid", (21,typemax(Int))=>"high"],
                         early_exit_threshold = 0.95,
                         sample_fraction::Float64 = 1.0,
                         seed::Int = 0)
    # Detect metric columns if not provided
    metric_syms = metrics === nothing ? _metric_syms_from_df(df) : metrics
    isempty(metric_syms) && error("No metric columns detected. Pass a DataFrame with numeric metric columns or provide `metrics`.")

    # Optional sampling for speed
    df_used = _maybe_sample(df; sample_fraction, seed)

    # Always compute correlations
    spearman = _spearman_corr(df_used, metric_syms)
    kendall  = _kendall_corr(df_used, metric_syms)

    # Early-exit if all metrics essentially agree
    # (use min of upper triangle of Spearman)
    min_spear = minimum(spearman.Spearman)
    early = min_spear ≥ early_exit_threshold

    result = MetricComparison(
        scores = df_used,
        rank_corr = let kdf = kendall
            if nrow(kdf) == 0
                rename!(spearman, :Spearman => :Spearman)  # keep as-is
            else
                # merge spearman + kendall on Metric1/Metric2
                outerjoin(spearman, kdf, on=[:Metric1,:Metric2])
            end
        end,
        metadata = Dict(
            :metrics => metric_syms,
            :ks => ks,
            :bands => bands,
            :sample_fraction => sample_fraction,
            :seed => seed,
            :early_exit_threshold => early_exit_threshold,
            :early_exit_triggered => early
        )
    )

    # If early-exit, return only correlations
    if early
        return result
    end

    # Otherwise compute selected diagnostics
    if :topk in diagnostics
        result.topk_overlap = _topk_overlap(df_used, metric_syms; ks)
    end
    if :deltas in diagnostics
        result.rank_diffs = _rank_deltas(df_used, metric_syms; pair=delta_pair)
    end
    if :bands in diagnostics
        result.by_band_corr = _banded_spearman(df_used, metric_syms; bands)
    end

    return result
end
