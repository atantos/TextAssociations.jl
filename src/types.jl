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


"""
    TextNorm(; strip_case=true,
               strip_accents=false,
               unicode_form=:NFC,
               strip_punctuation=true,
               punctuation_to_space=true,
               normalize_whitespace=true,
               strip_whitespace=false,
               use_prepare=false)

Configuration for text normalization used by `prep_string` and corpus loaders.

# Fields
- `strip_case::Bool` — Lowercase the text when `true`.  
- `strip_accents::Bool` — Remove combining diacritics (e.g., Greek tonos, diaeresis).  
- `unicode_form::Symbol` — Unicode normalization form (`:NFC`, `:NFD`, `:NFKC`, `:NFKD`).  
- `strip_punctuation::Bool` — If `true`, remove punctuation; combined with `punctuation_to_space`
  to decide **replace with a space** vs **delete**.
- `punctuation_to_space::Bool` — When stripping punctuation, replace it with a single space
  (if `true`) or remove it (if `false`).  
- `normalize_whitespace::Bool` — Collapse consecutive whitespace to a single space.  
- `strip_whitespace::Bool` — Trim leading/trailing whitespace.  
- `use_prepare::Bool` — Internal flag to route through a more aggressive prepare path.

# Constructors
- `TextNorm()` — defaults above.  
- `TextNorm(d::Dict)` and `TextNorm(nt::NamedTuple)` — convenience constructors; keys must match field names.

# Notes
In `prep_string`, normalization typically proceeds as: Unicode normalization → punctuation handling →
whitespace normalization, then (if enabled) case folding and accent stripping.

# Examples
```julia
julia> cfg = TextNorm(strip_accents=true, strip_whitespace=true);

julia> doc = prep_string("  Καφέ, naïve résumé!  ", cfg);
julia> text(doc)
"καφε naive resume"
"""
Base.@kwdef struct TextNorm
    strip_case::Bool = true
    strip_accents::Bool = false
    unicode_form::Symbol = :NFC
    strip_punctuation::Bool = true
    punctuation_to_space::Bool = true
    normalize_whitespace::Bool = true
    strip_whitespace::Bool = false
    use_prepare::Bool = false
end

# Convenience constructors
TextNorm(d::Dict) = TextNorm(; pairs(d)...)
TextNorm(nt::NamedTuple) = TextNorm(; pairs(nt)...)


# Define metric types programmatically
const METRIC_TYPES = [
    :PMI, :PMI², :PMI³, :PPMI,
    :LLR, :LLR², :FisherRight,
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
    :LexicalGravity, :DeltaPiLeft, :DeltaPiRight,
    :Tscore, :Zscore, :ChiSquare
]

# 2) Optional: a registry with nice descriptions & flags (fill incrementally)
const _METRIC_REGISTRY = Dict{Symbol,NamedTuple}(
    :PMI => (title="Pointwise Mutual Information",
        short="Association strength via surprisal of co-occurrence.",
        needs_tokens=false,
        family=:information_theoretic,
        ref="Church & Hanks (1990)"),
    :LexicalGravity => (title="Lexical Gravity",
        short="Direction-sensitive association weighting with n⁺/n⁻ context diversity.",
        needs_tokens=true,
        family=:directional,
        ref="Daudaravičius & Marcinkevičienė (2004)"),
    :DeltaPiRight => (title="ΔP Right",
        short="P(Y|X) − P(Y|¬X): rightward influence.",
        needs_tokens=false,
        family=:directional,
        ref="Gries (2013)"),
    :DeltaPiLeft => (title="ΔP Left",
        short="P(X|Y) − P(X|¬Y): leftward influence.",
        needs_tokens=false,
        family=:directional,
        ref="Gries (2013)"),
    :FisherRight => (title="Fisher’s Exact (right-tailed)",
        short="Exact hypergeometric p-value for over-association; often ranked via −log₁₀(p).",
        needs_tokens=false,
        family=:statistical,
        ref="Fisher (1922, 1925)")
    # …extend over time; unknowns get a generic doc
)

# 3) Helper to render a docstring for a given metric Symbol
function _metric_doc(sym::Symbol)
    meta = get(_METRIC_REGISTRY, sym, nothing)
    name = String(sym)
    if meta === nothing
        return """
        $name

        Association metric type used with `assoc_score($name, ...)`.

        !!! note
            This metric is generated programmatically. See the *Metric Catalog* page
            and `assoc_score` docs for usage and available options.
        """
    else
        needs_tokens = meta.needs_tokens ? "Yes" : "No"
        return """
        $(meta.title) — `$name`

        $(meta.short)

        **Use:** `assoc_score($name, data; kwargs...)`

        - **Family:** `$(meta.family)`
        - **Needs tokens?** $needs_tokens
        - **Reference:** $(meta.ref)

        !!! tip
            Implementation is internal (`eval_$(lowercase(name))`), called via `assoc_score`. 
            See `assoc_score` for API and examples.
        """
    end
end

# Generate the abstract type *and* attach the docstring
for metric in METRIC_TYPES
    @eval begin
        abstract type $metric <: AssociationMetric end
        @doc _metric_doc($(QuoteNode(metric))) $metric
    end
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
