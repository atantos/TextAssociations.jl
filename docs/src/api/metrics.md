# [Metric Functions](@id api_metrics)

```@meta
CurrentModule = TextAssociations
DocTestSetup = quote
    using TextAssociations
end
```

## How to score a metric

All metrics are called via `assoc_score(::Type{<:AssociationMetric}, data; kwargs...)`.
You pass a **metric type** (e.g., `DeltaPiRight`, `LexicalGravity`) and a data container (e.g., `ContingencyTable`).

```julia
using TextAssociations

# A tiny toy example
txt = "a b a c a b a b c"
ct  = ContingencyTable(txt, "a"; windowsize=2, minfreq=1)

# Compute ΔP Right (rightward influence)
scores = assoc_score(DeltaPiRight, ct; scores_only=true)

# One score per collocate row
length(scores) == nrow(assoc_df(ct))
```

### Notes

- Some metrics require access to the original `tokens` (e.g., Lexical Gravity). When needed, `assoc_score` injects them automatically based on internal traits.
- Results are aligned to rows of `assoc_df(data)` (one score per collocate).

## Common metric types

> These are the **types** you pass to `assoc_score`. For the full list, see **Metrics → Metric Catalog**.

```@docs
TextAssociations.LexicalGravity
TextAssociations.DeltaPiLeft
TextAssociations.DeltaPiRight
TextAssociations.PMI
TextAssociations.PPMI
TextAssociations.LLR
TextAssociations.LogDice
TextAssociations.Tscore
TextAssociations.Zscore
TextAssociations.ChiSquare
```

## See also

- [`assoc_score`](@ref)
- **Metrics → Metric Catalog** (auto-generated list of all available metrics)
- **Internals → Metric Implementations** for the underlying `eval_*` functions
