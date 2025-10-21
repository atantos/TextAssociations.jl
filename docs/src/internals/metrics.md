# Metric Implementations (Internal)

```@meta
CurrentModule = TextAssociations
```

These evaluator functions (`eval_*`) implement the mathematical formulas for each association metric.
They are **not part of the public API** and are normally called **indirectly via**
[`assoc_score`](@ref).

Each `eval_*` function receives an `AssociationDataFormat` (e.g., `ContingencyTable`) and returns
a vector of scores aligned with the rows of `assoc_df(data)`.

For metrics requiring contextual tokens (e.g., Lexical Gravity), the `tokens` argument
is automatically passed by `assoc_score` based on internal traits (`NeedsTokens(::Type{T})`).

---

## Evaluators by source file

```@autodocs
Modules = [TextAssociations]
Pages = [
  "metrics/directional.jl",
  "metrics/information_theoretic.jl",
  "metrics/similarity.jl",
  "metrics/statistical.jl",
  "metrics/epidemiological.jl",
]
Private = true
```
