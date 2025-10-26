# API Overview

## 🧭 How *TextAssociations.jl* Evaluates a Metric

*TextAssociations.jl* follows a transparent, two-layer model for computing word-association scores.  
No matter how you start—raw text, corpus, or contingency table—everything flows through a single unified pipeline.

---

### 1️⃣ Input levels

You can start from **any** of these representations:

| Input | Example | What happens internally |
|:------|:---------|:------------------------|
| **Raw text** | `"It is a truth universally acknowledged..."` | A `ContingencyTable` is built around the target node with the specified `windowsize`, `minfreq`, and normalization settings. |
| **Corpus object** | `corpus = read_corpus("data_austen"; preprocess=true, norm_config=norm)` | For each node, a `CorpusContingencyTable` (CCT) is built — merging co-occurrence counts across all documents. |
| **Prebuilt contingency data** | `ct = ContingencyTable(corpus, "might"; windowsize=5)` | Used directly — no text parsing. Ideal for reusing tables or parallel scoring. |

All of these objects implement the `AssociationDataFormat` interface, which gives the scorer a consistent view of the data.

---

### 2️⃣ Delegation chain

```
Raw Text / Corpus
      ↓
  ContingencyTable (CT)
      ↓
CorpusContingencyTable (CCT)
      ↓
assoc_score(::Type{<:AssociationMetric}, x::AssociationDataFormat; kwargs...)
      ↓
Metric evaluator (eval_pmi, eval_llr, eval_bayesllr, eval_logdice, …)
      ↓
Result as DataFrame  (or Vector if scores_only=true)
```

Each overload of `assoc_score` simply **delegates**:

- It constructs the appropriate CT or CCT.
- Then it calls the **core** `assoc_score(::Type{T}, x::AssociationDataFormat; ...)`.
- That function resolves the correct evaluator function, for example `eval_pmi` or `eval_llr`.

---


### 3️⃣ Node analyzers: `analyze_node` and `analyze_nodes`

While `assoc_score` focuses on **scoring collocates** for one node and (optionally) multiple metrics, the **analyzers** are higher-level convenience functions that bundle common steps and summaries.

#### What they do (conceptually)

- Build the appropriate contingency table (CT/CCT) for each node.
- Run one or more association metrics (internally calling `assoc_score`).
- Optionally apply sorting/selection (e.g., top-N).
- Return a tidy, user-facing result (single DataFrame or a Dict of DataFrames), with the same metadata conventions (`"status"`, `"message"`, `"node"`, `"windowsize"`, `"metrics"`).

#### When to use them
- **Use `assoc_score`** when you want tight control over which metric(s) run and you’ll handle post-processing yourself.
- **Use `analyze_node` / `analyze_nodes`** when you want the common “analyze this word / these words” workflow in one call, including sorting and selecting top collocates.

#### Typical signatures (illustrative)

```julia
# Single node, one or many metrics
analyze_node(::Type{<:AssociationMetric}}, x::AssociationDataFormat; kwargs...)      -> DataFrame
analyze_node(AbstractVector{<:Type{<:AssociationMetric}}, x::AssociationDataFormat; kwargs...) -> DataFrame

# Multiple nodes (returns a Dict)
analyze_nodes(::Type{<:AssociationMetric}}, x::AssociationDataFormat, nodes::Vector{String}; kwargs...) -> Dict{String,DataFrame}
analyze_nodes(AbstractVector{<:Type{<:AssociationMetric}}, x::AssociationDataFormat, nodes::Vector{String}; kwargs...) -> Dict{String,DataFrame}
```


### 4️⃣ Keyword layers

| Layer | Keywords | Description |
|:------|:----------|:------------|
| **Table construction** | `windowsize`, `minfreq`, `norm_config` | Control how co-occurrence contexts are collected. |
| **Metric evaluation** | `λ`, `base`, `direction`, `tokens`, `scores_only` | Affect how the metric is calculated. |
| **Output control** | `scores_only`, `top_n`, `sort_by` | Shape or filter the result, but not the math itself. |

All keywords are safely forwarded to the correct layer.  
Metrics that need tokens (e.g. `LexicalGravity`) are handled automatically through the `NeedsTokens` trait.

---

### 5️⃣ Return values & metadata

| Return type | Trigger | Description |
|:-------------|:---------|:-------------|
| `DataFrame` (default) | `scores_only=false` | Columns: `Node`, `Collocate`, `Frequency`, `<MetricName>` |
| `Vector{Float64}` | `scores_only=true` | Raw score values aligned with the collocate order |
| `Dict{String,DataFrame}` | Multi-node call | One table per node, optionally trimmed to `top_n` |

Each table carries embedded metadata:

| Metadata key | Meaning |
|:--------------|:---------|
| `"status"` | `"ok"`, `"empty"`, or `"error"` |
| `"message"` | Context or diagnostic (e.g. “Node not found.”) |
| `"node"` | The target word |
| `"windowsize"` | Context window used |
| `"metrics"` | Metrics evaluated (for multi-metric tables) |

Inspect with:

```julia
using DataFrames
metadata(df)
```

---

### 6️⃣ Mental model summary

> **Everything becomes a Contingency Table.**

Whether you begin with raw text, a corpus, or a precomputed table,  
the scorer always sees a uniform `AssociationDataFormat` object.  
From there, the metric evaluator handles the rest—robustly, transparently, reproducibly.

---

### 7️⃣ Quick examples

```julia
# From raw text
assoc_score(PMI, "It is a truth...", "truth"; windowsize=5, minfreq=2)

# From a corpus
assoc_score(LogDice, corpus, "love"; windowsize=4)

# Multiple metrics
assoc_score([PMI, LLR, LogDice], corpus, "world"; windowsize=5, minfreq=2)

# Token-requiring metric (Lexical Gravity)
assoc_score(LexicalGravity, corpus, "beautiful"; windowsize=5, tokens=mytokens)
```

All return standardized `DataFrame`s with attached metadata.

---

### 8️⃣ Design philosophy

- **Transparent:** identical scoring logic regardless of input type.  
- **Composable:** CT/CCT objects can be reused or serialized.  
- **Safe:** empty or failed evaluations never crash—always return a diagnostic table.  
- **Extensible:** new metrics only need an `eval_*` function and, if required,  
  a `NeedsTokens(::Type{YourMetric}) = Val(true)` specialization.

---

> Once you understand this flow, you understand the entire package.  
> Every advanced feature—comparison, graphing, temporal analysis—builds on this backbone.
