<p align="center">
  <img src="https://github.com/atantos/TextAssociations.jl/raw/main/assets/TextAssociations_logo.gif" alt="TextAssociations.jl" width="1100" height="500"/>
</p>


# TextAssociations

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://atantos.github.io/TextAssociations.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://atantos.github.io/TextAssociations.jl/dev/)
[![Build Status](https://github.com/atantos/TextAssociations.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/atantos/TextAssociations.jl/actions/workflows/CI.yml?query=branch%3Amain)

# Introduction

`TextAssociations.jl` is a `Julia` package designed for text analysis, focusing on the calculation of association metrics between words of interest -usually called node words- and their collocates within a prespecified window of a few words found in textual data. This package is planned to offer a comprehensive suite of tools for analyzing the strength and nature of a large set of association measures, facilitating deeper insights into text structure, syntagmatic relations, and word combination patterns.

Word-word association measures still play an important role in various aspects of natural language processing (NLP) and computational linguistics, even in the era of word2vec and transformer-based models. There are several reasons why associatio measures are still relevant, nowadays. Word-word association measures:

- are often more interpretable and transparent than the dense, high-dimensional vectors produced by neural models. They provide clear insights into why words are considered related based on observable statistics.
- can be used alongside neural embeddings to enhance performance. For example, they can help refine or interpret the relationships captured by word vectors.
- may serve as benchmarks to evaluate the performance of more complex models. They provide a baseline against which the improvements offered by word2vec or transformers can be quantified.
- may still provide useful information when neural models fail to generalize due to insufficient training examples.

## Core Features

**Association Metrics**: Compute a large set of association metrics based on the corpus/text analysis literature, such as Pointwise Mutual Information (PMI), Dice Coefficient, Jaccard Index, and many others, to quantify the association strength between words. For more on the list of the supported association metrics, see the relevant section [below](#supported-metrics).

**Contingency Table Support**: Utilize the `ContingencyTable` structure to efficiently organize and analyze word co-occurrence data.

**Flexible Data Input**: Support for diverse data formats, includeing raw texts, `StringDocument`s or filepaths to text files. Moreover, data import from pre-processed `CorpusDocument`s is planned.

**Extensibility**: Designed with extensibility in mind, allowing for the integration of additional metrics and data formats.

## Getting Started

To begin using `TextAssociations.jl`, install the package through Julia's package manager and import it into your project:

```julia
julia> using Pkg
julia> Pkg.add("https://github.com/atantos/TextAssociations.jl")

julia> using TextAssociations
```

## Basic Usage

At the heart of `TextAssociations.jl` is the `evalassoc()` function, which evaluates association metrics based on the provided metric type and data encapsulated in a `ContingencyTable`. The result is a vector of the node words' collocates and their association score on the selected metric. Here's a simple example to calculate the `PMI` between a node word and its collocates:

### Step 1: Create a ContingencyTable

First step is to create a `ContingencyTable` instance that prepares the textual data and returns the contingency tables of the node word with all its collocates found within a window size that co-occur to a minimum frequency with the node word. You need to specify the node word, the window size and minimum frequency that will be considered, while creating the contingency table.

```julia
julia> text_data = "Your text data here..."
julia> node_word = "example"
julia> window_size = 5
julia> min_frequency = 3
julia> cont_table = ContingencyTable(text_data, node_word, window_size, min_frequency)
```

### Step 2: Evaluate association scores

The second step is to use the `evalassoc()` function that takes an association metric as its first argument and the `ContingencyTable` instance created in the previous step as its second argument.

```julia
julia> evalassoc(PMI, cont_table)
```

### Supported Metrics

`TextAssociations.jl` supports a wide range of metrics for analyzing word co-occurrence associations based on the relevant literature. To see all available metrics, use the `listmetrics()` function:

```julia
julia> show(listmetrics())
[:PMI,
:PMI²,
:PMI³,
:PPMI,
:LLR,
:LLR2,
:LLR²,
:DeltaPi,
:MinSens,
:Dice,
:LogDice,
:RelRisk,
:LogRelRisk,
:RiskDiff,
:AttrRisk,
:OddsRatio,
:LogRatio,
:LogOddsRatio,
:JaccardIdx,
:OchiaiIdx,
:PiatetskyShapiro,
:YuleOmega,
:YuleQ,
:PhiCoef,
:CramersV,
:TschuprowT,
:ContCoef,
:CosineSim,
:OverlapCoef,
:KulczynskiSim,
:TanimotoCoef,
:RogersTanimotoCoef,
:RogersTanimotoCoef2,
:HammanSim,
:HammanSim2,
:GoodmanKruskalIdx,
:GowerCoef,
:GowerCoef2,
:CzekanowskiDiceCoef,
:SorgenfreyIdx,
:SorgenfreyIdx2,
:MountfordCoef,
:MountfordCoef2,
:SokalSneathIdx,
:SokalMichenerCoef]
```

Each association metric offers insights as to the association of a node word of interest to a collocate word that reveals deeper semantic or syntactic patterns.

| Association Metric                            | Metric Type           | Math formula                                                                                                                             |
| --------------------------------------------- | --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| $\text{Pearson's}  \chi  \text{Squared Test}$ | `ChiSquare`           | $`\sum_{i, j}\frac{\left(f_{i j}-\hat{f}_{i j}\right)^2}{\hat{f}_{i j}}`$                                                                |
| $\text{Pointwise Mutual Information (PMI)}$   | `PMI`                 | $`\log \left( \frac{P(x, y)}{P(x) \cdot P(y)} \right)`$                                                                                  |
| $\text{Squared PMI}$                          | `PMI²`                | $`\left( \log \left( \frac{P(x, y)}{P(x) \cdot P(y)} \right) \right)^2`$                                                                 |
| $\text{Cubed PMI}$                            | `PMI³`                | $`\left( \log \left( \frac{P(x, y)}{P(x) \cdot P(y)} \right) \right)^3`$                                                                 |
| $\text{Positive PMI}$                         | `PPMI`                | $`\max\left(0, \log \left( \frac{P(x, y)}{P(x) \cdot P(y)} \right)\right)`$                                                              |
| $\text{Log Likelihood Ratio}$                 | `LLR`                 | $`2 \sum_{i,j} \left( O_{ij} \ln \left( \frac{O_{ij}}{E_{ij}} \right) \right)`$                                                          |
| $\text{Log Likelihood Ratio 2}$               | `LLR2`                | $`2 \sum_{i,j} \left( O_{ij} \ln \left( \frac{O_{ij}}{E_{ij}} \right) \right) + 2 \left( \sum_{i,j} O_{ij} - \sum_{i,j} E_{ij} \right)`$ |
| $\text{Squared LLR}$                          | `LLR²`                | $`\sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}`$                                                                                        |
| $\Delta \pi$                                  | `DeltaPi`             | $`\frac{a}{a + b} - \frac{c}{c + d}`$                                                                                                    |
| $\text{Minimum Sensitivity}$                  | `MinSens`             | $`\min\left(\frac{a}{a + b}, \frac{d}{c + d}\right)`$                                                                                    |
| $\text{Dice}$                                 | `Dice`                | $`\frac{2a}{2a + b + c}`$                                                                                                                |
| $\text{Log Dice}$                             | `LogDice`             | $`14 + \log_2\left(\frac{2a}{2a + b + c}\right)`$                                                                                        |
| $\text{Relative Risk}$                        | `RelRisk`             | $`\frac{\frac{a}{a + b}}{\frac{c}{c + d}}`$                                                                                              |
| $\text{Log Relative Risk}$                    | `LogRelRisk`          | $`\log\left(\frac{\frac{a}{a + b}}{\frac{c}{c + d}}\right)`$                                                                             |
| $\text{Risk Difference}$                      | `RiskDiff`            | $`\frac{a}{a + b} - \frac{c}{c + d}`$                                                                                                    |
| $\text{Attributable Risk}$                    | `AttrRisk`            | $`\frac{a}{a + b} - \frac{c}{c + d}`$                                                                                                    |
| $\text{Odds Ratio}$                           | `OddsRation`          | $`\frac{a \cdot d}{b \cdot c}`$                                                                                                          |
| $\text{Log Odds Ratio}$                       | `LogOddsRatio`        | $`\log\left(\frac{a \cdot d}{b \cdot c}\right)`$                                                                                         |
| $\text{Jaccard Index}$                        | `JaccardIdx`          | $`\frac{a}{a + b + c}`$                                                                                                                  |
| $\text{Ochiai Index}$                         | `OchiaiIdx`           | $`\frac{a}{\sqrt{(a + b)(a + c)}}`$                                                                                                      |
| $\text{Piatetsky-Shapiro}$                    | `PiatetskyShapiro`    | $`\frac{a}{n} - \frac{(a + b)(a + c)}{n^2}`$                                                                                             |
| $\text{Yule's Omega}$                         | `YuleOmega`           | $`\frac{\sqrt{a \cdot d} - \sqrt{b \cdot c}}{\sqrt{a \cdot d} + \sqrt{b \cdot c}}`$                                                      |
| $\text{Yule's Q}$                             | `YuleQ`               | $`\frac{a \cdot d - b \cdot c}{a \cdot d + b \cdot c}`$                                                                                  |
| $\phi$                                        | `PhiCoef`             | $`\frac{a \cdot d - b \cdot c}{\sqrt{(a + b)(c + d)(a + c)(b + d)}}`$                                                                    |
| $\text{Cramer's V}$                           | `CramersV`            | $`\sqrt{\frac{\phi^2}{\min(1, 1)}} = \sqrt{\phi^2} = \|\phi\|`$                                                                          |
| $\text{Tschuprow's T}$                        | `TschuprowT`          | $`\sqrt{\frac{\chi^2}{n \cdot \sqrt{(k - 1)(r - 1)}}}`$                                                                                  |
| $\text{Contigency Coefficient}$               | `ContCoef`            | $`\sqrt{\frac{\chi^2}{\chi^2 + n}}`$                                                                                                     |
| $\text{Cosine Similarity}$                    | `CosineSim`           | $`\frac{a}{\sqrt{(a + b)(a + c)}}`$                                                                                                      |
| $\text{Overlap Coefficient}$                  | `OverlapCoef`         | $`\frac{a}{\min(a + b, a + c)}`$                                                                                                         |
| $\text{Kulczynski Similarity}$                | `KulczynskiSim`       | $`\frac{a}{a + b} + \frac{a}{a + c}`$                                                                                                    |
| $\text{Tanimoto Coefficient}$                 | `TanimotoCoef`        | $`\frac{a}{a + b + c}`$                                                                                                                  |
| $\text{Rogers-Tanimoto Coefficient}$          | `RogersTanimotoCoef`  | $`\frac{a}{a + 2(b + c)}`$                                                                                                               |
| $\text{Rogers-Tanimoto Coefficient 2}$        | `RogersTanimotoCoef2` | $`\frac{a + d}{a + 2(b + c) + d}`$                                                                                                       |
| $\text{Hamman Similarity}$                    | `HammanSim`           | $`\frac{a + d - b - c}{N}`$                                                                                                              |
| $\text{Hamman Similarity 2}$                  | `HammanSim2`          | $`\frac{a - d}{a + b + c - d}`$                                                                                                          |
| $\text{Goodman-Kruskal Index } \gamma$        | `GoodmanKruskalIdx`   | $`\frac{a - d}{a + d}`$                                                                                                                  |
| $\text{Gower's Coefficient}$                  | `GowerCoef`           | $`\frac{a}{a + b + c}`$                                                                                                                  |
| $\text{Gower's Coefficient 2}$                | `GowerCoef2`          | $`\frac{a + d}{a + d + 2(b + c)}`$                                                                                                       |
| $\text{Czekanowski Dice Coefficient}$         | `CzekanowskiDiceCoef` | $`\frac{2a}{2a + b + c}`$                                                                                                                |
| $\text{Sorgenfrey Index}$                     | `SorgenfreyIdx`       | $`\frac{2a - b - c}{2a + b + c}`$                                                                                                        |
| $\text{Sorgenfrey Index 2}$                   | `SorgenfreyIdx2`      | $`\frac{a + d}{2(a + d) + b + c}`$                                                                                                       |
| $\text{Mountford's Coefficient}$              | `MountfordCoef`       | $`\frac{a}{a + 2b + 2c}`$                                                                                                                |
| $\text{Mountford's Coefficient 2}$            | `MountfordCoef2`      | $`\frac{a + d}{a + d + 2 \sqrt{(b + c) \cdot (k + m)}}`$                                                                                 |
| $\text{Sokal-Sneath Index}$                   | `SokalSneathIdx`      | $`\frac{a}{a + 2(b + c)}`$                                                                                                               |
| $\text{Sokal-Michener Coefficient}$           | `SokalMichenerCoef`   | $`\frac{a +d}{a + b + c + d}`$                                                                                                           |

### Further Exploration

If you are interested in diving into the maths of these metrics or in exploring advanced usage scenarios, visit the package [documentation](https://atantos.github.io/TextAssociations.jl/dev/).

## Aims & scope

`TextAssociations.jl` aims to be a robust and user-friendly tool for humanities researchers, such as (corpus) linguists, philologists, historians or literary scholars and language data analysts in general, and anyone interested in word co-occurrence analysis.
