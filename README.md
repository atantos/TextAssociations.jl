# TextAssociations

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://atantos.github.io/TextAssociations.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://atantos.github.io/TextAssociations.jl/dev/)
[![Build Status](https://github.com/atantos/TextAssociations.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/atantos/TextAssociations.jl/actions/workflows/CI.yml?query=branch%3Amain)

# Introduction

`TextAssociations.jl` is a `Julia` package designed for text analysis, focusing on the calculation of association metrics between words -usually called node words- and their collocates within a prespecified window of a few words found in textual data. This package is planned to offer a comprehensive suite of tools for analyzing the strength and nature of a large set of association measures, facilitating deeper insights into text structure, syntagmatic relations, and word combination patterns.

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
:PMI2,
:PMI3,
:PPMI,
:LLR,
:LLR2,
:LLR²,
:DeltaPi,
:Dice,
:LogDice,
:RelRisk,
:LogRelRisk,
:RiskDiff,
:AttrRisk,
:OddsRatio,
:LogRatio,
:LogOddsRatio,
:JaccardIndex,
:OchiaiIndex,
:OchiaiCoef,
:PiatetskyShapiro,
:YuleQ,
:YuleY,
:PhiCoef,
:CramersV,
:TschuprowT,
:ContCoef,
:CosineSim,
:OverlapCoef,
:KulczynskiSim,
:TanimotoCoef,
:GoodmanKruskalIndex,
:GowerCoef,
:CzekanowskiDiceCoef,
:SorgenfreyIndex,
:MountfordCoef,
:SokalSneathIndex,
:RogersTanimotoCoef,
:SokalmMchenerCoef,
:Tscore,
:Zscore,
:ChiSquare,
:FisherExactTest]
```

Each association metric offers insights as to the association of a node word of interest to a collocate word that reveals deeper semantic or syntactic patterns.

| Association Metric | Metric Type | Math formula |
| ------------------ | ----------- | ------------ | 
| $\text{Pearson's}  \chi  \text{Squared Test}$ |  `ChiSquare` | $`\sum_{i, j}\frac{\left(f_{i j}-\hat{f}_{i j}\right)^2}{\hat{f}_{i j}}`$ |
| $\text{Pointwise Mutual Information (PMI)}$ | `PMI` |$`\log \left( \frac{P(x, y)}{P(x) \cdot P(y)} \right)`$|
| $\text{Squared PMI}$ | `PMI²` |$`\left( \log \left( \frac{P(x, y)}{P(x) \cdot P(y)} \right) \right)^2`$|
| $\text{Cubed PMI}$ | `PMI³` |$`\left( \log \left( \frac{P(x, y)}{P(x) \cdot P(y)} \right) \right)^3`$|
| $\text{Positive PMI}$ | `PPMI` |$`\max\left(0, \log \left( \frac{P(x, y)}{P(x) \cdot P(y)} \right)\right)`$|
| $\text{Log Likelihood Ratio}$ | `LLR` | $`2 \sum_{i,j} \left( O_{ij} \ln \left( \frac{O_{ij}}{E_{ij}} \right) \right)`$|
| $\text{Log Likelihood Ratio 2}$ | `LLR2` | $`2 \sum_{i,j} \left( O_{ij} \ln \left( \frac{O_{ij}}{E_{ij}} \right) \right) + 2 \left( \sum_{i,j} O_{ij} - \sum_{i,j} E_{ij} \right)`$|
| $\text{Squared LLR}$ | `LLR²` |$`\sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}`$|
| $\Delta \pi$ | `DeltaPi` | $`\frac{a}{a + b} - \frac{c}{c + d}`$|
| $\text{Minimum Sensitivity}$ | `MinSens` | $`\min\left(\frac{a}{a + b}, \frac{d}{c + d}\right)`$|
| $\text{Dice}$ | `Dice` | $`\frac{2a}{2a + b + c}`$|
| $\text{Log Dice}$ | `LogDice` | $`14 + \log_2\left(\frac{2a}{2a + b + c}\right)`$|
| $\text{Relative Risk}$ | `RelRisk` | $`\frac{\frac{a}{a + b}}{\frac{c}{c + d}}`$|
| $\text{Log Relative Risk}$ | `LogRelRisk` | $`\log\left(\frac{\frac{a}{a + b}}{\frac{c}{c + d}}\right)`$|
| $\text{Risk Difference}$ | `RiskDiff` | $`\frac{a}{a + b} - \frac{c}{c + d}`$|
| $\text{Attributable Risk}$ | `AttrRisk` | $`\frac{a}{a + b} - \frac{c}{c + d}`$|
| $\text{Odds Ratio}$ | `OddsRation` | $`\frac{a \cdot d}{b \cdot c}`$|
| $\text{Log Odds Ratio}$ | `LogOddsRatio` | $`\log\left(\frac{a \cdot d}{b \cdot c}\right)`$|
| $\text{Jaccard Index}$ | `JaccardIdx` | $`\frac{a}{a + b + c}`$|
| $\text{Ochiai Index}$ | `OchiaiIdx` | $`\frac{a}{\sqrt{(a + b)(a + c)}}`$|

### Further Exploration

If you are interested in diving into the maths of these metrics or in exploring advanced usage scenarios, visit the package [documentation](https://atantos.github.io/TextAssociations.jl/dev/). 

## Aims & scope

`TextAssociations.jl` aims to be a robust and user-friendly tool for humanities researchers, such as (corpus) linguists, philologists, historians or literary scholars and language data analysts in general, and anyone interested in word co-occurrence analysis. 
