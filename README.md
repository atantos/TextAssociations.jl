# TextAssociations

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://atantos.github.io/TextAssociations.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://atantos.github.io/TextAssociations.jl/dev/)
[![Build Status](https://github.com/atantos/TextAssociations.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/atantos/TextAssociations.jl/actions/workflows/CI.yml?query=branch%3Amain)

# Introduction

`TextAssociations.jl` is a `Julia` package designed for text analysis, focusing on the calculation of association metrics between words (nodes) and their collocates within textual data. This package offers a comprehensive suite of tools for analyzing the strength and nature of a comprehensive set of association measures, facilitating deeper insights into text structure, syntagmatic relations, and word combination patterns.

## Core Features

**Association Metrics**: Compute a comprehensive set of association metrics based on the corpus/text analysis literature, such as Pointwise Mutual Information (PMI), Dice Coefficient, Jaccard Index, and many others, to quantify the association strength between words. For more on the list of association metrics, see [below](#supported-metrics) on the Supported metrics section.

**Contingency Table Support**: Utilize `ContingencyTable` structures to efficiently organize and analyze word co-occurrence data.

**Flexible Data Input**: Support for diverse data formats, enabling analyses on raw texts, pre-processed corpora, or custom data structures.

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

### Create a ContingencyTable

First step is to create a `ContingencyTable` instance that prepares the textual data and returns the contingency tables of the node word with all its collocates found within a window size that co-occur to a minimum frequency with the node word. You need to specify the node word, the window size and minimum frequency that will be considered, while creating the contingency table.


```julia
julia> text_data = "Your text data here..."
julia> node_word = "example"
julia> window_size = 5
julia> min_frequency = 3
julia> cont_table = ContingencyTable(text_data, node_word, window_size, min_frequency)
```

### Evaluate association scores

The second step is to use the `evalassoc()` function that takes an association metric as its first argument and the `ContingencyTable` instance created in the previous step. 

```julia
julia> evalassoc(PMI, cont_table)
```

### Supported Metrics

`TextAssociations.jl` provides a wide range of metrics for analyzing word co-occurrence associations. To see all available metrics, use the `listmetrics()` function:

```julia
julia> show(listmetrics())
[:PMI,
:PMI2,
:PMI3,
:PPMI,
:LLR,
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
|Pearson's $\chi^2$ test |  `ChiSquare` | $`\sum_{i, j}\frac{\left(f_{i j}-\hat{f}_{i j}\right)^2}{\hat{f}_{i j}}`$ |


### Further Exploration

If you are interested in diving into the maths of these metrics or in exploring advanced usage scenarios, visit the package [documentation](https://atantos.github.io/TextAssociations.jl/dev/). 

## Aims & scope

`TextAssociations.jl` aims to be a robust and user-friendly tool for humanities researchers, such as (corpus) linguists, philologists, historians or literary scholars and language data analysts, and anyone interested in word co-occurrence analysis. 
