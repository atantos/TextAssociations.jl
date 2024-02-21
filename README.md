# TextAssociations

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://atantos.github.io/TextAssociations.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://atantos.github.io/TextAssociations.jl/dev/)
[![Build Status](https://github.com/atantos/TextAssociations.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/atantos/TextAssociations.jl/actions/workflows/CI.yml?query=branch%3Amain)

# Introduction

`TextAssociations.jl` is a `Julia` package designed for text analysis, focusing on the calculation of association metrics between words (nodes) and their collocates within textual data. This package offers a comprehensive suite of tools for analyzing the strength and nature of a comprehensive set of association measures, facilitating deeper insights into text structure, syntagmatic relations, and word combination patterns.

## Core Features

**Association Metrics**: Compute a complete set of association metrics like Pointwise Mutual Information (PMI), Dice Coefficient, Jaccard Index, and many others, to quantify the association strength between words.

**Contingency Table Support**: Utilize `ContingencyTable` structures to efficiently organize and analyze word co-occurrence data.

**Flexible Data Input**: Support for diverse data formats, enabling analyses on raw texts, pre-processed corpora, or custom data structures.

**Performance Optimization**: Leverages Julia's high-performance computing capabilities, offering fast computations even on large text datasets.

**Extensibility**: Designed with extensibility in mind, allowing for the integration of additional metrics and data formats.

## Getting Started

To begin using `TextAssociations.jl`, install the package through Julia's package manager and import it into your project:

```
using Pkg
Pkg.add("TextAssociations")

using TextAssociations
```

## Basic Usage

At the heart of `TextAssociations.jl` is the `evalassoc()` function, which dynamically evaluates association metrics based on the provided metric type and data encapsulated in a `ContingencyTable`. Here's a simple example to calculate the PMI between a node word and its collocates:

### Create a ContingencyTable

Prepare your textual data, specify the node word, window size and minimum frequency that will be considered while creating the contingency table.


```julia
text_data = "Your text data here..."
node_word = "example"
window_size = 5
min_frequency = 3
cont_table = ContingencyTable(text_data, node_word, window_size, min_frequency)
```

### Evaluate PMI

```julia
pmi_result = evalassoc(PMI, cont_table)
```

### Supported Metrics

`TextAssociations.jl` provides a wide range of metrics for analyzing text associations. To see all available metrics, use the `listmetrics()` function:

```julia
metrics_list = listmetrics()
println(metrics_list)
```

Each metric offers unique insights into the text, from simple co-occurrence frequencies to complex statistical measures that reveal deeper semantic or syntactic patterns.

### Further Exploration

For those interested in the mathematical foundations of these metrics or in exploring advanced usage scenarios, we encourage diving into the detailed documentation provided with the package. This includes in-depth discussions on each metric, examples of advanced analyses, and tips for optimizing performance.

## Conclusion

`TextAssociations.jl` aims to be a robust and user-friendly tool for linguists, data scientists, and anyone interested in text analysis. By providing a rich set of functionalities for association metric calculations, it opens up new possibilities for understanding and exploring textual data.
