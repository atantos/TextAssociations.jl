# Installation

```@meta
CurrentModule = TextAssociations
```

## Requirements

- Julia **1.11** 
- **8 GB RAM** recommended (4 GB minimum)

---

## Install (Package not yet registered)

Install directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/atantos/TextAssociations.jl")
```

> When the package gets registered, you’ll be able to do: `Pkg.add("TextAssociations")`.

---

## Quick check

Once installation completes successfully, you can run a basic smoke test to confirm that the package loads and its core functions work correctly:

```julia
using TextAssociations

# Basic smoke test from raw text
text = "The quick brown fox jumps over the lazy dog."
df = assoc_score(PMI, text, "the"; windowsize=3, minfreq=1)
@show first(df, min(5, nrow(df)))
```

This snippet verifies that:
- The package precompiles without errors.
- The `assoc_score` API works directly from raw text input.
- The output is a DataFrame of collocates with PMI scores.

--- 

## Indicative corpus analysis workflow:

Once you’ve confirmed the installation, try a minimal corpus-level workflow.
This example shows how to load a small dataset, analyze one node word, and export the results:


```julia
using TextAssociations

corpus = read_corpus("path/to/data.csv"; text_column=:text)  # or a folder of .txt files
res = analyze_node(corpus, "important", PMI; windowsize=5, minfreq=5)

CSV.write("path/to/output_file.csv", res)
```

This verifies that:

Corpus loading (`read_corpus`) and normalization pipelines run correctly. Association metrics (here, `PMI`) execute over corpus-level contingency tables. 

---

## Run the test suite

To run all automated package tests:

```julia
using Pkg
Pkg.test("TextAssociations")
```

---

## Environments (recommended)

Using a dedicated `Julia` environment is the best way to manage dependencies for your projects.
It keeps the package versions used by `TextAssociations.jl` isolated from those of other projects and makes your setup fully reproducible on another machine or by collaborators.

You can create and activate a clean environment as follows:

On the system terminal, first create a new folder and move into it. This will serve as your project directory:

```bash
mkdir("TextAssocDemo") # creates a new folder that will host your project
cd("TextAssocDemo") # changes the working directory into that folder
```

These commands simply prepare an empty workspace where `Julia` will store the project’s `Project.toml` and `Manifest.toml` files.

Then, on the `Julia` `REPL`, activate that directory as your working environment and install the package:

```julia
using Pkg; Pkg.activate(".")
Pkg.add(url="https://github.com/atantos/TextAssociations.jl")
# later: Pkg.instantiate()  # reproduce
```

This setup:

- Prevents version conflicts with other `Julia` packages you may use.
- Lets you reproduce your exact software environment via `Project.toml` and `Manifest.toml`.
- Simplifies sharing your workflow with others (or with future you).

---

## Next steps

- [Quick Tutorial](@ref getting_started_tutorial)
- [Basic Examples](@ref getting_started_examples)
- [Core Concepts](@ref guide_concepts)
