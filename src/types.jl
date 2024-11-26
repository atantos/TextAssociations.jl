abstract type AssociationMetric <: SemiMetric end

abstract type AssociationDataFormat end

# List of metric names
metric_names = [:PMI, :PMI², :PMI³, :PPMI, :LLR, :LLR2, :LLR², :DeltaPi, :MinSens, :Dice, :LogDice, :RelRisk, :LogRelRisk, :RiskDiff, :AttrRisk, :OddsRatio, :LogRatio, :LogOddsRatio, :JaccardIdx, :OchiaiIdx, :PiatetskyShapiro, :YuleOmega, :YuleQ, :PhiCoef, :CramersV, :TschuprowT, :ContCoef, :CosineSim, :OverlapCoef, :KulczynskiSim, :TanimotoCoef, :RogersTanimotoCoef, :RogersTanimotoCoef2, :HammanSim, :HammanSim2, :GoodmanKruskalIdx, :GowerCoef, :GowerCoef2, :CzekanowskiDiceCoef, :SorgenfreyIdx, :SorgenfreyIdx2, :MountfordCoef, :MountfordCoef2, :SokalSneathIdx, :SokalMichenerCoef, :Tscore, :Zscore, :ChiSquare, :FisherExactTest, :CohensKappa]

# Programmatically define an abstract type for each metric
for name in metric_names
    @eval abstract type $name <: AssociationMetric end
end



"""
    PMI(inputstring)

Calculate the Pointwise Mutual Information (PMI) score based on the following formula: log2(O₁₁/(E₁₁*span))

```math
\text{log}_{2}\frac{O_{11}}{E_{11} \ast S}
```

```jldoc

```

"""
# struct PMI <: AssociationMetric end

# Caching the text preprocessing for co-occurrence matrix and the countmap() for the cells of the table that will be counting https://discourse.julialang.org/t/best-practice-approach-for-caching-data-in-objects/20419?u=alex_tantos
mutable struct LazyProcess{T,R}
    f::T
    cached_result::Union{Nothing,R}
    cached_process::Bool
    # Default constructor that enforces R = DataFrame
    LazyProcess(f::T) where {T} = new{T,DataFrame}(f, nothing, false)

    # Constructor with explicit R
    LazyProcess(f::T, ::Type{R}) where {T,R} = new{T,R}(f, nothing, false)
end

"""
    ContingencyTable <: AssociationDataFormat

A type representing a contingency table for analyzing the context of a target word within a specified window size in a given input string.

# Fields
- `con_tbl::LazyProcess{T, DataFrame}`: A lazy process that generates the contingency table when accessed.
- `node::AbstractString`: The target word for which the contingency table is generated.
- `windowsize::Int`: The window size around the target word to consider for context.
- `minfreq::Int64`: The minimum frequency threshold for including words in the contingency table.

# Constructors
- `ContingencyTable(inputstring::AbstractString, node::AbstractString, windowsize::Int, minfreq::Int64=5; auto_prep::Bool=true)`

    Creates a new `ContingencyTable` instance.

    - `inputstring::AbstractString`: The input text to be analyzed.
    - `node::AbstractString`: The target word for which the contingency table will be generated.
    - `windowsize::Int`: The window size around the target word to consider for context.
    - `minfreq::Int64`: The minimum frequency threshold for including words in the contingency table (default: 5).
    - `auto_prep::Bool`: If true, preprocesses the input string before analysis (default: true).

# Example
```julia-doc
inputstring = "This is a sample text where the target word appears multiple times. The target word is analyzed for context."
node = "target"
windowsize = 5
minfreq = 2

cont_table = ContingencyTable(inputstring, node, windowsize, minfreq)
```
"""
struct ContingencyTable{T} <: AssociationDataFormat
    con_tbl::LazyProcess{T,DataFrame}
    node::AbstractString
    windowsize::Int
    minfreq::Int64

    function ContingencyTable(inputstring::AbstractString, node::AbstractString, windowsize::Int, minfreq::Int64=5; auto_prep::Bool=true)
        # Prepare the input string
        prepared_string = auto_prep ? prepstring(inputstring) : inputstring

        # Define the function that will compute the contingency table lazily
        f = () -> conttbl(prepared_string, node, windowsize, minfreq)

        # Create the LazyProcess
        con_tbl = LazyProcess(f)

        # Initialize the ContingencyTable
        new{typeof(f)}(con_tbl, node, windowsize, minfreq)
    end
end

# function createContingencyTable(inputstring::AbstractString, node::AbstractString, windowsize::Int, minfreq::Int64=5; auto_prep::Bool=true, store_prep::Bool=false)
#     prepared_string = auto_prep ? prepstring(inputstring) : inputstring

#     contingency_table = ContingencyTable(inputstring, node, windowsize, minfreq; auto_prep=auto_prep)
#     if store_prep
#         contingency_table.prepared_string = prepared_string
#     end
#     return contingency_table
# end


#################################################################################
#
#  Help functions for navigating the CachedData type
#
#  extract_cached_coom(m::CachedData) = extract_cached_data(m.coo_matrix)
#  extract_cached_freq(m::CachedData) = extract_cached_data(m.unigram_frequencies)
#  extract_cached_paircounts(cached_data::CachedData) = extract_cached_data(cached_data.paircounts)
#  get_N(m::CachedData) = extract_cached_data(m.unigram_frequencies)
#  browse_coom_pairs(coo::CooMatrix, term1::AbstractString, term2::AbstractString)
# 
#################################################################################

"""
    extract_cached_data(z::LazyProcess{T}) 

Intermediate function for the `LazyProcess` type. If the `cached_process` is `false` then the function `f()` will be called and the result will be stored in `cached_result` and `cached_process` will be set to `true`. If `cached_process` is `true` then the `cached_result` will be returned (which is either a Cooccurrence Matrix or a Dictionary with words as keys and frequencies as their values).

# Example
```julia-repl
julia> doc = StringDocument("This is a text about an apple. There are many texts about apples.");
       z = CachedData(doc, 2, :default)
       extract_cached_data(z)
2.0
```
"""
function extract_cached_data(z::LazyProcess{T,R}) where {T,R}
    z.cached_process || (z.cached_result = z.f(); z.cached_process = true)
    z.cached_result::R
end
