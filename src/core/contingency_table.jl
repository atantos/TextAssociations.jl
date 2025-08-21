# =====================================
# File: src/core/contingency_table.jl
# ContingencyTable implementation
# =====================================
"""
    ContingencyTable <: AssociationDataFormat

Represents a contingency table for word co-occurrence analysis.

# Fields
- `con_tbl`: Lazy-loaded contingency table data
- `node`: Target word
- `windowsize`: Context window size
- `minfreq`: Minimum frequency threshold
- `input_ref`: Reference to the processed input document
"""
struct ContingencyTable{T} <: AssociationDataFormat
    con_tbl::LazyProcess{T,DataFrame}
    node::AbstractString
    windowsize::Int
    minfreq::Int64
    input_ref::LazyInput

    function ContingencyTable(inputstring::AbstractString,
        node::AbstractString,
        windowsize::Int,
        minfreq::Int64=5;
        auto_prep::Bool=true)

        # Validate inputs
        windowsize > 0 || throw(ArgumentError("Window size must be positive"))
        minfreq > 0 || throw(ArgumentError("Minimum frequency must be positive"))
        !isempty(node) || throw(ArgumentError("Node word cannot be empty"))

        # Prepare the input string
        prepared_string = auto_prep ? prepstring(inputstring) : StringDocument(inputstring)

        # Create LazyInput to store reference to processed document
        input_ref = LazyInput(prepared_string)

        # Define lazy computation for contingency table
        f = () -> conttbl(prepared_string, node, windowsize, minfreq)
        con_tbl = LazyProcess(f)

        new{typeof(f)}(con_tbl, node, windowsize, minfreq, input_ref)
    end
end

"""
    conttbl(input_doc::StringDocument, target_word::AbstractString, 
            windowsize::Int64=5, minfreq::Int64=3) -> DataFrame

Compute the contingency table for a target word in a document.
"""
function conttbl(input_doc::StringDocument, target_word::AbstractString,
    windowsize::Int64=5, minfreq::Int64=3)

    input_tokens = TextAnalysis.tokenize(language(input_doc), text(input_doc))

    # Find target word indices
    indices = findall(==(target_word), input_tokens)
    isempty(indices) && return DataFrame()

    # Collect context words
    context_indices = falses(length(input_tokens))
    contexts = Tuple{UnitRange{Int64},UnitRange{Int64}}[]

    for index in indices
        left_start = max(1, index - windowsize)
        right_end = min(length(input_tokens), index + windowsize)

        context_indices[left_start:index-1] .= true
        context_indices[index+1:right_end] .= true
        push!(contexts, (left_start:index-1, index+1:right_end))
    end

    # Count unique context words
    unique_counts = Dict{String,Int}()
    for ctx in contexts
        seen_words = Set{String}()
        words = input_tokens[union(ctx...)]
        for word in words
            if !in(word, seen_words)
                push!(seen_words, word)
                unique_counts[word] = get(unique_counts, word, 0) + 1
            end
        end
    end

    # Build frequency tables
    node_context_words = input_tokens[context_indices]
    a = freqtable(node_context_words)
    filter!(x -> x >= minfreq, a)

    # Early return if no words meet minimum frequency
    isempty(a) && return DataFrame()

    # Compute remaining cells
    valid_words = Set(names(a)[1])

    # Compute b values
    unique_counts_intersect = intersect(Set(keys(unique_counts)), valid_words)
    b = Dict(key => length(indices) - unique_counts[key]
             for key in unique_counts_intersect)
    b = freqtable(collect(keys(b)), weights=collect(values(b)))

    # Compute c and d
    context_indices[indices] .= true
    other_words = input_tokens[.!context_indices]
    c_words = filter(x -> x in valid_words, other_words)
    c = freqtable(c_words)

    # Ensure all words are in c
    idx = 0
    for name in valid_words
        get!(c.dicts[1], name) do
            idx += 1
            return length(c) + idx
        end
    end

    # Pad array if needed
    if idx > 0
        append!(c.array, zeros(Int64, idx))
    end

    reference_length = length(filter(x -> !in(x, valid_words), other_words))
    d = reference_length .- c

    # Build DataFrame
    con_table = hcat(a, b, c, d)
    con_df = DataFrame(con_table.array, Symbol.(["a", "b", "c", "d"]))
    insertcols!(con_df, 1, :Collocate => Symbol.(names(a)[1]))

    # Add derived columns
    @chain con_df begin
        transform!([:a, :b] => ((a, b) -> a .+ b) => :m)
        transform!([:c, :d] => ((c, d) -> c .+ d) => :n)
        transform!([:a, :c] => ((a, c) -> a .+ c) => :k)
        transform!([:b, :d] => ((b, d) -> b .+ d) => :l)
        transform!([:m, :n] => ((m, n) -> m .+ n) => :N)
        transform!([:m, :k, :N] => ((m, k, N) -> (m .* k) ./ N) => :E₁₁)
        transform!([:m, :l, :N] => ((m, l, N) -> (m .* l) ./ N) => :E₁₂)
        transform!([:n, :k, :N] => ((n, k, N) -> (n .* k) ./ N) => :E₂₁)
        transform!([:n, :l, :N] => ((n, l, N) -> (n .* l) ./ N) => :E₂₂)
    end

    return con_df
end

function Base.show(io::IO, con_tbl::ContingencyTable)
    println(io, "ContingencyTable instance with:")
    println(io, "* Node word: $(con_tbl.node)")
    println(io, "* Window size: $(con_tbl.windowsize)")
    println(io, "* Minimum collocation frequency: $(con_tbl.minfreq)")
end