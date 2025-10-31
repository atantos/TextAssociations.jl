# =====================================
# File: src/core/contingency_table.jl
# ContingencyTable implementation for word co-occurrence analysis
# =====================================

using FreqTables: freqtable

"""
    ContingencyTable <: AssociationDataFormat

Represents a contingency table for word co-occurrence analysis.

# Fields
- `con_tbl`: Lazy-loaded contingency table data
- `node`: Target word (normalized)
- `windowsize`: Context window size
- `minfreq`: Minimum frequency threshold
- `input_ref`: Reference to the processed input document
- `norm_config`: Text normalization configuration

# Notes
For multi-word nodes (n-grams), the window extends from the boundaries of the n-gram.
For example, with node "machine learning" and windowsize=5:
- Left context: 5 tokens before "machine"
- Right context: 5 tokens after "learning"
"""
struct ContingencyTable{T} <: AssociationDataFormat
    con_tbl::LazyProcess{T,DataFrame}
    node::AbstractString
    windowsize::Int
    minfreq::Int
    input_ref::LazyInput
    norm_config::TextNorm

    # Main constructor from raw text / Main constructor from raw text (PUBLIC - uses keyword args)
    function ContingencyTable(inputstring::AbstractString,
        node::AbstractString;
        windowsize::Int,
        minfreq::Int=5,
        norm_config::TextNorm=TextNorm())

        windowsize > 0 || throw(ArgumentError("Window size must be positive"))
        minfreq > 0 || throw(ArgumentError("Minimum frequency must be positive"))
        !isempty(node) || throw(ArgumentError("Node word cannot be empty"))

        # Normalize node using config (handles both single words and n-grams)
        normalized_node = normalize_node(node, norm_config)
        isempty(normalized_node) && throw(ArgumentError("Node becomes empty after normalization"))

        # Preprocess text using same config
        prepared_string = prep_string(inputstring, norm_config)
        input_ref = LazyInput(prepared_string)

        f = () -> cont_table(prepared_string, normalized_node; windowsize, minfreq)
        con_tbl = LazyProcess(f)

        return new{typeof(f)}(con_tbl, normalized_node, windowsize, minfreq,
            input_ref, norm_config)
    end

    # Constructor from LazyProcess (for corpus aggregation) / Constructor from LazyProcess (INTERNAL - all positional)
    function ContingencyTable(con_tbl::LazyProcess{T,DataFrame},
        node::AbstractString,
        windowsize::Int,
        minfreq::Int,
        input_ref::LazyInput,
        norm_config::TextNorm) where {T}

        windowsize > 0 || throw(ArgumentError("Window size must be positive"))
        minfreq > 0 || throw(ArgumentError("Minimum frequency must be positive"))
        !isempty(node) || throw(ArgumentError("Node word cannot be empty"))

        # Node should already be normalized when passed here
        return new{T}(con_tbl, node, windowsize, minfreq, input_ref, norm_config)
    end
end

# Convenience constructor from DataFrame
function ContingencyTable(df::DataFrame,
    node::AbstractString;
    windowsize::Int,
    minfreq::Int,
    norm_config::TextNorm=TextNorm(),
    input_ref::LazyInput=LazyInput(StringDocument("")))

    ContingencyTable(LazyProcess(() -> df, DataFrame), node, windowsize,
        minfreq, input_ref, norm_config)
end

"""
    cont_table(input_doc::StringDocument, target_word::AbstractString;
              windowsize::Int=5, minfreq::Int=3) -> DataFrame

Compute the contingency table for a target word or n-gram in a document.

# Arguments
- `input_doc`: Preprocessed document (tokens already normalized)
- `target_word`: Normalized node (can be single word or n-gram like "machine learning")
- `windowsize`: Number of tokens on each side of the node
- `minfreq`: Minimum co-occurrence frequency threshold

# Returns
DataFrame with columns: Collocate, a, b, c, d, m, n, k, l, N, E₁₁, E₁₂, E₂₁, E₂₂

# Notes
- For n-grams, target_word should be space-separated (e.g., "machine learning")
- The function automatically detects single vs. multi-word nodes
- Windows are calculated from n-gram boundaries for multi-word nodes
- Empty DataFrame is returned if node not found or no collocates meet minfreq

# Examples
```julia
doc = StringDocument("machine learning is great and machine learning works")
ct = cont_table(doc, "machine learning"; windowsize=2, minfreq=1)
Note: target_word should already be normalized before calling this function.
"""
function cont_table(input_doc::StringDocument, target_word::AbstractString;
    windowsize::Int=5, minfreq::Int=3)

    input_tokens = TextAnalysis.tokenize(language(input_doc), text(input_doc))

    # Determine if node is single word or n-gram
    n = ngram_length(target_word)

    # Find all occurrences of the node
    indices = if n == 1
        # Single word - use existing logic
        findall(==(target_word), input_tokens)
    else
        # Multi-word - find n-gram occurrences
        find_ngram_positions(input_tokens, target_word)
    end

    isempty(indices) && return DataFrame()

    # Extract context words with n-gram-aware windowing
    context_mask, contexts = if n == 1
        # Single word - original logic
        mask = falses(length(input_tokens))
        contexts = Tuple{UnitRange{Int64},UnitRange{Int64}}[]

        for index in indices
            left_start = max(1, index - windowsize)
            right_end = min(length(input_tokens), index + windowsize)

            mask[left_start:index-1] .= true
            mask[index+1:right_end] .= true
            push!(contexts, (left_start:index-1, index+1:right_end))
        end

        mask, contexts
    else
        # Multi-word - use new n-gram context extraction
        extract_ngram_contexts(input_tokens, indices, n, windowsize)
    end

    # Count unique context words (same logic for both single and multi-word)
    unique_counts = Dict{String,Int}()
    for ctx in contexts
        seen_words = Set{String}()
        # Combine left and right ranges
        words = input_tokens[union(ctx...)]
        for word in words
            if !in(word, seen_words)
                push!(seen_words, word)
                unique_counts[word] = get(unique_counts, word, 0) + 1
            end
        end
    end

    # Build frequency tables
    context_indices = findall(context_mask)
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
    # For n-grams, we need to mark all tokens that are part of any occurrence
    if n == 1
        context_mask[indices] .= true
    else
        # Mark all tokens that are part of n-gram occurrences
        for idx in indices
            context_mask[idx:idx+n-1] .= true
        end
    end
    other_words = input_tokens[.!context_mask]
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
    insertcols!(con_df, 1, :Collocate => String.(names(a)[1]))

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
    n = ngram_length(con_tbl.node)
    node_type = n == 1 ? "unigram" : "$(n)-gram"

    println(io, "ContingencyTable instance with:")
    println(io, "* Node ($(node_type)): \"$(con_tbl.node)\"")
    println(io, "* Window size: $(con_tbl.windowsize) tokens")
    println(io, "* Minimum collocation frequency: $(con_tbl.minfreq)")
    println(io, "* Normalization config: $(con_tbl.norm_config)")
end