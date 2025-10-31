# Common accessors all metrics rely on


# Internal document separator token. Used to prevent co-occurrences across document boundaries. 
if !isdefined(@__MODULE__, :_DOC_SEP)
    const _DOC_SEP = "\uFFFF"
end



"""
    assoc_node_present(x::AssociationDataFormat)::Union{Bool,Nothing}

Return `true` if the (normalized) node occurs ≥ 1 time in the underlying data,
`false` if it definitely does not, or `nothing` if this carrier cannot tell.
Default is `nothing`; concrete types should overload.

# Notes
For multi-word nodes (n-grams), checks if the complete n-gram sequence appears.
"""
assoc_node_present(x::AssociationDataFormat) = nothing
assoc_df(x::AssociationDataFormat) = error("assoc_df not implemented for $(typeof(x))")
assoc_node(x::AssociationDataFormat) = error("assoc_node not implemented")
assoc_ws(x::AssociationDataFormat) = error("assoc_ws not implemented")
assoc_tokens(x::AssociationDataFormat) = nothing  # default: no tokens
assoc_norm_config(x::AssociationDataFormat) = error("assoc_norm_config not implemented for $(typeof(x))")

# Concrete impls

# ContingencyTable

# ContingencyTable: check tokens (already normalized consistently via assoc_tokens)
"""
    assoc_node_present(x::ContingencyTable) -> Union{Bool,Nothing}

Check if the node (single word or n-gram) is present in the document.

For single words, uses simple token matching.
For n-grams, searches for the complete sequence in the token stream.
"""
assoc_node_present(x::ContingencyTable) = begin
    toks = assoc_tokens(x)
    toks === nothing && return nothing

    node = assoc_node(x)
    n = ngram_length(node)

    if n == 1
        # Single word - simple search
        return any(==(node), toks)
    else
        # N-gram - search for sequence
        return !isempty(find_ngram_positions(toks, node))
    end
end

assoc_df(x::ContingencyTable) = cached_data(x.con_tbl)
assoc_node(x::ContingencyTable) = x.node
assoc_ws(x::ContingencyTable) = x.windowsize
assoc_tokens(x::ContingencyTable) = String.(tokens(document(x.input_ref)))
assoc_norm_config(x::ContingencyTable) = x.norm_config

# CorpusContingencyTable
# CorpusContingencyTable: O(1) via corpus vocabulary (same TextNorm)
# NOTE: field is `corpus_ref` in the struct, not `corpus`
"""
    assoc_node_present(x::CorpusContingencyTable) -> Bool

Check if the node (single word or n-gram) is present in the corpus.

For single words, uses O(1) vocabulary lookup.
For n-grams, checks if all constituent words are in vocabulary (conservative check).
"""
function assoc_node_present(x::CorpusContingencyTable)
    node = assoc_node(x)
    n = ngram_length(node)

    if n == 1
        # Single word - definitive check
        return get(x.corpus_ref.vocabulary, node, 0) > 0
    else
        # N-gram - check if it's even possible
        ngram_words = split_ngram(node)
        if any(word -> get(x.corpus_ref.vocabulary, word, 0) == 0, ngram_words)
            return false  # Impossible: at least one word doesn't exist
        else
            return nothing  # Uncertain: all words exist but n-gram might not
        end
    end
end
assoc_df(x::CorpusContingencyTable) = cached_data(x.aggregated_table)
assoc_node(x::CorpusContingencyTable) = x.node
assoc_ws(x::CorpusContingencyTable) = x.windowsize
assoc_norm_config(x::CorpusContingencyTable) = x.norm_config

"""
    assoc_tokens(x::CorpusContingencyTable) -> Vector{String}

Extract all tokens from the corpus with document separators.

For n-gram analysis, tokens from each document are separated by a special
token (_DOC_SEP) to prevent cross-document n-gram matches.
"""
function assoc_tokens(x::CorpusContingencyTable)
    toks = String[]
    for (i, ct) in enumerate(x.tables)
        append!(toks, String.(tokens(document(ct.input_ref))))
        # Add separator between documents to prevent cross-document n-grams
        if i < length(x.tables)
            push!(toks, _DOC_SEP)
        end
    end
    return toks
end