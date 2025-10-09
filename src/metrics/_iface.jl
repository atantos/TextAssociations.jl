# Common accessors all metrics rely on

"""
    assoc_node_present(x::AssociationDataFormat)::Union{Bool,Nothing}

Return `true` if the (normalized) node occurs ≥ 1 time in the underlying data,
`false` if it definitely does not, or `nothing` if this carrier cannot tell.
Default is `nothing`; concrete types should overload.
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
assoc_node_present(x::ContingencyTable) = begin
    toks = assoc_tokens(x)
    toks === nothing ? nothing : any(==(assoc_node(x)), toks)
end
assoc_df(x::ContingencyTable) = cached_data(x.con_tbl)
assoc_node(x::ContingencyTable) = x.node
assoc_ws(x::ContingencyTable) = x.windowsize
assoc_tokens(x::ContingencyTable) = String.(tokens(document(x.input_ref)))
assoc_norm_config(x::ContingencyTable) = x.norm_config

# CorpusContingencyTable
# CorpusContingencyTable: O(1) via corpus vocabulary (same TextNorm)
# NOTE: field is `corpus_ref` in the struct, not `corpus`
assoc_node_present(x::CorpusContingencyTable) =
    get(x.corpus_ref.vocabulary, assoc_node(x), 0) > 0
assoc_df(x::CorpusContingencyTable) = cached_data(x.aggregated_table)
assoc_node(x::CorpusContingencyTable) = x.node
assoc_ws(x::CorpusContingencyTable) = x.windowsize
assoc_norm_config(x::CorpusContingencyTable) = x.norm_config
# Extract all tokens from the corpus
function assoc_tokens(x::CorpusContingencyTable)
    toks = String[]
    for ct in x.tables
        append!(toks, String.(tokens(document(ct.input_ref))))
    end
    return toks
end

