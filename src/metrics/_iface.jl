# Common accessors all metrics rely on

assoc_df(x::AssociationDataFormat) = error("assoc_df not implemented for $(typeof(x))")
assoc_node(x::AssociationDataFormat) = error("assoc_node not implemented")
assoc_ws(x::AssociationDataFormat) = error("assoc_ws not implemented")
assoc_tokens(x::AssociationDataFormat) = nothing  # default: no tokens

# Concrete impls

# ContingencyTable
assoc_df(x::ContingencyTable) = cached_data(x.con_tbl)
assoc_node(x::ContingencyTable) = x.node
assoc_ws(x::ContingencyTable) = x.windowsize
assoc_tokens(x::ContingencyTable) = String.(tokens(document(x.input_ref)))

# CorpusContingencyTablex
assoc_df(x::CorpusContingencyTable) = cached_data(x.aggregated_table)
assoc_node(x::CorpusContingencyTable) = x.node
assoc_ws(x::CorpusContingencyTable) = x.windowsize
# Extract all tokens from the corpus
function assoc_tokens(x::CorpusContingencyTable)
    all_tokens = String[]
    for doc in x.corpus_ref.documents
        append!(all_tokens, tokens(doc))
    end
    return all_tokens
end
