# =====================================
# File: src/metrics/lexical_gravity.jl
# Lexical Gravity and related metrics
# =====================================

"""
    eval_lexicalgravity(data::ContingencyTable)

Compute the Lexical Gravity measure for word associations.

Lexical Gravity captures the asymmetric nature of word associations by
considering both forward and backward context windows.

# Formula
G(w1, w2) = log(f(w1,w2) * n(w1) / f(w1)) + log(f(w1,w2) * n'(w2) / f(w2))

Where:
- f(w1,w2) = co-occurrence frequency
- n(w1) = number of unique words in w1's context
- f(w1) = total frequency of w1
- n'(w2) = number of unique words that precede w2
- f(w2) = total frequency of w2
"""
function eval_lexicalgravity(data::ContingencyTable)
    # Extract contingency table
    con_tbl = extract_cached_data(data.con_tbl)
    isempty(con_tbl) && return Float64[]

    # Extract the original document
    input_doc = extract_document(data.input_ref)

    # Get collocate words as strings
    collocates = string.(con_tbl.Collocate)

    # Initialize result array
    n_collocates = length(collocates)
    gravity_scores = zeros(Float64, n_collocates)

    # Compute for each collocate
    for (i, collocate) in enumerate(collocates)
        # f(w1,w2): co-occurrence frequency from contingency table
        f_w1_w2 = con_tbl.a[i]

        # Skip if no co-occurrence
        if f_w1_w2 == 0
            gravity_scores[i] = -Inf
            continue
        end

        # n(w1): number of unique collocates of the node word
        n_w1 = nrow(con_tbl)

        # f(w1): total frequency of the node word in contexts
        f_w1 = sum(con_tbl.a)

        # n'(w2): number of unique words that appear before collocate
        prior_words = find_prior_words(input_doc, collocate, data.windowsize)
        n_w2 = length(prior_words)

        # f(w2): total frequency of the collocate word
        f_w2 = count_word_frequency(input_doc, collocate)

        # Avoid division by zero
        if f_w1 == 0 || f_w2 == 0 || n_w2 == 0
            gravity_scores[i] = -Inf
        else
            # Compute lexical gravity
            term1 = log(f_w1_w2 * n_w1 / f_w1)
            term2 = log(f_w1_w2 * n_w2 / f_w2)
            gravity_scores[i] = term1 + term2
        end
    end

    return gravity_scores
end