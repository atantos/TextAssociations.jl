# =====================================
# File: src/metrics/lexical_gravity.jl
# Lexical Gravity - Daudaravičius & Marcinkevičienė (2004)
# =====================================
"""
Directional analysis helper: get bounded range that does not cross document boundaries.
"""
@inline function _bounded_range(tokens::Vector{String}, start::Int, stop::Int)
    stop′ = stop
    @inbounds for j in start:stop
        if tokens[j] === _DOC_SEP
            stop′ = j - 1
            break
        end
    end
    return start <= stop′ ? (start, stop′) : (1, 0)
end

# Mark LexicalGravity as needing tokens
NeedsTokens(::Type{LexicalGravity}) = Val(true)

"""
    eval_lexicalgravity(data::AssociationDataFormat; 
                       tokens::Vector{String},
                       formula::Symbol=:original)

Compute Lexical Gravity measure based on Daudaravičius & Marcinkevičienė (2004).

# Arguments
- `data`: AssociationDataFormat with co-occurrence data
- `tokens`: Required tokenized text (provided by assoc_score when called through API)
- `formula`: Which formula to use:
  - `:original` - The main formula from the paper: G→(w1,w2) = log(f→×n+/f1) + log(f←×n-/f2)
  - `:simplified` - Simplified version: G = log₂((f²×span)/(f1×f2))
  - `:pmi_weighted` - PMI-style weighting: G = f(w1,w2) × log((f×N)/(f1×f2))

# Original Formula from Paper
G→(w1,w2) = log(f→(w1,w2)/f(w1) × n+(w1)) + log(f←(w1,w2)/f(w2) × n-(w2))

Where:
- f→(w1,w2) = frequency of w2 following w1 within window
- f←(w1,w2) = frequency of w1 preceding w2 within window  
- n+(w1) = number of different word types that follow w1
- n-(w2) = number of different word types that precede w2
- f(w1), f(w2) = total frequencies of words

# Note
This function expects tokens to be provided. When called through assoc_score(),
tokens are automatically fetched based on the NeedsTokens trait.

# N-gram Notes
For multi-word nodes:
- f(node) counts complete n-gram occurrences
- n+(node) counts types following the last word of the n-gram
- f→ counts when collocate appears after the n-gram
- f← counts when collocate appears before the n-gram
"""
function eval_lexicalgravity(data::AssociationDataFormat;
    tokens::Vector{String},  # Required parameter
    formula::Symbol=:original,
    kwargs...)

    con_tbl = assoc_df(data)  # Use the generic accessor
    isempty(con_tbl) && return Float64[]

    if formula == :original
        return _gravity_original_formula(data, con_tbl, tokens)
    elseif formula == :simplified
        return _gravity_simplified_formula(data, con_tbl, tokens)
    elseif formula == :pmi_weighted
        return _gravity_pmi_weighted(data, con_tbl, tokens)
    else
        throw(ArgumentError("Unknown formula: $formula. Use :original, :simplified, or :pmi_weighted"))
    end
end

"""
The original Daudaravičius & Marcinkevičienė formula.
This is the main contribution of their paper.
"""
function _gravity_original_formula(
    x::AssociationDataFormat,
    con_tbl::DataFrame,
    tokens::Vector{String}
)::Vector{Float64}
    # Inputs from the association context
    node = assoc_node(x)
    ws = assoc_ws(x)
    n = ngram_length(node)

    nrows = nrow(con_tbl)
    scores = Vector{Float64}(undef, nrows)
    nrows == 0 && return scores
    ws <= 0 && return fill(-Inf, nrows)

    # Find node positions (handles both single word and n-gram)
    node_positions = if n == 1
        findall(==(node), tokens)
    else
        find_ngram_positions(tokens, node)
    end

    f_node = length(node_positions)
    f_node == 0 && return fill(-Inf, nrows)

    # n⁺(node): distinct types in the RIGHT window of each node occurrence
    # For n-grams, this is after the last word of the n-gram
    words_after_node = Set{String}()
    @inbounds for npos in node_positions
        # pre-clamp, then bound within document
        # For n-grams, the right window starts after the last token
        node_end = npos + n - 1
        r_start = npos + 1
        r_stop = min(node_end + ws, length(tokens))
        (r1, r2) = _bounded_range(tokens, max(r_start, 1), r_stop)
        r2 < r1 && continue
        for j in r1:r2
            t = tokens[j]
            t == _DOC_SEP && break
            push!(words_after_node, t)
        end
    end
    n_plus_node = length(words_after_node)

    # Main loop over collocates in the contingency table
    @inbounds for (i, row) in enumerate(eachrow(con_tbl))
        coll = String(row.Collocate)

        # Positions & frequency of collocate
        coll_positions = findall(==(coll), tokens)
        f_coll = length(coll_positions)

        # n⁻(coll): distinct types in the LEFT window of each collocate occurrence
        words_before_coll = Set{String}()
        for cpos in coll_positions
            l_start = max(cpos - ws, 1)
            l_stop = cpos - 1
            (l1, l2) = _bounded_range(tokens, l_start, l_stop)
            l2 < l1 && continue
            @inbounds for j in reverse(l1:l2)
                t = tokens[j]
                t == _DOC_SEP && break
                push!(words_before_coll, t)
            end
        end
        n_minus_coll = length(words_before_coll)

        # Directional co-occurrence counts
        f_forward = 0  # node → coll (coll to the right of node)
        f_backward = 0  # node ← coll (node to the left of coll)

        # Count node → coll
        # For n-grams, check if coll appears after the n-gram
        for npos in node_positions
            node_end = npos + n - 1
            r_start = node_end + 1
            r_stop = min(node_end + ws, length(tokens))
            (r1, r2) = _bounded_range(tokens, max(r_start, 1), r_stop)
            r2 < r1 && continue
            @inbounds for j in r1:r2
                t = tokens[j]
                t == _DOC_SEP && break
                if t == coll
                    f_forward += 1
                end
            end
        end

        # Count node ← coll
        # For n-grams, check if coll appears before the n-gram
        for cpos in coll_positions
            # Find if any node occurrence has coll in its left window
            for npos in node_positions
                l_start = max(npos - ws, 1)
                l_stop = npos - 1
                (l1, l2) = _bounded_range(tokens, l_start, l_stop)
                l2 < l1 && continue
                if cpos in l1:l2
                    f_backward += 1
                    break  # Count each coll position only once per node
                end
            end
        end

        # G(w1,w2) = log(f→ * n⁺ / f_node) + log(f← * n⁻ / f_coll)
        if f_forward > 0 && f_backward > 0 && f_node > 0 && f_coll > 0 &&
           n_plus_node > 0 && n_minus_coll > 0
            term1 = log((f_forward * n_plus_node) / f_node)
            term2 = log((f_backward * n_minus_coll) / f_coll)
            scores[i] = term1 + term2
        else
            scores[i] = -Inf
        end
    end

    return scores
end


"""
Simplified gravity formula often used in implementations.
G = log₂((f(w1,w2)² × span) / (f(w1) × f(w2)))
"""
function _gravity_simplified_formula(data::AssociationDataFormat, con_tbl::DataFrame, tokens::Vector{String})
    node = assoc_node(data)
    windowsize = assoc_ws(data)
    n = ngram_length(node)

    # Count node occurrences (handles n-grams)
    f_node = if n == 1
        count(==(node), tokens)
    else
        length(find_ngram_positions(tokens, node))
    end

    span_size = windowsize * 2

    gravity_scores = zeros(Float64, nrow(con_tbl))

    for (i, row) in enumerate(eachrow(con_tbl))
        collocate = string(row.Collocate)
        f_cooc = row.a  # co-occurrence frequency from contingency table
        f_collocate = count(==(collocate), tokens)

        if f_cooc > 0 && f_node > 0 && f_collocate > 0
            # Simplified: emphasizes co-occurrence frequency squared
            gravity_scores[i] = 2 * log2_safe(f_cooc) + log2_safe(span_size) -
                                log2_safe(f_node) - log2_safe(f_collocate)
        else
            gravity_scores[i] = -Inf
        end
    end

    return gravity_scores
end

"""
PMI-weighted gravity (alternative formulation).
G = f(w1,w2) × PMI(w1,w2)
"""
function _gravity_pmi_weighted(data::AssociationDataFormat, con_tbl::DataFrame, tokens::Vector{String})
    node = assoc_node(data)
    n = ngram_length(node)

    N = length(tokens)
    # Count node occurrences (handles n-grams)
    f_node = if n == 1
        count(==(node), tokens)
    else
        length(find_ngram_positions(tokens, node))
    end

    gravity_scores = zeros(Float64, nrow(con_tbl))

    for (i, row) in enumerate(eachrow(con_tbl))
        collocate = string(row.Collocate)
        f_cooc = row.a
        f_collocate = count(==(collocate), tokens)

        if f_cooc > 0 && f_node > 0 && f_collocate > 0
            # PMI component
            pmi = log_safe((f_cooc * N) / (f_node * f_collocate))
            # Weight by frequency
            gravity_scores[i] = f_cooc * pmi
        else
            gravity_scores[i] = -Inf
        end
    end

    return gravity_scores
end

"""
    lexical_gravity_analysis(data::AssociationDataFormat; 
                            tokens::Union{Nothing,Vector{String}}=nothing)

Comprehensive analysis using all gravity formulas for comparison.
Returns results from all three formulas plus directional analysis.

Note: If tokens are not provided, will attempt to fetch them using assoc_tokens.
"""
function lexical_gravity_analysis(data::AssociationDataFormat;
    tokens::Union{Nothing,Vector{String}}=nothing)
    results = Dict{Symbol,Any}()

    # Get tokens if not provided
    if tokens === nothing
        tokens = assoc_tokens(data)
        if tokens === nothing
            throw(ArgumentError("LexicalGravity analysis requires tokens. Pass tokens= or implement assoc_tokens for $(typeof(data))"))
        end
    end

    # Calculate all three formulas (calling through assoc_score to ensure proper token handling)
    results[:gravity_original] = assoc_score(LexicalGravity, data, formula=:original, tokens=tokens, scores_only=true)
    results[:gravity_simplified] = assoc_score(LexicalGravity, data, formula=:simplified, tokens=tokens, scores_only=true)
    results[:gravity_pmi_weighted] = assoc_score(LexicalGravity, data, formula=:pmi_weighted, tokens=tokens, scores_only=true)

    # Add directional analysis
    results[:directional] = _gravity_directional_analysis(data, tokens)

    return (; results...)
end

"""
Analyze directional preferences (left vs right) for collocations.
"""
function _gravity_directional_analysis(data::AssociationDataFormat, tokens::Vector{String})
    con_tbl = assoc_df(data)
    isempty(con_tbl) && return (left=Float64[], right=Float64[], asymmetry=Float64[])

    node = assoc_node(data)
    windowsize = assoc_ws(data)
    n = ngram_length(node)

    # Find node positions (handles n-grams)
    node_positions = if n == 1
        findall(==(node), tokens)
    else
        find_ngram_positions(tokens, node)
    end

    nrows = nrow(con_tbl)
    left_scores = zeros(Float64, nrows)
    right_scores = zeros(Float64, nrows)
    asymmetry = zeros(Float64, nrows)

    for (i, row) in enumerate(eachrow(con_tbl))
        collocate = string(row.Collocate)

        left_count = 0
        right_count = 0

        for pos in node_positions
            node_end = pos + n - 1

            # Count left occurrences
            left_window = max(1, pos - windowsize):(pos-1)
            left_count += count(==(collocate), tokens[left_window])

            # Count right occurrences in the right window, after the n-gram
            right_window = (node_end+1):min(length(tokens), node_end + windowsize)
            right_count += count(==(collocate), tokens[right_window])
        end

        total = left_count + right_count
        if total > 0
            left_scores[i] = left_count / total
            right_scores[i] = right_count / total
            # Asymmetry: positive = right preference, negative = left preference
            asymmetry[i] = (right_count - left_count) / total
        else
            left_scores[i] = right_scores[i] = asymmetry[i] = 0.0
        end
    end

    return (left=left_scores, right=right_scores, asymmetry=asymmetry)
end

# Convenience alias
const lexicalgravity = eval_lexicalgravity


# ΔP→(X→Y) = P(Y|X) − P(Y|¬X) = a/m − c/n
"""
Directional analysis: rightward influence from X to Y following Gries(2013).
Works with both single words and n-grams.
"""
function eval_deltapiright(data::AssociationDataFormat)
    @extract_values data a c m n
    (a ./ max.(m, eps())) .- (c ./ max.(n, eps()))
end

# ΔP←(Y→X) = P(X|Y) − P(X|¬Y) = a/k − b/l
"""
Directional analysis: leftward influence from Y to X following Gries(2013).
Works with both single words and n-grams.
"""
function eval_deltapileft(data::AssociationDataFormat)
    @extract_values data a b k l
    (a ./ max.(k, eps())) .- (b ./ max.(l, eps()))
end