# =====================================
# File: src/metrics/lexical_gravity.jl
# Lexical Gravity - Daudaravičius & Marcinkevičienė (2004)
# =====================================

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
function _gravity_original_formula(data::AssociationDataFormat, con_tbl::DataFrame, tokens::Vector{String})
    # Get node and window size using generic accessors
    node = assoc_node(data)
    windowsize = assoc_ws(data)

    # Find all positions of the node word
    node_positions = findall(==(node), tokens)
    f_node = length(node_positions)

    # Calculate n+(node): unique word types that appear AFTER the node
    words_after_node = Set{String}()
    for pos in node_positions
        for i in (pos+1):min(pos + windowsize, length(tokens))
            push!(words_after_node, tokens[i])
        end
    end
    n_plus_node = length(words_after_node)

    # Initialize results
    gravity_scores = zeros(Float64, nrow(con_tbl))

    for (i, row) in enumerate(eachrow(con_tbl))
        collocate = string(row.Collocate)

        # Find positions of collocate
        collocate_positions = findall(==(collocate), tokens)
        f_collocate = length(collocate_positions)

        # Calculate n-(collocate): unique word types that appear BEFORE the collocate
        words_before_collocate = Set{String}()
        for pos in collocate_positions
            for i in max(pos - windowsize, 1):(pos-1)
                push!(words_before_collocate, tokens[i])
            end
        end
        n_minus_collocate = length(words_before_collocate)

        # Count directional co-occurrences
        f_forward = 0   # node→collocate (collocate follows node)
        f_backward = 0  # node←collocate (node precedes collocate)

        # Count node→collocate
        for node_pos in node_positions
            right_start = node_pos + 1
            right_end = min(node_pos + windowsize, length(tokens))
            if right_end >= right_start
                for pos in right_start:right_end
                    if tokens[pos] == collocate
                        f_forward += 1
                    end
                end
            end
        end

        # Count node←collocate (node appears before collocate)
        for coll_pos in collocate_positions
            left_start = max(coll_pos - windowsize, 1)
            left_end = coll_pos - 1
            if left_end >= left_start
                for pos in left_start:left_end
                    if tokens[pos] == node
                        f_backward += 1
                    end
                end
            end
        end

        # Calculate gravity using the original formula
        if f_forward > 0 && f_backward > 0 && f_node > 0 && f_collocate > 0
            # G→(w1,w2) = log(f→×n+/f1) + log(f←×n-/f2)
            term1 = log_safe(f_forward * n_plus_node / f_node)
            term2 = log_safe(f_backward * n_minus_collocate / f_collocate)
            gravity_scores[i] = term1 + term2
        else
            gravity_scores[i] = -Inf
        end
    end

    return gravity_scores
end

"""
Simplified gravity formula often used in implementations.
G = log₂((f(w1,w2)² × span) / (f(w1) × f(w2)))
"""
function _gravity_simplified_formula(data::AssociationDataFormat, con_tbl::DataFrame, tokens::Vector{String})
    node = assoc_node(data)
    windowsize = assoc_ws(data)

    f_node = count(==(node), tokens)
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

    N = length(tokens)
    f_node = count(==(node), tokens)

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

    node_positions = findall(==(node), tokens)

    n = nrow(con_tbl)
    left_scores = zeros(Float64, n)
    right_scores = zeros(Float64, n)
    asymmetry = zeros(Float64, n)

    for (i, row) in enumerate(eachrow(con_tbl))
        collocate = string(row.Collocate)

        left_count = 0
        right_count = 0

        for pos in node_positions
            # Count left occurrences
            left_window = max(1, pos - windowsize):(pos-1)
            left_count += count(==(collocate), tokens[left_window])

            # Count right occurrences  
            right_window = (pos+1):min(length(tokens), pos + windowsize)
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