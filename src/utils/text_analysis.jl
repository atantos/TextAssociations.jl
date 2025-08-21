# =====================================
# File: src/utils/text_analysis.jl
# Text analysis utilities for advanced metrics
# =====================================
"""
    find_prior_words(doc::StringDocument, word::String, window::Int) -> Set{String}
    
Find unique words that appear within `window` words before each occurrence
of `word` in the document.
"""
function find_prior_words(doc::StringDocument, word::String, window::Int)
    tokens = TextAnalysis.tokenize(language(doc), text(doc))
    prior_words = Set{String}()

    positions = findall(==(word), tokens)

    for pos in positions
        start_idx = max(1, pos - window)
        end_idx = pos - 1

        if end_idx >= start_idx
            for idx in start_idx:end_idx
                push!(prior_words, tokens[idx])
            end
        end
    end

    return prior_words
end

"""
    find_following_words(doc::StringDocument, word::String, window::Int) -> Set{String}
    
Find unique words that appear within `window` words after each occurrence
of `word` in the document.
"""
function find_following_words(doc::StringDocument, word::String, window::Int)
    tokens = TextAnalysis.tokenize(language(doc), text(doc))
    following_words = Set{String}()

    positions = findall(==(word), tokens)

    for pos in positions
        start_idx = pos + 1
        end_idx = min(length(tokens), pos + window)

        if end_idx >= start_idx
            for idx in start_idx:end_idx
                push!(following_words, tokens[idx])
            end
        end
    end

    return following_words
end

"""
    count_word_frequency(doc::StringDocument, word::String) -> Int
    
Count the frequency of a word in the document.
"""
function count_word_frequency(doc::StringDocument, word::String)
    tokens = TextAnalysis.tokenize(language(doc), text(doc))
    return count(==(word), tokens)
end

"""
    count_substrings(text::String, substring::String) -> Int

Count occurrences of a substring in text.
"""
function count_substrings(text::String, substring::String)::Int
    count = 0
    for _ in eachmatch(Regex(escape_string(substring)), text)
        count += 1
    end
    return count
end

"""
    count_substrings(text::String, substrings::Vector{String}) -> Dict{String,Int}

Count occurrences of multiple substrings in text.
"""
function count_substrings(text::String, substrings::Vector{String})::Dict{String,Int}
    counts = Dict{String,Int}()
    for substring in substrings
        count = 0
        for _ in eachmatch(Regex(escape_string(substring)), text)
            count += 1
        end
        counts[substring] = count
    end
    return counts
end