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

"""
    check_stopwords_support(languages::Vector{Language}) -> DataFrame

Check which languages have stopwords available in Languages.jl.

# Example
```julia
using Languages

# Test common languages
test_langs = [
    Languages.English(),
    Languages.German(),
    Languages.Greek(),
    Languages.Spanish(),
    Languages.French(),
    Languages.Italian(),
    Languages.Portuguese(),
    Languages.Dutch(),
    Languages.Russian(),
    Languages.Chinese(),
    Languages.Japanese()
]

support_df = check_stopwords_support(test_langs)
```
"""
function check_stopwords_support(languages::Vector{<:Language})
    results = []

    for lang in languages
        lang_name = string(typeof(lang))
        lang_name = replace(lang_name, r"^Languages\." => "")

        try
            sw = Languages.stopwords(lang)
            push!(results, (
                Language=lang_name,
                Supported=true,
                NumStopwords=length(sw),
                Sample=join(first(sw, 3), ", ")
            ))
        catch e
            push!(results, (
                Language=lang_name,
                Supported=false,
                NumStopwords=0,
                Sample="N/A"
            ))
        end
    end

    return DataFrame(results)
end