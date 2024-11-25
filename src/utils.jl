# helper functions for creating a vocabulary from a StringDocument or a Vector{String}
# Converts a StringDocument to Vector{String}
tostringvector(doc::StringDocument) = tokens(doc)
# Identity function for Vector{String}
tostringvector(vec::Vector{String}) = vec

function createvocab(input::Union{StringDocument,Vector{String}})
    string_vector = tostringvector(input) |> unique
    # string_vector = length(string_vector) != length(unique(string_vector)) ? unique(string_vector) : string_vector

    # preallocating the ordered dictionary with the size of the string_vector
    ordered_dict = OrderedDict{String,Int}()
    sizehint!(ordered_dict, length(string_vector))

    # reverse the order of the keys and values in the enumerate iterator to get an ordered dict.
    for (index, key) in enumerate(string_vector)
        ordered_dict[key] = index
    end
    return ordered_dict
end


"""
    prepstring(input_path::AbstractString)

Prepare and preprocess a text document, a corpus of text documents or a plain raw string.

# Arguments
- `input_path::AbstractString`: The path to a text file, a directory containing text files, or a raw text string. 
    - If `input_path` is a path to a text file, the function reads and preprocesses the content of the file.
    - If `input_path` is a path to a directory, the function reads and preprocesses the content of all `.txt` files in the directory.
    - If `input_path` is a raw text string, the function preprocesses the string directly.

# Returns
- A `StringDocument` object with normalized text data.

# Preprocessing Steps
The function performs the following preprocessing steps on the text:
- Strips punctuation
- Strips whitespace
- Converts text to lowercase

# Examples
```julia
# Process a single text file
doc = prepstring("path/to/textfile.txt")

# Process a directory of text files
docs = prepstring("path/to/directory")

# Process a raw string
text_string = prepstring("This is a raw string.")
```

# Errors
- Throws an ArgumentError if input_path is not a valid file path, a directory path, or a raw string.

# Notes
- The function assumes that text files in the directory have a .txt extension.
- The maximum length for the input path is set to 256 characters to comply with Windows file path limits.
"""
function prepstring(input_path::AbstractString)
    # Helper function to prepare a single document
    function prepare_document(content::String)
        contentdoc = StringDocument(content)
        prepare!(contentdoc, strip_punctuation | strip_whitespace | strip_case)
        return contentdoc
    end

    # Process a single file or a string
    if length(input_path) < 256 && isfile(input_path)
        content = read(input_path, String)
        return prepare_document(content)
        # Process a folder of text files
    elseif length(input_path) < 256 && isdir(input_path)
        documentstring = ""
        for (root, dirs, files) in walkdir(input_path)
            for file in files
                if endswith(file, ".txt")  # Assuming text files have .txt extension
                    file_path = joinpath(root, file)
                    content = read(file_path, String)
                    documentstring = string("$documentstring", "$content")
                end
            end
        end
        return prepare_document(documentstring)# 
    # Process a raw string input
    elseif isa(input_path, String)
        return prepare_document(input_path)
    else
        throw(ArgumentError("Input must be a file path, directory path, or a string"))
    end
end

"""
    conttbl(input_string::StringDocument{String}, target_word::AbstractString, windowsize::Int64=5, minfreq::Int64=3)

Generate a contingency table for a target word in a given text document, analyzing its surrounding context within a specified window size.

# Arguments
- `input_string::StringDocument{String}`: The input text document to be analyzed, represented as a `StringDocument` from the `TextAnalysis` package.
- `target_word::AbstractString`: The word for which the context analysis is to be performed.
- `windowsize::Int64=5`: The number of words to consider on either side of the target word for context (default is 5).
- `minfreq::Int64=3`: The minimum frequency threshold for words to be included in the contingency table (default is 3).

# Returns
- A DataFrame containing the contingency table with columns:
    - `Collocate`: The context word.
    - `a`: Frequency of the context word in the target window.
    - `b`: Frequency of the context word outside the target window but in the document.
    - `c`: Frequency of other words in the target window.
    - `d`: Frequency of other words outside the target window but in the document.
    - `m`, `n`, `k`, `l`, `N`, `E₁₁`, `E₁₂`, `E₂₁`, `E₂₂`: Various calculated statistics for further analysis.

# Description
The `conttbl` function processes a `StringDocument`, tokenizes the text, and identifies the indices of the target word. It then collects context words within a specified window size around each occurrence of the target word. The function calculates the frequency of these context words and constructs a contingency table. The table includes the frequency of context words inside and outside the target window and computes various statistical measures used for further linguistic analysis.

# Example
```julia
using TextAnalysis, DataFrames, Chain

doc = StringDocument("This is a sample text document for testing the contingency table function. This function will analyze the text and provide useful statistics.")
conttbl(doc, "text", 5, 3)

# Notes

The function assumes that the input text is preprocessed and tokenized correctly.

Ensure that the TextAnalysis package is properly imported and used for StringDocument and tokenization.
"""
function conttbl(input_string::StringDocument{String}, target_word::AbstractString, windowsize::Int64=5, minfreq::Int64=3)

    input_string_tokenized = TextAnalysis.tokenize(language(input_string), text(input_string))

    indices = findall(==(target_word), input_string_tokenized)

    context_indices = falses(length(input_string_tokenized))

    contexts = Array{Tuple{UnitRange{Int64},UnitRange{Int64}},1}()

    for index in indices
        left_start_index = max(1, index - windowsize)
        right_end_index = min(length(input_string_tokenized), index + windowsize)

        context_indices[left_start_index:index-1] .= true
        context_indices[index+1:right_end_index] .= true
        push!(contexts, (left_start_index:index-1, index+1:right_end_index))
    end

    unique_counts = Dict{String,Int}()
    seen_words = Set{String}()  # To track unique words for each tuple

    for idx in contexts
        empty!(seen_words)  # Clear the set for the new tuple

        words = input_string_tokenized[union(idx...)]
        for word in words
            if !in(word, seen_words)
                push!(seen_words, word)
                unique_counts[word] = get(unique_counts, word, 0) + 1
            end
        end
    end

    node_context_words = input_string_tokenized[context_indices]

    a = freqtable(node_context_words)
    filter!(x -> x >= minfreq, a)

    unique_counts_intersect = intersect(Set(keys(unique_counts)), names(a)...)

    b = Dict(key => length(indices) - unique_counts[key] for key in unique_counts_intersect)
    b = freqtable(collect(keys(b)), weights=collect(values(b)))

    context_indices[indices] .= true
    context_word_indices = .!context_indices
    context_words = input_string_tokenized[context_word_indices]

    set_node_context_words = Set(names(a)[1])

    context_words_updated = filter(x -> x in set_node_context_words, context_words)

    c = freqtable(context_words_updated)

    idx = 0
    for name in Set(names(a)[1])
        get!(c.dicts[1], name) do
            idx += 1
            return length(c) + idx
        end
    end

    append!(c.array, zeros(Int64, length(c) - (length(c) - idx)))

    reference_word_length = length(filter(x -> !in(x, set_node_context_words), context_words))
    d = reference_word_length .- c

    con_table_names = names(a)[1]
    con_table = hcat(a, b, c, d)
    # NamedArray(con_table.array, (["a", "b", "c", "d"], con_table_names))
    con_df = DataFrame(con_table.array, Symbol.(["a", "b", "c", "d"]))

    insertcols!(con_df, 1, :Collocate => Symbol.(con_table_names))

    # *m = a + b*, *n = c + d*, *k = a + c*, *l = b + d* and *N = m + n* 
    @chain con_df begin
        transform!(AsTable([:a, :b]) => (x -> x.a .+ x.b) => :m)
        transform!(AsTable([:c, :d]) => (x -> x.c .+ x.d) => :n)
        transform!(AsTable([:a, :c]) => (x -> x.a .+ x.c) => :k)
        transform!(AsTable([:b, :d]) => (x -> x.b .+ x.d) => :l)
        transform!(AsTable([:m, :n]) => (x -> x.m .+ x.n) => :N)
        transform!(AsTable([:m, :k, :N]) => (x -> (x.m .* x.k) ./ x.N) => :E₁₁)
        transform!(AsTable([:m, :l, :N]) => (x -> (x.m .* x.l) ./ x.N) => :E₁₂)
        transform!(AsTable([:n, :l, :N]) => (x -> (x.n .* x.l) ./ x.N) => :E₂₁)
        transform!(AsTable([:n, :d, :N]) => (x -> (x.n .* x.d) ./ x.N) => :E₂₂)

    end
end

"""
    count_substring_regex_optimized(text::String, substring::String) -> Int
    count_substrings(text::String, substrings::Vector{String}) -> Dict{String, Int}

Count the number of occurrences of one or more substrings within a given text using regular expressions.

# Methods

## Single Substring

    count_substring_regex_optimized(text::String, substring::String) -> Int

Count the number of occurrences of a single substring within the text.

### Arguments
- `text::String`: The text in which to search for the substring.
- `substring::String`: The substring to count within the text.

### Returns
- `Int`: The number of times the substring appears in the text.

### Example
```julia
text = "hello world, hello universe"
substring = "hello"
count = count_substring_regex_optimized(text, substring)
println(count)  # Output: 2
```

## Multiple Substrings

    count_substrings(text::String, substrings::Vector{String}) -> Dict{String, Int}

Count the number of occurrences of each substring in a given vector within the larger string. 

### Arguments
- `text::String`: The text in which to search for the substrings.
- `substrings::Vector{String}`: A vector of substrings to count within the text.

### Returns
- `Dict{String, Int}`: A dictionary where keys are substrings and values are their respective counts in the laeger string.

### Example

```julia-doc
text = "hello world, hello universe"
substrings = ["hello", "world"]
counts = count_substrings(text, substrings)
println(counts)  # Output: Dict("hello" => 2, "world" => 1)
```

# Description

These methods utilize regular expressions to efficiently find and count all non-overlapping occurrences of one or more substrings within the provided text. The single substring function returns an integer count, while the multiple substrings function returns a dictionary of counts.
"""
function count_substrings(text::String, substring::String)::Int
    count = 0
    for _ in eachmatch(Regex(escape_string(substring)), text)
        count += 1
    end
    return count
end

function count_substrings(text::String, substrings::Vector{String})::Dict{String,Int}
    counts = Dict{String,Int}()
    for substring in substrings
        count = 0
        for _ in eachmatch(Regex(escape_string(substring)), text)
            count += 1
        end
        counts[substring] = count
    end
    return count
end

"""
    find_prior_words(strdoc::StringDocument, substring::String, n::Int) -> Tuple{Set{String}, Int64}

Find the unique words that appear `n` words before each match of the given substring within a larger string, and return the set of unique words along with its length.

# Arguments
- `strdoc::StringDocument`: The text document in which to search for the substring.
- `substring::String`: The substring to search within the text.
- `n::Int`: The number of words to look back before each match.

# Returns
- `Tuple{Set{String}, Int64}`: A tuple containing:
    - A set of unique words that appear `n` words before each match of the given substring.
    - The number of unique words in the set.

# Example
```julia
strdoc = StringDocument("Γρήγορη καφέ αλεπού πηδάει πάνω από τον τεμπέλη σκύλο. Η γρήγορη καφέ αλεπού είναι πολύ γρήγορη.")
substring = "γρήγορη"
n = 2
unique_prior_words, count = find_prior_words(strdoc, substring, n)
println(unique_prior_words)  # Output: Set(["Η", "καφέ", "από"])
println(count)  # Output: 3

This function utilizes regular expressions to find the substring within the text and then identifies the unique words that appear `n` words before each match. It handles Unicode boundaries correctly to ensure valid indexing and processes the text to strip punctuation, whitespace, and normalize case.
"""
function find_prior_words(strdoc::StringDocument, substring::String, n::Int)::Tuple{Set{String},Int64}
    prior_words = Set{String}()
    max_chars_before = n * 10 # Approximation for characters to look back
    # Iterate over each substring and find prior words
    for m in eachmatch(Regex(escape_string(substring)), strdoc.text)
        match_start = m.offset[1]
        start_pos = match_start

        # Move start_pos back by max_chars_before, respecting Unicode boundaries
        for _ in 1:max_chars_before
            if start_pos <= 1
                break
            end
            start_pos = prevind(strdoc.text, start_pos)
        end

        prior_text = strdoc.text[start_pos:match_start]

        # Create a temporary StringDocument for prior_text
        prior_doc = StringDocument(prior_text)
        prepare!(prior_doc, strip_punctuation | strip_whitespace | strip_case)

        words = split(prior_doc.text)
        if length(words) >= n
            push!(prior_words, words[end-n+1:end]...)
        else
            push!(prior_words, words...)
        end
    end

    return (prior_words, length(prior_words))
end


# *m = a + b*, *n = c + d*, *k = a + c*, *l = b + d* and *N = m + n*.


#  create a function that within a window size that the user will set, will count the type frequency of the words that are in the window of the node word. It will separate the type frequencies of the words that precede and come after the node word.

"""
    listmetrics() -> Vector{Symbol}

Returns a list of all association metrics supported by the package. This function provides an easy way for users to discover and understand the different types of metrics they can calculate using the package.

# Returns
- `Vector{Symbol}`: A vector containing the symbols representing each of the supported association metrics.

# Examples

```julia-doc
metricsvector = listmetrics()
println(metricsvector)
```

"""
function listmetrics()
    return [:PMI, :PMI², :PMI³, :PPMI, :LLR, :LLR2, :LLR², :DeltaPi, :MinSens, :Dice, :LogDice, :RelRisk, :LogRelRisk, :RiskDiff, :AttrRisk, :OddsRatio, :LogRatio, :LogOddsRatio, :JaccardIdx, :OchiaiIdx, :PiatetskyShapiro, :YuleOmega, :YuleQ, :PhiCoef, :CramersV, :TschuprowT, :ContCoef, :CosineSim, :OverlapCoef, :KulczynskiSim, :TanimotoCoef, :RogersTanimotoCoef, :RogersTanimotoCoef2, :HammanSim, :HammanSim2, :GoodmanKruskalIdx, :GowerCoef, :GowerCoef2, :CzekanowskiDiceCoef, :SorgenfreyIdx, :SorgenfreyIdx2, :MountfordCoef, :MountfordCoef2, :SokalSneathIdx, :SokalMichenerCoef, :Tscore, :Zscore, :ChiSquare, :FisherExactTest, :CohensKappa]
end