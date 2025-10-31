# =====================================
# File: src/utils/text_processing.jl
# Text preprocessing utilities
# =====================================

"""
    strip_diacritics(s::AbstractString; target_form::Symbol = :NFC) -> String

Remove all combining diacritics (e.g., Greek tonos, dialytika) using Unicode normalization.

Internally:
- canonically decomposes as needed,
- strips combining marks (`Mn`),
- and normalizes to `target_form` (default **:NFC**).

If you don’t care about the final form, leave `target_form` at the default.

Example:
    julia> strip_diacritics("ένα το χελιδόϊι")
    "ενα το χελιδοιι"

"""
function strip_diacritics(s::AbstractString; target_form::Symbol=:NFC)
    # 1) strip marks (returns NFC by default)
    s_nomark = Unicode.normalize(s; stripmark=true)
    # 2) ensure desired final form (idempotent if target_form == :NFC)
    return Unicode.normalize(s_nomark, target_form)
end

"""
    normalize_node(node::AbstractString, config::TextNorm) -> String

Normalize a node word or n-gram according to the given TextNorm configuration.
This is the single source of truth for node normalization.

For multi-word nodes, each word is normalized individually then rejoined with single space.

# Examples
```julia
julia> cfg = TextNorm(strip_case=true, strip_accents=true);
julia> normalize_node("New York", cfg)
"new york"

julia> normalize_node("machine learning", cfg)
"machine learning"
"""
function normalize_node(node::AbstractString, config::TextNorm)

    # Trim and normalize whitespace first
    node_trimmed = strip(node)
    isempty(node_trimmed) && return ""

    # Split on whitespace to handle multi-word nodes
    words = split(node_trimmed)

    # Normalize each word individually
    normalized_words = map(words) do word
        # Start with unicode normalization
        normalized = Unicode.normalize(word, config.unicode_form)

        # Apply transformations in consistent order
        if config.strip_case
            normalized = lowercase(normalized)
        end

        if config.strip_accents
            normalized = strip_diacritics(normalized; target_form=config.unicode_form)
        end

        normalized
    end

    # Rejoin with single space
    return join(normalized_words, " ")
end

"""
    split_ngram(ngram::AbstractString) -> Vector{String}

Split a normalized n-gram into its constituent words.

# Examples
```julia
julia> split_ngram("machine learning")
2-element Vector{String}:
 "machine"
 "learning"
```
"""
split_ngram(ngram::AbstractString) = split(ngram)

"""
    ngram_length(ngram::AbstractString) -> Int

Get the number of words in an n-gram.

# Examples
```julia
julia> ngram_length("machine learning")
2

julia> ngram_length("new york city")
3
```
"""
ngram_length(ngram::AbstractString) = length(split_ngram(ngram))

"""
    is_single_word(node::AbstractString) -> Bool

Check if a node is a single word (unigram) or multi-word (n-gram).

# Examples
```julia
julia> is_single_word("hello")
true

julia> is_single_word("hello world")
false
```
"""
is_single_word(node::AbstractString) = !contains(strip(node), r"\s")

"""
    find_ngram_positions(tokens::Vector{String}, ngram::AbstractString) -> Vector{Int}

Find all starting positions where the n-gram occurs in the token sequence.
Returns the index of the first token of each occurrence.

# Arguments
- `tokens`: Vector of tokens (already normalized)
- `ngram`: Normalized n-gram string (e.g., "machine learning")

# Returns
Vector of starting positions (1-indexed) where the n-gram occurs

# Examples
```julia
julia> tokens = ["i", "love", "machine", "learning", "and", "machine", "learning"];
julia> find_ngram_positions(tokens, "machine learning")
2-element Vector{Int64}:
 3
 6
```
"""
function find_ngram_positions(tokens::Vector{String}, ngram::AbstractString)
    ngram_words = split_ngram(ngram)
    n = length(ngram_words)
    n == 0 && return Int[]

    positions = Int[]

    # Scan through tokens looking for n-gram matches
    for i in 1:(length(tokens)-n+1)
        # Check if the next n tokens match the n-gram
        if view(tokens, i:i+n-1) == ngram_words
            push!(positions, i)
        end
    end

    return positions
end

"""
    extract_ngram_contexts(tokens::Vector{String}, 
                          ngram_positions::Vector{Int},
                          ngram_length::Int,
                          windowsize::Int) -> Tuple{Vector{Bool}, Vector{Tuple{UnitRange{Int},UnitRange{Int}}}}

Extract context words around n-gram occurrences.

# Arguments
- `tokens`: Full token vector
- `ngram_positions`: Starting positions of n-gram occurrences
- `ngram_length`: Number of words in the n-gram
- `windowsize`: Size of context window (in tokens)

# Returns
Tuple of:
- Boolean mask indicating which tokens are in any context window
- Vector of (left_range, right_range) tuples for each occurrence

# Notes
The window extends `windowsize` tokens from the n-gram boundaries:
- Left context: [position - windowsize : position - 1]
- Right context: [position + ngram_length : position + ngram_length + windowsize - 1]
"""
function extract_ngram_contexts(tokens::Vector{String},
    ngram_positions::Vector{Int},
    ngram_length::Int,
    windowsize::Int)
    context_mask = falses(length(tokens))
    contexts = Tuple{UnitRange{Int},UnitRange{Int}}[]

    for pos in ngram_positions
        # Calculate boundaries
        ngram_end = pos + ngram_length - 1

        # Left context: windowsize tokens before the n-gram
        left_start = max(1, pos - windowsize)
        left_end = pos - 1

        # Right context: windowsize tokens after the n-gram
        right_start = ngram_end + 1
        right_end = min(length(tokens), ngram_end + windowsize)

        # Mark context tokens
        if left_end >= left_start
            context_mask[left_start:left_end] .= true
        end
        if right_end >= right_start
            context_mask[right_start:right_end] .= true
        end

        # Store ranges
        push!(contexts, (left_start:left_end, right_start:right_end))
    end

    return context_mask, contexts
end


"""
    prep_string(input_path::AbstractString, config::TextNorm) -> StringDocument

Prepare and preprocess text from various sources into a `StringDocument`.

# Arguments
- `input_path`: File path, directory path, or raw text string.

## Preprocessing options
Uses `TextNorm` configuration for all preprocessing options.

# Returns
A preprocessed `StringDocument` object suitable for downstream corpus analysis.
"""
function prep_string(input_path::AbstractString, config::TextNorm=TextNorm())

    function prepare_document(content::String, config::TextNorm)
        # 1) Unicode normalization
        s = Unicode.normalize(content, config.unicode_form)

        # 2) Punctuation handling
        if config.strip_punctuation && config.punctuation_to_space
            s = replace(s, r"\p{P}+" => " ")
        elseif config.strip_punctuation
            s = replace(s, r"\p{P}+" => "")
        end

        # 3) Whitespace handling
        if config.normalize_whitespace
            s = replace(s, r"\s+" => " ")
        elseif config.strip_whitespace
            s = replace(s, r"\s+" => "")
        end

        # 4) Case folding
        if config.strip_case
            s = lowercase(s)
        end

        # 5) Strip diacritics if requested
        if config.strip_accents
            s = strip_diacritics(s; target_form=config.unicode_form)
        end

        # 6) Create document
        doc = StringDocument(s)

        # 7) Optional TextAnalysis pipeline
        if config.use_prepare
            prepare!(doc, 0x00)
        end

        return doc
    end

    # Detect input type and process
    if length(input_path) < 256 && isfile(input_path)
        return prepare_document(read(input_path, String), config)
    elseif length(input_path) < 256 && isdir(input_path)
        texts = String[]
        for (root, _, files) in walkdir(input_path)
            for file in files
                endswith(file, ".txt") || continue
                file_path = joinpath(root, file)
                push!(texts, read_text_smart(file_path))
            end
        end
        return prepare_document(join(texts, " "), config)
    else
        # Treat as raw string
        return prepare_document(input_path, config)
    end
end

"""
Convert input to string vector for vocabulary creation.
"""
tostringvector(doc::StringDocument) = tokens(doc)
tostringvector(vec::Vector{String}) = vec

"""
    build_vocab(input::Union{StringDocument,Vector{String}}) -> OrderedDict

Create vocabulary dictionary from text input.
"""
function build_vocab(input::Union{StringDocument,Vector{String}})
    string_vector = unique(tostringvector(input))
    vocab = OrderedDict{String,Int}()
    sizehint!(vocab, length(string_vector))

    for (index, key) in enumerate(string_vector)
        vocab[key] = index
    end
    return vocab
end
