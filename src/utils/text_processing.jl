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

Normalize a node word according to the given TextNorm configuration.
This is the single source of truth for node normalization.
"""
function normalize_node(node::AbstractString, config::TextNorm)
    # Start with unicode normalization
    normalized = Unicode.normalize(strip(node), config.unicode_form)

    # Apply transformations in consistent order
    if config.strip_case
        normalized = lowercase(normalized)
    end

    if config.strip_accents
        normalized = strip_diacritics(normalized; target_form=config.unicode_form)
    end

    return normalized
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
function prep_string(input_path::AbstractString, config::TextNorm)

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
