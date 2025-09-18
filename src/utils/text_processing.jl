# =====================================
# File: src/utils/text_processing.jl
# Text preprocessing utilities
# =====================================

"""
    prepstring(input_path::AbstractString; kwargs...) -> StringDocument

Prepare and preprocess text from various sources.

# Arguments
- `input_path`: File path, directory path, or raw text string
- `strip_punctuation`: Remove punctuation (default: true)
- `strip_whitespace`: Normalize whitespace (default: true)
- `strip_case`: Convert to lowercase (default: true)

# Returns
A preprocessed `StringDocument` object.
"""
function prepstring(input_path::AbstractString;
    # Keep your original keywords for compatibility:
    strip_punctuation::Bool=true,
    strip_whitespace::Bool=false,   # IMPORTANT: do NOT delete spaces by default
    strip_case::Bool=true,

    # New, safer controls:
    punctuation_to_space::Bool=true,       # map punctuation → spaces (prevents "glued" words)
    normalize_whitespace::Bool=true,       # collapse multiple spaces
    unicode_form::Symbol=:NFC,             # NFC preserves tonos
    use_prepare::Bool=false                # set true only if you’re sure your pipeline is tonos-safe
)

    # --- helper to prepare one string (TONOS-SAFE) ---
    function prepare_document(content::String)
        # 1) Unicode normalization (preserves tonos in Greek)
        s = Base.Unicode.normalize(content, unicode_form)

        # 2) Punctuation handling (prefer mapping → spaces to keep boundaries)
        if strip_punctuation && punctuation_to_space
            s = replace(s, r"\p{P}+" => " ")
        elseif strip_punctuation
            # WARNING: deleting punctuation may glue words; only do this if you really want it
            s = replace(s, r"\p{P}+" => "")
        end

        # 3) Whitespace handling
        if normalize_whitespace
            # Normalize all whitespace runs to a single space
            s = replace(s, r"\s+" => " ")
        elseif strip_whitespace
            # If caller explicitly wants to delete whitespace entirely (not recommended)
            s = replace(s, r"\s+" => "")
        end

        # 4) Case folding (Unicode-aware; keeps tonos)
        if strip_case
            s = lowercase(s)
        end

        # 5) Create the document
        doc = StringDocument(s)

        # 6) Optional TextAnalysis pipeline (disabled by default to protect tonos)
        if use_prepare
            pipeline = 0x00
            # If you already handled punctuation/whitespace/case above,
            # *do not* also set those flags here (to avoid double-processing).
            # Leave pipeline at 0x00 unless you have other TextAnalysis steps to apply.
            pipeline != 0x00 && prepare!(doc, pipeline)
        end

        return doc
    end

    # --- detect input kind (path / dir / raw string) ---
    if length(input_path) < 256 && isfile(input_path)
        return prepare_document(read(input_path, String))
    elseif length(input_path) < 256 && isdir(input_path)
        texts = String[]
        for (root, _, files) in walkdir(input_path)
            for file in files
                endswith(file, ".txt") || continue
                file_path = joinpath(root, file)
                # decode per-file robustly and NFC-normalize (preserves tonos)
                push!(texts, read_text_smart(file_path; normalize_form=:NFC))
            end
        end
        return prepare_document(String(take!(buf)))
    else
        # Treat as raw string
        return prepare_document(input_path)
    end
end


"""
Convert input to string vector for vocabulary creation.
"""
tostringvector(doc::StringDocument) = tokens(doc)
tostringvector(vec::Vector{String}) = vec

"""
    createvocab(input::Union{StringDocument,Vector{String}}) -> OrderedDict

Create vocabulary dictionary from text input.
"""
function createvocab(input::Union{StringDocument,Vector{String}})
    string_vector = unique(tostringvector(input))
    vocab = OrderedDict{String,Int}()
    sizehint!(vocab, length(string_vector))

    for (index, key) in enumerate(string_vector)
        vocab[key] = index
    end
    return vocab
end
