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
    prepstring(input_path::AbstractString; kwargs...) -> StringDocument

Prepare and preprocess text from various sources into a `StringDocument`.

# Arguments
- `input_path`: File path, directory path, or raw text string.

## Preprocessing options
- `strip_punctuation::Bool=true`: Remove punctuation.  
- `punctuation_to_space::Bool=true`: Replace punctuation with spaces (avoids glued words).  
- `strip_whitespace::Bool=false`: Remove whitespace entirely (not recommended).  
- `normalize_whitespace::Bool=true`: Collapse multiple spaces to one.  
- `strip_case::Bool=true`: Convert to lowercase (Unicode-aware).  
- `strip_accents::Bool=false`: Remove diacritics (tonos, dialytika) using Unicode normalization.  
- `unicode_form::Symbol=:NFC`: Unicode normalization form to apply (`:NFC` by default).  
- `use_prepare::Bool=false`: Apply TextAnalysis `prepare!` pipeline (disabled by default to protect tonos).

# Behavior
- Text can be provided as a file, directory of `.txt` files, or raw string.  
- For directories, all `.txt` files are concatenated before preprocessing.  
- Unicode normalization (`unicode_form`) is applied first.  
- If `strip_accents=true`, diacritics are removed after normalization.  
- Output is wrapped in a `StringDocument`.

# Returns
A preprocessed `StringDocument` object suitable for downstream corpus analysis.
"""
function prepstring(input_path::AbstractString;
    # Keep your original keywords for compatibility:
    strip_punctuation::Bool=true,
    strip_whitespace::Bool=false,   # IMPORTANT: do NOT delete spaces by default
    strip_case::Bool=true,

    # New, safer controls (as you had them):
    punctuation_to_space::Bool=true,       # map punctuation → spaces (prevents "glued" words)
    normalize_whitespace::Bool=true,       # collapse multiple spaces
    unicode_form::Symbol=:NFC,             # NFC preserves tonos
    use_prepare::Bool=false,               # set true only if you’re sure your pipeline is tonos-safe

    # NEW toggle (defaults off to preserve current behaviour):
    strip_accents::Bool=false
)

    # --- helper to prepare one string (TONOS-SAFE) ---
    function prepare_document(content::String)
        # 1) Unicode normalization (preserves tonos in Greek by default)
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

        # 4.5) OPTIONAL: strip diacritics/tonos/dialytika if requested
        if strip_accents
            s = strip_diacritics(s; target_form=unicode_form)
        end

        # 5) Create the document
        doc = StringDocument(s)

        # 6) Optional TextAnalysis pipeline (disabled by default to protect tonos)
        if use_prepare
            pipeline = 0x00
            # If you already handled punctuation/whitespace/case above,
            # *do not* also set those flags here (to avoid double-processing).
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
                # your environment already has a tonos-safe reader; keep as-is:
                push!(texts, read_text_smart(file_path))  # unchanged call
            end
        end
        return prepare_document(join(texts, " "))
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
