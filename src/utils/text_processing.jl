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
    strip_punctuation::Bool=true,
    strip_whitespace::Bool=true,
    strip_case::Bool=true)

    # Build preprocessing pipeline
    pipeline = 0x00
    strip_punctuation && (pipeline |= TextAnalysis.strip_punctuation)
    strip_whitespace && (pipeline |= TextAnalysis.strip_whitespace)
    strip_case && (pipeline |= TextAnalysis.strip_case)

    # Helper function to prepare a single document
    function prepare_document(content::String)
        doc = StringDocument(content)
        pipeline != 0x00 && prepare!(doc, pipeline)
        return doc
    end

    # Process based on input type
    if length(input_path) < 256 && isfile(input_path)
        content = read(input_path, String)
        return prepare_document(content)
    elseif length(input_path) < 256 && isdir(input_path)
        contents = String[]
        for (root, dirs, files) in walkdir(input_path)
            for file in files
                if endswith(file, ".txt")
                    file_path = joinpath(root, file)
                    push!(contents, read(file_path, String))
                end
            end
        end
        return prepare_document(join(contents, " "))
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
