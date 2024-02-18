# helper functions for creating a vocabulary from a StringDocument or a Vector{String}
# Converts a StringDocument to Vector{String}
to_string_vector(doc::StringDocument) = tokens(doc)
# Identity function for Vector{String}
to_string_vector(vec::Vector{String}) = vec

function vocab(input::Union{StringDocument,Vector{String}})
    string_vector = to_string_vector(input) |> unique
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

# Function for preparing the input string
function prep_string(input_string::AbstractString)
    # for the length of the input string. Windows upper limit for filepaths is 256 chars
    if length(input_string) < 256 && isfile(input_string)
        content = read(input_string, String)
    elseif isa(input_string, String)
        content = input_string
    else
        throw(ArgumentError("Input string must be a file or a string"))
    end
    contentdoc = StringDocument(content)
    prepare!(contentdoc, strip_punctuation | strip_whitespace | strip_case)
    return contentdoc
end

function cont_tbl(input_string::StringDocument{String}, target_word::AbstractString, windowsize::Int64=5, minfreq::Int64=3)
    # input_string_clean = prepare_input_string(input_string)

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
    end
end

# *m = a + b*, *n = c + d*, *k = a + c*, *l = b + d* and *N = m + n*.


#  create a function that within a window size that the user will set, will count the type frequency of the words that are in the window of the node word. It will separate the type frequencies of the words that precede and come after the node word.
