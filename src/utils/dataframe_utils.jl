# =====================================
# File: src/utils/dataframe_utils.jl
# DataFrame utilities and metadata helpers
# =====================================

"""
    write_results(df::DataFrame) -> Dict{String, Any}

Extract all analysis metadata from a result DataFrame.

# Returns
Dictionary containing available metadata fields like metric, node, windowsize, etc.
"""
function write_results(df::DataFrame)
    info = Dict{String,Any}()
    metadata_keys = ["metric", "metrics", "node", "windowsize", "minfreq",
        "top_n", "analysis_type", "corpus_size", "timestamp"]

    for key in metadata_keys
        if haskey(metadata(df), key)
            info[key] = metadata(df, key)
        end
    end
    return info
end

"""
    analysis_metric(df::DataFrame) -> String

Get the metric(s) used in the analysis from DataFrame metadata.
"""
function analysis_metric(df::DataFrame)
    # Check for single metric first
    if haskey(metadata(df), "metric")
        return metadata(df, "metric")
    elseif haskey(metadata(df), "metrics")
        return metadata(df, "metrics")
    else
        return "Unknown"
    end
end

"""
    analysis_summary(df::DataFrame) -> String

Generate a human-readable summary of the analysis parameters.

# Example
```julia
julia> analysis_summary(results)
"Analysis of 'innovation' using PMI (window=5, minfreq=10)"
```
"""
function analysis_summary(df::DataFrame)
    metric = analysis_metric(df)
    node = metadata(df, "node", "Unknown")
    ws = metadata(df, "windowsize", "Unknown")
    mf = metadata(df, "minfreq", "Unknown")

    return "Analysis of '$node' using $metric (window=$ws, minfreq=$mf)"
end

"""
    copy_metadata!(dest::DataFrame, src::DataFrame)

Copy all metadata from source DataFrame to destination DataFrame.
"""
function copy_metadata!(dest::DataFrame, src::DataFrame)
    for (key, value) in metadata(src)
        metadata!(dest, key, value, style=:note)
    end
    return dest
end

"""
    combine_results_with_metadata(dfs::Vector{DataFrame}) -> DataFrame

Combine multiple result DataFrames while preserving metadata.
Creates a combined metadata entry listing all sources.
"""
function combine_results_with_metadata(dfs::Vector{DataFrame})
    if isempty(dfs)
        return DataFrame()
    end

    # Combine DataFrames
    combined = vcat(dfs..., cols=:union)

    # Collect metadata from all sources
    all_nodes = String[]
    all_metrics = String[]

    for df in dfs
        node = metadata(df, "node", nothing)
        metric = analysis_metric(df)

        if node !== nothing
            push!(all_nodes, node)
        end
        if metric != "Unknown"
            push!(all_metrics, metric)
        end
    end

    # Add combined metadata
    metadata!(combined, "combined_nodes", join(unique(all_nodes), ", "), style=:note)
    metadata!(combined, "combined_metrics", join(unique(all_metrics), ", "), style=:note)
    metadata!(combined, "n_sources", length(dfs), style=:note)
    metadata!(combined, "analysis_type", "combined_results", style=:note)

    return combined
end

"""
    filter_scores(df::DataFrame, min_score::Real; metric::Union{Symbol,Nothing}=nothing) -> DataFrame

Filter results by minimum score threshold, preserving metadata.

# Arguments
- `df`: Results DataFrame
- `min_score`: Minimum score threshold
- `metric`: Score column to use (default: :Score or first metric column)
"""
function filter_scores(df::DataFrame, min_score::Real; metric::Union{Symbol,Nothing}=nothing)
    # Determine which column to filter by
    score_col = if metric !== nothing
        metric
    elseif hasproperty(df, :Score)
        :Score
    else
        # Try to find a metric column (e.g., :PMI, :Dice, etc.)
        metric_cols = filter(col -> col ∉ [:Node, :Collocate, :Frequency, :DocFrequency], names(df))
        isempty(metric_cols) ? error("No score column found") : Symbol(metric_cols[1])
    end

    # Filter
    filtered = filter(row -> row[score_col] >= min_score, df)

    # Copy metadata
    copy_metadata!(filtered, df)

    # Add filter info to metadata
    metadata!(filtered, "filtered", true, style=:note)
    metadata!(filtered, "min_score_threshold", min_score, style=:note)
    metadata!(filtered, "filter_column", String(score_col), style=:note)

    return filtered
end

"""
    export_with_metadata(df::DataFrame, filename::String; format::Symbol=:csv)

Export DataFrame with metadata preserved in file header or separate file.
"""
function export_with_metadata(df::DataFrame, filename::String; format::Symbol=:csv)
    if format == :csv
        # Write metadata as comments at the top of CSV
        open(filename, "w") do io
            # Write metadata as comments
            println(io, "# Analysis Metadata")
            for (key, value) in metadata(df)
                println(io, "# $key: $value")
            end
            println(io, "#")

            # Write the actual CSV data
            CSV.write(io, df)
        end

    elseif format == :json
        # Include metadata in JSON structure
        json_data = Dict(
            "metadata" => Dict(metadata(df)),
            "data" => [Dict(pairs(row)) for row in eachrow(df)]
        )

        open(filename, "w") do io
            JSON.print(io, json_data, 2)
        end

    else
        throw(ArgumentError("Unsupported format: $format. Use :csv or :json"))
    end
end

"""
    load_with_metadata(filename::String; format::Symbol=:csv) -> DataFrame

Load DataFrame with preserved metadata.
"""
function load_with_metadata(filename::String; format::Symbol=:csv)
    if format == :csv
        # Read metadata from comments
        meta_dict = Dict{String,Any}()

        # First pass: read metadata
        open(filename, "r") do io
            for line in eachline(io)
                if startswith(line, "# ") && contains(line, ":")
                    # Parse metadata line
                    content = line[3:end]  # Remove "# "
                    if contains(content, ":")
                        key, value = split(content, ":", limit=2)
                        meta_dict[strip(key)] = strip(value)
                    end
                elseif !startswith(line, "#")
                    break  # End of metadata section
                end
            end
        end

        # Load the actual CSV
        df = CSV.read(filename, DataFrame, comment="#")

        # Apply metadata
        for (key, value) in meta_dict
            metadata!(df, key, value, style=:note)
        end

        return df

    elseif format == :json
        # Load JSON with metadata
        json_data = JSON.parsefile(filename)

        # Create DataFrame from data
        df = DataFrame(json_data["data"])

        # Apply metadata
        if haskey(json_data, "metadata")
            for (key, value) in json_data["metadata"]
                metadata!(df, key, value, style=:note)
            end
        end

        return df

    else
        throw(ArgumentError("Unsupported format: $format. Use :csv or :json"))
    end
end