# =====================================
# File: src/utils/io.jl
# I/O utilities
# =====================================

using StringEncodings

function read_text_smart(path::AbstractString)::String
    bytes = read(path)
    n = length(bytes)
    # UTF-8 BOM
    if n ≥ 3 && bytes[1] == 0xEF && bytes[2] == 0xBB && bytes[3] == 0xBF
        return String(bytes[4:end])
        # UTF-16 LE BOM
    elseif n ≥ 2 && bytes[1] == 0xFF && bytes[2] == 0xFE
        return StringEncodings.decode(bytes, "UTF-16LE")
        # UTF-16 BE BOM
    elseif n ≥ 2 && bytes[1] == 0xFE && bytes[2] == 0xFF
        return StringEncodings.decode(bytes, "UTF-16BE")
    else
        # Try UTF-8 first
        try
            return String(bytes)
        catch
            # Fallback to Windows-1253 (Greek)
            return StringEncodings.decode(bytes, "windows-1253")
        end
    end
end
