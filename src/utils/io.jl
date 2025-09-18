# =====================================
# File: src/utils/io.jl
# I/O utilities
# =====================================

using StringEncodings

function read_text_smart(path::AbstractString; normalize_form::Symbol=:NFC)::String
    bytes = read(path)
    n = length(bytes)

    # 1) BOM-based fast paths
    if n ≥ 3 && bytes[1] == 0xEF && bytes[2] == 0xBB && bytes[3] == 0xBF
        s = String(bytes[4:end])  # UTF-8 with BOM
        return Base.Unicode.normalize(s, normalize_form)
    elseif n ≥ 2 && bytes[1] == 0xFF && bytes[2] == 0xFE
        s = StringEncodings.decode(bytes, "UTF-16LE")
        return Base.Unicode.normalize(s, normalize_form)
    elseif n ≥ 2 && bytes[1] == 0xFE && bytes[2] == 0xFF
        s = StringEncodings.decode(bytes, "UTF-16BE")
        return Base.Unicode.normalize(s, normalize_form)
    end

    # 2) Try UTF-8 (no BOM)
    try
        s = String(bytes)
        return Base.Unicode.normalize(s, normalize_form)
    catch
        # fallthrough
    end

    # 3) Try Windows-1253 then ISO-8859-7
    for enc in ("windows-1253", "ISO-8859-7")
        try
            s = StringEncodings.decode(bytes, enc)
            # Heuristic: if we decoded into mostly replacement chars, keep trying
            # (optional) Or check that Greek letters are present:
            # if occursin(r"\p{Greek}", s)
            return Base.Unicode.normalize(s, normalize_form)
            # end
        catch
            # try next encoding
        end
    end

    # 4) Last resort: lossy UTF-8 with replacement
    s = String(take!(TranscodingStreams.NoopStream(IOBuffer(bytes))))
    return Base.Unicode.normalize(s, normalize_form)
end
