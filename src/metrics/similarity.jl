# =====================================
# File: src/metrics/similarity.jl
# Similarity metrics
# =====================================

# Dice Coefficient
function eval_dice(data::AssociationDataFormat)
    @extract_values data a m k
    (2 .* a) ./ max.(m .+ k, eps())
end

# Log Dice
function eval_logdice(data::AssociationDataFormat)
    @extract_values data a m k
    14 .+ log2_safe.(2 .* a) .- log2_safe.(m .+ k)
end

# Jaccard Index
function eval_jaccardidx(data::AssociationDataFormat)
    @extract_values data a b c
    a ./ max.(a .+ b .+ c, eps())
end

# Ochiai Index
function eval_ochiaiidx(data::AssociationDataFormat)
    @extract_values data a m k
    a ./ sqrt.(max.(m .* k, eps()))
end

# Cosine Similarity
function eval_cosinesim(data::AssociationDataFormat)
    @extract_values data a m k
    a ./ sqrt.(max.(m .* k, eps()))
end

# Overlap Coefficient
function eval_overlapcoef(data::AssociationDataFormat)
    @extract_values data a m k
    a ./ min.(m, k)
end

# Kulczynski Similarity
function eval_kulczynskisim(data::AssociationDataFormat)
    @extract_values data a m k
    0.5 .* ((a ./ max.(m, eps())) .+ (a ./ max.(k, eps())))
end

# Tanimoto Coefficient
function eval_tanimotocoef(data::AssociationDataFormat)
    @extract_values data a m k
    a ./ max.(m .+ k .- a, eps())
end

# Rogers-Tanimoto Coefficient
function eval_rogerstanimotocoef(data::AssociationDataFormat)
    @extract_values data a b c
    a ./ max.(a .+ 2 .* (b .+ c), eps())
end

# Rogers-Tanimoto Coefficient 2
function eval_rogerstanimotocoef2(data::AssociationDataFormat)
    @extract_values data a b c d
    (a .+ d) ./ max.(a .+ 2 .* (b .+ c) .+ d, eps())
end

# Hamman Similarity (fixed name from original HammanSim)
function eval_hammansim(data::AssociationDataFormat)
    @extract_values data a b c d N
    ((a .+ d) .- (b .+ c)) ./ N
end

# Hamman Similarity 2
function eval_hammansim2(data::AssociationDataFormat)
    @extract_values data a b c d
    (a .- d) ./ max.(a .+ b .+ c .- d, eps())
end

# Goodman-Kruskal Index
function eval_goodmankruskalidx(data::AssociationDataFormat)
    @extract_values data a b c d
    num = (a .* d) .- (b .* c)
    denom = (a .* d) .+ (b .* c)
    num ./ max.(denom, eps())
end

# Gower Coefficient
function eval_gowercoef(data::AssociationDataFormat)
    @extract_values data a b c
    a ./ max.(a .+ b .+ c, eps())
end

# Gower Coefficient 2
function eval_gowercoef2(data::AssociationDataFormat)
    @extract_values data a b c d
    (a .+ d) ./ max.(a .+ d .+ 2 .* (b .+ c), eps())
end

# Czekanowski-Dice Coefficient
function eval_czekanowskidicecoef(data::AssociationDataFormat)
    @extract_values data a b c
    (2 .* a) ./ max.(2 .* a .+ b .+ c, eps())
end

# Sorgenfrey Index
function eval_sorgenfreyidx(data::AssociationDataFormat)
    @extract_values data a b c
    num = 2 .* a .- b .- c
    denom = 2 .* a .+ b .+ c
    num ./ max.(denom, eps())
end

# Sorgenfrey Index 2
function eval_sorgenfreyidx2(data::AssociationDataFormat)
    @extract_values data a b c d
    (a .+ d) ./ max.(2 .* (a .+ d) .+ b .+ c, eps())
end

# Mountford Coefficient
function eval_mountfordcoef(data::AssociationDataFormat)
    @extract_values data a b c
    a ./ max.(a .+ 2 .* b .+ 2 .* c, eps())
end

# Mountford Coefficient 2
function eval_mountfordcoef2(data::AssociationDataFormat)
    @extract_values data a b c d k m
    num = a .+ d
    denom = a .+ d .+ 2 .* sqrt.(max.((b .+ c) .* (k .+ m), 0))
    num ./ max.(denom, eps())
end

# Sokal-Sneath Index
function eval_sokalsneathidx(data::AssociationDataFormat)
    @extract_values data a b c
    a ./ max.(a .+ 2 .* (b .+ c), eps())
end

# Sokal-Michener Coefficient
function eval_sokalmichenercoef(data::AssociationDataFormat)
    @extract_values data a d N
    (a .+ d) ./ N
end
