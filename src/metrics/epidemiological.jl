# =====================================
# File: src/metrics/epidemiological.jl
# Epidemiological metrics
# =====================================

# Relative Risk
function eval_relrisk(data::ContingencyTable)
    @extract_values data a c m n
    (a .* n) ./ max.(c .* m, eps())
end

# Log Relative Risk
function eval_logrelrisk(data::ContingencyTable)
    @extract_values data a c m n
    log_safe.(a) .+ log_safe.(n) .- log_safe.(c) .- log_safe.(m)
end

# Risk Difference
function eval_riskdiff(data::ContingencyTable)
    @extract_values data a c m n
    (a ./ max.(m, eps())) .- (c ./ max.(n, eps()))
end

# Attributable Risk (same as Risk Difference)
function eval_attrrisk(data::ContingencyTable)
    eval_riskdiff(data)
end

# Odds Ratio
function eval_oddsratio(data::ContingencyTable)
    @extract_values data a b c d
    (a .* d) ./ max.(b .* c, eps())
end

# Log Odds Ratio
function eval_logoddsratio(data::ContingencyTable)
    @extract_values data a b c d
    log_safe.(a) .+ log_safe.(d) .- log_safe.(b) .- log_safe.(c)
end
