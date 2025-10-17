# Metric Catalog (auto-generated)

This table lists all available metrics currently registered in **TextAssociations.jl**.
It is generated automatically from the package’s internal list of metric types (`METRIC_TYPES`)
via the `available_metrics()` function, ensuring the documentation always stays up-to-date.

```@example
using TextAssociations, Markdown

names = sort(string.(available_metrics()))
rows = String[]
push!(rows, "| Metric | Call example |")
push!(rows, "|:------ |:------------ |")
for n in names
    push!(rows, "| `$(n)` | `assoc_score($(n), data)` |")
end
Markdown.parse(join(rows, "\n"))
```

---

### Notes
- The list above is populated dynamically at build time.
- You can inspect details of each metric type in the REPL using `?MetricName`.
- For implementation details, see **Internals → Metric Implementations**.
