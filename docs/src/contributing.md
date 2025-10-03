# [Contributing Guide](@id contributing)

We welcome contributions to `TextAssociations.jl`! This guide will help you get started.

## Ways to Contribute

### 1. Report Bugs

Found a bug? Please [open an issue](https://github.com/yourusername/TextAssociations.jl/issues) with:

- A clear description of the problem
- A minimal reproducible example
- Your system information (Julia version, OS)
- Any error messages or stack traces

### 2. Suggest Features

Have an idea for a new feature? Open an issue with:

- Description of the proposed feature
- Use cases and examples
- How it fits with existing functionality

### 3. Improve Documentation

Documentation improvements are always welcome:

- Fix typos or unclear explanations
- Add examples and tutorials
- Improve API documentation
- Translate documentation

### 4. Add New Metrics

Adding a new association metric:

```julia
# 1. Add the metric type in src/types.jl
abstract type MyMetric <: AssociationMetric end

# 2. Implement the evaluation function in src/metrics/
function eval_mymetric(data::AssociationDataFormat)
    @extract_values data a b c d N

    # Your metric calculation
    return result_vector
end

# 3. Add tests in test/metrics_test.jl
@testset "MyMetric" begin
    ct = ContingencyTable("test text", "test", windowsize=3, minfreq=1)
    results = assoc_score(MyMetric, ct)
    @test !isempty(results)
    @test all(isfinite.(results.MyMetric))
end

# 4. Add documentation
```
