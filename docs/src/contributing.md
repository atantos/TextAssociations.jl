# Contributing to TextAssociations.jl

We welcome contributions to TextAssociations.jl! This guide will help you get started.

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
    ct = ContingencyTable("test text", "test", 3, 1)
    results = assoc_score(MyMetric, ct)
    @test !isempty(results)
    @test all(isfinite.(results.MyMetric))
end

# 4. Add documentation
```

## Development Setup

### 1. Fork and Clone

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/yourusername/TextAssociations.jl.git
cd TextAssociations.jl

# Add upstream remote
git remote add upstream https://github.com/originalrepo/TextAssociations.jl.git
```

### 2. Create Development Environment

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()

# Install development dependencies
Pkg.add("Test")
Pkg.add("BenchmarkTools")
Pkg.add("Profile")
```

### 3. Create Feature Branch

```bash
git checkout -b feature-name
```

## Code Style

### Julia Style Guide

Follow the [Julia Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/):

```julia
# Good
function calculate_pmi(contingency_table::ContingencyTable)
    # ...
end

# Bad
function CalculatePMI(CT)
    # ...
end
```

### Documentation

All public functions should have docstrings:

````julia
"""
    my_function(arg1::Type1, arg2::Type2) -> ReturnType

Brief description of what the function does.

# Arguments
- `arg1`: Description of first argument
- `arg2`: Description of second argument

# Returns
- Description of return value

# Examples
```julia
result = my_function(value1, value2)
````

"""
function my_function(arg1::Type1, arg2::Type2) # Implementation
end

````

## Testing

### Running Tests

```julia
using Pkg
Pkg.test()

# Or run specific test files
include("test/runtests.jl")
````

### Writing Tests

```julia
using Test
using TextAssociations

@testset "My Feature Tests" begin
    @testset "Basic functionality" begin
        # Test normal cases
        @test my_function(1, 2) == 3
    end

    @testset "Edge cases" begin
        # Test edge cases
        @test_throws ArgumentError my_function(-1, 2)
    end

    @testset "Performance" begin
        # Optional performance tests
        @test @elapsed(my_function(1, 2)) < 1.0
    end
end
```

## Pull Request Process

### 1. Before Submitting

- [ ] All tests pass
- [ ] Code follows style guide
- [ ] Documentation is updated
- [ ] Commit messages are clear

### 2. PR Template

```markdown
## Description

Brief description of changes

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing

- [ ] Tests pass locally
- [ ] New tests added for changes

## Checklist

- [ ] Code follows style guide
- [ ] Self-review completed
- [ ] Documentation updated
```

### 3. Review Process

- Maintainers will review your PR
- Address any feedback
- Once approved, it will be merged

## Project Structure

```
TextAssociations.jl/
├── src/
│   ├── TextAssociations.jl    # Main module
│   ├── types.jl               # Type definitions
│   ├── api.jl                 # Public API
│   ├── core/
│   │   ├── contingency_table.jl
│   │   └── corpus_analysis.jl
│   ├── metrics/
│   │   ├── _iface.jl          # Interface
│   │   ├── metrics.jl         # Metric implementations
│   │   └── ...
│   └── utils/
│       ├── text_processing.jl
│       └── ...
├── test/
│   ├── runtests.jl
│   └── ...
├── docs/
│   ├── make.jl
│   └── src/
└── README.md
```

## Adding Documentation

### 1. Add Documentation Files

````markdown
# docs/src/my_feature.md

```@meta
CurrentModule = TextAssociations
```
````

# My Feature

Description of the feature...

## Examples

```@example myfeature
using TextAssociations

# Example code
```

````

### 2. Update make.jl

```julia
pages = [
    # ...
    "My Feature" => "my_feature.md",
    # ...
]
````

### 3. Build Documentation Locally

```julia
cd("docs")
include("make.jl")
```

## Performance Considerations

When contributing performance improvements:

### 1. Benchmark Before and After

```julia
using BenchmarkTools

# Before changes
@benchmark old_function($data)

# After changes
@benchmark new_function($data)
```

### 2. Profile Code

```julia
using Profile

@profile for i in 1:1000
    my_function(data)
end

Profile.print()
```

### 3. Memory Profiling

```julia
using Profile.Allocs

Profile.Allocs.@profile my_function(data)
```

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive criticism
- No harassment or discrimination

### Communication

- Use clear, descriptive commit messages
- Comment complex code
- Update documentation with changes
- Respond to feedback constructively

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- MAJOR: Incompatible API changes
- MINOR: Backwards-compatible functionality
- PATCH: Backwards-compatible bug fixes

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in Project.toml
- [ ] Tag created

## Getting Help

### For Contributors

- Open an issue for discussion
- Ask in discussions
- Check existing issues and PRs

### Resources

- [Julia Documentation](https://docs.julialang.org/)
- [Julia Discourse](https://discourse.julialang.org/)
- [Package Development Guide](https://pkgdocs.julialang.org/)

## Recognition

Contributors are recognized in:

- CONTRIBUTORS.md file
- Release notes
- Documentation credits

Thank you for contributing to TextAssociations.jl!
