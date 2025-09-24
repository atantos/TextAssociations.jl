# Installation

```@meta
CurrentModule = TextAssociations
```

## Requirements

- Julia 1.6 or higher
- 4GB RAM minimum (8GB+ recommended for large corpora)
- Operating System: Windows, macOS, or Linux

## Standard Installation

### Using Julia's Package Manager

```julia
using Pkg
Pkg.add("TextAssociations")
```

### Development Version

To install the latest development version directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/yourusername/TextAssociations.jl")
```

### From Local Source

If you've cloned the repository:

```julia
using Pkg
Pkg.develop(path="/path/to/TextAssociations.jl")
```

## Installing Dependencies

### Required Dependencies

These are automatically installed with the package:

```julia
# Core dependencies
Pkg.add([
    "DataFrames",      # Data manipulation
    "TextAnalysis",    # Text processing
    "Chain",          # Pipeline operations
    "FreqTables",     # Frequency tables
    "DataStructures", # Ordered dictionaries
    "Unicode",        # Unicode operations
    "Statistics"      # Statistical functions
])
```

### Optional Dependencies

For additional functionality:

```julia
# I/O operations
Pkg.add([
    "CSV",           # CSV file support
    "JSON",          # JSON file support
    "XLSX",          # Excel file support
    "StringEncodings" # Character encoding
])

# Visualization
Pkg.add([
    "Plots",         # General plotting
    "StatsPlots"     # Statistical plots
])

# Parallel processing
Pkg.add("Distributed")

# Progress indicators
Pkg.add("ProgressMeter")
```

## Verification

### Basic Verification

```julia
using TextAssociations

# Check that the package loads
println("TextAssociations version: ", pkgversion(TextAssociations))

# List available metrics
metrics = available_metrics()
println("Available metrics: ", length(metrics))

# Run a simple test
text = "The quick brown fox jumps over the lazy dog"
ct = ContingencyTable(text, "the", 3, 1)
results = assoc_score(PMI, ct)
println("Test successful: ", !isempty(results))
```

### Comprehensive Test

```julia
using Pkg
Pkg.test("TextAssociations")
```

## Platform-Specific Notes

### Windows

On Windows, you might need to install the Visual C++ Redistributable if you encounter DLL errors:

- Download from [Microsoft's official site](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads)

### macOS

For Apple Silicon (M1/M2) Macs, ensure you're using native Julia:

```bash
# Check architecture
julia> Sys.ARCH
:aarch64  # For Apple Silicon
:x86_64   # For Intel Macs
```

### Linux

Some distributions may require additional system packages:

```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# Fedora/RHEL
sudo dnf install gcc gcc-c++ make
```

## Environment Management

### Creating a Project Environment

It's recommended to use Julia's environment management:

```julia
# Create new environment
mkdir MyProject
cd MyProject

# Activate environment
using Pkg
Pkg.activate(".")

# Add TextAssociations
Pkg.add("TextAssociations")
```

### Using Existing Environment

```julia
# Activate existing environment
using Pkg
Pkg.activate("path/to/MyProject")
Pkg.instantiate()  # Install all dependencies
```

### Environment File (Project.toml)

Create a `Project.toml` file for reproducible environments:

```toml
[deps]
TextAssociations = "uuid-here"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"

[compat]
TextAssociations = "0.1"
DataFrames = "1.0"
julia = "1.6"
```

## Troubleshooting Installation

### Common Issues

#### Package Not Found

```julia
# Error: Package TextAssociations not found
# Solution: Add the registry
using Pkg
Pkg.Registry.update()
Pkg.add("TextAssociations")
```

#### Version Conflicts

```julia
# Error: Unsatisfiable requirements detected
# Solution: Update all packages
Pkg.update()

# Or create a fresh environment
Pkg.activate(temp=true)
Pkg.add("TextAssociations")
```

#### Build Errors

```julia
# Error: Package failed to precompile
# Solution: Rebuild the package
Pkg.build("TextAssociations")

# Clear precompiled files
Base.compilecache(Base.identify_package("TextAssociations"))
```

#### Memory Issues During Installation

```julia
# For systems with limited RAM
ENV["JULIA_NUM_THREADS"] = 1
Pkg.add("TextAssociations")
```

### Getting Help

If you encounter installation issues:

1. Check the [GitHub Issues](https://github.com/yourusername/TextAssociations.jl/issues)
2. Ask on [Julia Discourse](https://discourse.julialang.org/)
3. Join the [Julia Slack](https://julialang.org/slack/)

## Docker Installation

For containerized deployments:

```dockerfile
FROM julia:1.10

WORKDIR /app

# Install TextAssociations
RUN julia -e 'using Pkg; Pkg.add("TextAssociations")'

# Copy your scripts
COPY . .

CMD ["julia", "your_script.jl"]
```

Build and run:

```bash
docker build -t textassoc-app .
docker run -it textassoc-app
```

## Next Steps

After successful installation:

- Continue to [Quick Tutorial](@ref getting_started_tutorial)
- See [Basic Examples](@ref getting_started_examples)
- Review [Core Concepts](@ref guide_concepts)
