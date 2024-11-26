using Test
using TextAssociations

@testset "Association Metrics" begin
    # Create a mock ContingencyTable
    input_string = "This is a test string with some test data"
    node = "test"
    windowsize = 3
    minfreq = 1
    cont_table = ContingencyTable(input_string, node, windowsize, minfreq)

    # Test eval_pmi
    @test typeof(TextAssociations.eval_pmi(cont_table)) == Vector{Float64}
    @test all(x -> x >= 0, TextAssociations.eval_pmi(cont_table))  # PMI should be non-negative

    # Test eval_pmi²
    @test typeof(TextAssociations.eval_pmi²(cont_table)) == Vector{Float64}

    # Test evalassoc for a single metric
    result = evalassoc(PMI, cont_table)
    @test typeof(result) == Vector{Float64}

    # Test evalassoc for multiple metrics
    metrics = [PMI, PMI²]
    results_df = evalassoc(metrics, cont_table)
    @test size(results_df) == (length(result), 2)
end