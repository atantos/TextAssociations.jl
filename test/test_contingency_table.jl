using Test
using TextAssociations
using DataFrames

@testset "Contingency Table" begin
    input_string = "A test string with test repeated several times."
    node = "test"
    windowsize = 2
    minfreq = 1
    cont_table = ContingencyTable(input_string, node, windowsize, minfreq)

    # Test ContingencyTable fields
    @test typeof(cont_table.con_tbl) <: LazyProcess
    @test cont_table.node == node
    @test cont_table.windowsize == windowsize

    # Test extracting cached data
    result = extract_cached_data(cont_table.con_tbl)
    @test typeof(result) <: DataFrame
    @test !isempty(result)

    # Verify the structure of the contingency table
    @test "Collocate" in names(result)
    @test all(col -> all(x -> x isa Float64 || x isa Int, result[!, col]), [:a, :b, :c, :d])
end
