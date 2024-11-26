using Test
using TextAssociations
using TextAnalysis
using DataStructures

@testset "Utils Functions" begin
    input_string = "This is a test string with repeated test words"
    string_document = prepstring(input_string)

    # Test prepstring
    @test typeof(string_document) == TextAnalysis.StringDocument{String}
    @test !occursin(r"\s{2,}", string_document.text)  # Ensure no whitespace

    # Test createvocab
    vocab = createvocab(string_document)
    @test typeof(vocab) == OrderedDict{String,Int}
    @test length(vocab) > 0
end

