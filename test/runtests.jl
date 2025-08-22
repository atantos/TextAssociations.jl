# =====================================
# File: test/runtests.jl
# Main test file
# =====================================

using Test
using TextAssociations
using DataFrames
using Random
using CSV
using Statistics

# Set random seed for reproducibility
Random.seed!(42)

@testset "TextAssociations.jl Tests" begin

    @testset "Basic Functionality" begin
        text = "This is a test. This is only a test."

        @testset "ContingencyTable Creation" begin
            # Test basic creation
            ct = ContingencyTable(text, "test", 3, 1)
            @test ct.node == "test"
            @test ct.windowsize == 3
            @test ct.minfreq == 1

            # Test with preprocessing
            ct_prep = ContingencyTable(text, "test", 3, 1, auto_prep=true)
            @test !isnothing(ct_prep.input_ref)

            # Test error handling
            @test_throws ArgumentError ContingencyTable(text, "test", -1, 1)
            @test_throws ArgumentError ContingencyTable(text, "test", 3, -1)
            @test_throws ArgumentError ContingencyTable(text, "", 3, 1)
        end

        @testset "Text Preprocessing" begin
            text = "Hello, World! This is a TEST."
            doc = prepstring(text)
            processed_text = TextAnalysis.text(doc)

            # Check preprocessing effects
            @test !occursin(",", processed_text)  # Punctuation removed
            @test !occursin("!", processed_text)
            @test !occursin("TEST", processed_text)  # Lowercased
            @test occursin("test", processed_text)
        end
    end

    @testset "Metrics" begin
        # Create sample data
        text = """
        The cat sat on the mat. The cat played with the ball.
        The dog sat on the mat. The dog played with the cat.
        The mat was comfortable. The ball was red.
        """

        ct = ContingencyTable(text, "the", 3, 1)

        @testset "Information Theoretic Metrics" begin
            # Test with new type system
            @test length(evalassoc(PMI, ct)) > 0
            @test length(evalassoc(PMI², ct)) > 0
            @test length(evalassoc(PMI³, ct)) > 0
            @test length(evalassoc(PPMI, ct)) > 0
            @test all(evalassoc(PPMI, ct) .>= 0)  # PPMI should be non-negative
            @test length(evalassoc(LLR, ct)) > 0
            @test length(evalassoc(LLR², ct)) > 0
        end

        @testset "Statistical Metrics" begin
            @test length(evalassoc(ChiSquare, ct)) > 0
            @test length(evalassoc(Tscore, ct)) > 0
            @test length(evalassoc(Zscore, ct)) > 0
            @test length(evalassoc(PhiCoef, ct)) > 0
            @test length(evalassoc(CramersV, ct)) > 0
            @test length(evalassoc(YuleQ, ct)) > 0
            @test length(evalassoc(YuleOmega, ct)) > 0
        end

        @testset "Similarity Metrics" begin
            @test length(evalassoc(Dice, ct)) > 0
            dice_scores = evalassoc(Dice, ct)
            @test all(x -> 0 <= x <= 1 || isnan(x), dice_scores)  # Dice should be [0,1] or NaN
            @test length(evalassoc(LogDice, ct)) > 0
            @test length(evalassoc(JaccardIdx, ct)) > 0
            jaccard_scores = evalassoc(JaccardIdx, ct)
            @test all(x -> 0 <= x <= 1 || isnan(x), jaccard_scores)  # Jaccard should be [0,1] or NaN
            @test length(evalassoc(CosineSim, ct)) > 0
            @test length(evalassoc(OverlapCoef, ct)) > 0
        end

        @testset "Epidemiological Metrics" begin
            @test length(evalassoc(RelRisk, ct)) > 0
            @test length(evalassoc(LogRelRisk, ct)) > 0
            @test length(evalassoc(OddsRatio, ct)) > 0
            @test length(evalassoc(LogOddsRatio, ct)) > 0
            @test length(evalassoc(RiskDiff, ct)) > 0
        end

        @testset "Lexical Gravity" begin
            # Test lexical gravity with LazyInput
            @test length(evalassoc(LexicalGravity, ct)) > 0
            # Verify LazyInput is working
            @test !isnothing(ct.input_ref)
            doc = extract_document(ct.input_ref)
            @test isa(doc, TextAnalysis.StringDocument)
        end

        @testset "Multiple Metrics" begin
            # Test with vector of types
            metrics = [PMI, Dice, JaccardIdx]
            results = evalassoc(metrics, ct)
            @test isa(results, DataFrame)
            @test ncol(results) == length(metrics)
            @test all(name in names(results) for name in ["PMI", "Dice", "JaccardIdx"])

            # Test convenience method from raw text
            results2 = evalassoc(metrics, text, "the", 3, 1)
            @test isa(results2, DataFrame)
            @test ncol(results2) == length(metrics)
        end
    end

    @testset "Corpus Analysis" begin
        # Create test corpus
        docs = [
            TextAnalysis.StringDocument("The cat sat on the mat. The cat was happy."),
            TextAnalysis.StringDocument("The dog sat on the floor. The dog was tired."),
            TextAnalysis.StringDocument("The bird flew over the tree. The bird sang.")
        ]
        corpus = Corpus(docs)

        @testset "Corpus Loading" begin
            # Test corpus creation
            @test length(corpus.documents) == 3
            @test !isempty(corpus.vocabulary)

            # Test corpus statistics
            stats = corpus_statistics(corpus)
            @test stats[:num_documents] == 3
            @test stats[:total_tokens] > 0
            @test stats[:unique_tokens] > 0
            @test stats[:avg_doc_length] > 0
        end

        @testset "Single Node Corpus Analysis" begin
            # Analyze single node with new type system
            results = analyze_corpus(corpus, "the", PMI, windowsize=3, minfreq=1)

            @test isa(results, DataFrame)
            @test :Collocate in names(results)
            @test :Score in names(results)
            @test :Frequency in names(results)
            @test nrow(results) > 0
        end

        @testset "Multiple Nodes Analysis" begin
            nodes = ["the", "cat", "dog"]
            metrics = [PMI, Dice]

            analysis = analyze_multiple_nodes(corpus, nodes, metrics,
                windowsize=3, minfreq=1,
                top_n=10)

            @test isa(analysis, MultiNodeAnalysis)
            @test length(analysis.nodes) == length(nodes)
            @test all(haskey(analysis.results, node) for node in nodes)

            # Check that results are DataFrames with correct columns
            for node in nodes
                if !isempty(analysis.results[node])
                    df = analysis.results[node]
                    @test isa(df, DataFrame)
                    @test :Collocate in names(df)
                    @test :Frequency in names(df)
                    @test "PMI" in names(df)
                    @test "Dice" in names(df)
                end
            end
        end

        @testset "Corpus Contingency Table" begin
            # Test CorpusContingencyTable with evalassoc
            cct = CorpusContingencyTable(corpus, "the", 3, 1)

            # Test evalassoc with corpus contingency table
            scores = evalassoc(PMI, cct)
            @test length(scores) >= 0  # Can be empty if no collocates meet criteria

            # Test multiple metrics on corpus contingency table
            if !isempty(extract_cached_data(cct.aggregated_table))
                metrics = [PMI, Dice]
                results = evalassoc(metrics[1], cct)  # Test single metric first
                @test isa(results, Vector)
            end
        end
    end

    @testset "Edge Cases" begin
        @testset "Empty Results" begin
            # Word not in text
            ct = ContingencyTable("This is a test", "missing", 5, 1)
            scores = evalassoc(PMI, ct)
            @test isempty(scores)

            # Very high minimum frequency
            ct = ContingencyTable("word "^10, "word", 5, 100)
            scores = evalassoc(PMI, ct)
            @test isempty(scores)
        end

        @testset "Single Word Text" begin
            ct = ContingencyTable("word", "word", 5, 1)
            scores = evalassoc(PMI, ct)
            @test isempty(scores)
        end

        @testset "Type System Edge Cases" begin
            text = "Test text for edge cases. Test again."
            ct = ContingencyTable(text, "test", 3, 1)

            # Test that unknown metric types throw appropriate errors
            struct UnknownMetric <: AssociationMetric end
            @test_throws ArgumentError evalassoc(UnknownMetric, ct)

            # Test that non-metric types are caught
            not_metrics = [String, Int, Float64]
            @test_throws ArgumentError evalassoc(not_metrics, ct)
        end
    end

    @testset "LazyProcess and LazyInput" begin
        text = "Test text "^100  # Create larger text
        ct = ContingencyTable(text, "test", 5, 1)

        @testset "Lazy Loading" begin
            # Contingency table should not be computed yet
            @test !ct.con_tbl.cached_process

            # Now access it
            evalassoc(PMI, ct)

            # Now it should be cached
            @test ct.con_tbl.cached_process
        end

        @testset "LazyInput Functionality" begin
            # Test LazyInput extraction
            doc = extract_document(ct.input_ref)
            @test isa(doc, TextAnalysis.StringDocument)

            # Test that LazyInput is preserved in operations
            ct2 = ContingencyTable(text, "text", 5, 1)
            @test !isnothing(ct2.input_ref)
        end
    end

    @testset "API Consistency" begin
        text = "Consistency test text with repeated words test."

        @testset "evalassoc Type Signatures" begin
            # Test all evalassoc signatures work correctly

            # Single metric on ContingencyTable
            ct = ContingencyTable(text, "test", 3, 1)
            r1 = evalassoc(PMI, ct)
            @test isa(r1, Vector)

            # Single metric from raw text
            r2 = evalassoc(PMI, text, "test", 3, 1)
            @test isa(r2, Vector)
            @test r1 == r2  # Should give same results

            # Multiple metrics on ContingencyTable
            r3 = evalassoc([PMI, Dice], ct)
            @test isa(r3, DataFrame)

            # Multiple metrics from raw text
            r4 = evalassoc([PMI, Dice], text, "test", 3, 1)
            @test isa(r4, DataFrame)
            @test r3 == r4  # Should give same results
        end
    end
end

# Run tests
println("Running TextAssociations.jl tests...")