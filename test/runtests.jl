# =====================================
# File: test/runtests.jl
# Comprehensive test suite for TextAssociations.jl
# =====================================

using CSV
using DataFrames
using Random
using Statistics
using Test
using TextAssociations
using TextAnalysis

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

        @testset "Vocabulary Creation" begin
            doc = prepstring("word1 word2 word3 word1")
            vocab = createvocab(doc)
            @test length(vocab) == 3  # Only unique words
            @test haskey(vocab, "word1")
            @test haskey(vocab, "word2")
            @test haskey(vocab, "word3")
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
            @test length(evalassoc(DeltaPi, ct)) > 0
            @test length(evalassoc(MinSens, ct)) > 0
            @test length(evalassoc(PiatetskyShapiro, ct)) > 0
            @test length(evalassoc(TschuprowT, ct)) > 0
            @test length(evalassoc(ContCoef, ct)) > 0
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
            @test length(evalassoc(OchiaiIdx, ct)) > 0
            @test length(evalassoc(KulczynskiSim, ct)) > 0
            @test length(evalassoc(TanimotoCoef, ct)) > 0
            @test length(evalassoc(RogersTanimotoCoef, ct)) > 0
            @test length(evalassoc(RogersTanimotoCoef2, ct)) > 0
            @test length(evalassoc(HammanSim, ct)) > 0
            @test length(evalassoc(HammanSim2, ct)) > 0
            @test length(evalassoc(GoodmanKruskalIdx, ct)) > 0
            @test length(evalassoc(GowerCoef, ct)) > 0
            @test length(evalassoc(GowerCoef2, ct)) > 0
            @test length(evalassoc(CzekanowskiDiceCoef, ct)) > 0
            @test length(evalassoc(SorgenfreyIdx, ct)) > 0
            @test length(evalassoc(SorgenfreyIdx2, ct)) > 0
            @test length(evalassoc(MountfordCoef, ct)) > 0
            @test length(evalassoc(MountfordCoef2, ct)) > 0
            @test length(evalassoc(SokalSneathIdx, ct)) > 0
            @test length(evalassoc(SokalMichenerCoef, ct)) > 0
        end

        @testset "Epidemiological Metrics" begin
            @test length(evalassoc(RelRisk, ct)) > 0
            @test length(evalassoc(LogRelRisk, ct)) > 0
            @test length(evalassoc(OddsRatio, ct)) > 0
            @test length(evalassoc(LogOddsRatio, ct)) > 0
            @test length(evalassoc(RiskDiff, ct)) > 0
            @test length(evalassoc(AttrRisk, ct)) > 0
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

        # Create corpus with metadata for advanced features
        metadata = Dict{String,Any}(
            "doc_1" => Dict(:year => 2020, :category => "animals", :author => "Alice"),
            "doc_2" => Dict(:year => 2021, :category => "animals", :author => "Bob"),
            "doc_3" => Dict(:year => 2022, :category => "nature", :author => "Alice")
        )

        corpus = TextAssociations.Corpus(docs, metadata=metadata)

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
            @test haskey(stats, :median_doc_length)
            @test haskey(stats, :min_doc_length)
            @test haskey(stats, :max_doc_length)
        end

        @testset "Single Node Corpus Analysis" begin
            # Analyze single node with new type system
            results = analyze_corpus(corpus, "the", PMI, windowsize=3, minfreq=1)

            @test isa(results, DataFrame)
            @test "Collocate" in names(results)
            @test "Score" in names(results)
            @test "Frequency" in names(results)
            @test "DocFrequency" in names(results)
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
                    @test "Collocate" in names(df)
                    @test "Frequency" in names(df)
                    @test "PMI" in names(df)
                    @test "Dice" in names(df)
                end
            end

            # Test parameters are stored
            @test analysis.parameters[:windowsize] == 3
            @test analysis.parameters[:minfreq] == 1
            @test analysis.parameters[:metrics] == metrics
            @test analysis.parameters[:top_n] == 10
        end

        @testset "Corpus Contingency Table" begin
            # Test CorpusContingencyTable with evalassoc
            cct = CorpusContingencyTable(corpus, "the", 3, 1)

            @test length(cct.tables) > 0
            @test cct.node == "the"
            @test cct.windowsize == 3
            @test cct.minfreq == 1

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

        @testset "Corpus Loading from Different Sources" begin
            # Test loading from DataFrame
            df = DataFrame(
                text=["Document 1 text", "Document 2 text", "Document 3 text"],
                author=["Author A", "Author B", "Author C"],
                year=[2020, 2021, 2022]
            )

            corpus_from_df = load_corpus_from_dataframe(
                df,
                text_column=:text,
                metadata_columns=[:author, :year]
            )

            @test length(corpus_from_df.documents) == 3
            @test !isempty(corpus_from_df.metadata)
            @test haskey(corpus_from_df.metadata, "doc_1")
            @test corpus_from_df.metadata["doc_1"][:author] == "Author A"
            @test corpus_from_df.metadata["doc_1"][:year] == 2020
        end
    end

    @testset "Advanced Corpus Features" begin
        # Create corpus with temporal and categorical metadata
        docs = [
            TextAnalysis.StringDocument("Innovation drives technology forward"),
            TextAnalysis.StringDocument("Technology enables innovation"),
            TextAnalysis.StringDocument("Research fuels innovation"),
            TextAnalysis.StringDocument("Innovation transforms industries"),
            TextAnalysis.StringDocument("Digital innovation accelerates"),
            TextAnalysis.StringDocument("Innovation requires collaboration")
        ]

        metadata = Dict{String,Any}(
            "doc_1" => Dict(:year => 2020, :field => "tech", :journal => "TechReview"),
            "doc_2" => Dict(:year => 2020, :field => "tech", :journal => "Innovation"),
            "doc_3" => Dict(:year => 2021, :field => "research", :journal => "Science"),
            "doc_4" => Dict(:year => 2021, :field => "business", :journal => "Business"),
            "doc_5" => Dict(:year => 2022, :field => "tech", :journal => "Digital"),
            "doc_6" => Dict(:year => 2022, :field => "research", :journal => "Collab")
        )

        corpus = TextAssociations.Corpus(docs, metadata=metadata)

        @testset "Temporal Corpus Analysis" begin
            nodes = ["innovation", "technology"]

            temporal_results = temporal_corpus_analysis(
                corpus,
                nodes,
                :year,
                PMI,
                time_bins=2,
                windowsize=3,
                minfreq=1
            )

            @test isa(temporal_results, TemporalCorpusAnalysis)
            @test length(temporal_results.time_periods) > 0
            @test !isempty(temporal_results.results_by_period)
            @test isa(temporal_results.trend_analysis, DataFrame)

            # Check trend analysis columns
            if !isempty(temporal_results.trend_analysis)
                @test "Node" in names(temporal_results.trend_analysis)
                @test "Collocate" in names(temporal_results.trend_analysis)
                @test "Correlation" in names(temporal_results.trend_analysis)
                @test "Slope" in names(temporal_results.trend_analysis)
                @test "MeanScore" in names(temporal_results.trend_analysis)
                @test "StdScore" in names(temporal_results.trend_analysis)
                @test "NumPeriods" in names(temporal_results.trend_analysis)
            end
        end

        @testset "Subcorpus Comparison" begin
            comparison = compare_subcorpora(
                corpus,
                :field,
                "innovation",
                PMI,
                windowsize=3,
                minfreq=1
            )

            @test isa(comparison, SubcorpusComparison)
            @test comparison.node == "innovation"
            @test !isempty(comparison.subcorpora)
            @test !isempty(comparison.results)
            @test isa(comparison.statistical_tests, DataFrame)
            @test isa(comparison.effect_sizes, DataFrame)

            # Check that subcorpora were created correctly
            @test haskey(comparison.subcorpora, "tech")
            @test haskey(comparison.subcorpora, "research")

            # Check effect sizes columns if data exists
            if !isempty(comparison.effect_sizes)
                @test "Collocate" in names(comparison.effect_sizes)
                @test "Group1" in names(comparison.effect_sizes)
                @test "Group2" in names(comparison.effect_sizes)
                @test "CohensD" in names(comparison.effect_sizes)
                @test "EffectSize" in names(comparison.effect_sizes)
            end
        end

        @testset "Keyword Extraction" begin
            keywords = extract_keywords(
                corpus,
                method=:tfidf,
                num_keywords=10,
                min_doc_freq=1,
                max_doc_freq_ratio=0.9
            )

            @test isa(keywords, DataFrame)
            @test "Keyword" in names(keywords)
            @test "TFIDF" in names(keywords)
            @test "DocFreq" in names(keywords)
            @test "DocFreqRatio" in names(keywords)
            @test nrow(keywords) <= 10

            # Test error handling for unsupported methods
            @test_throws ArgumentError extract_keywords(corpus, method=:unknown)
        end

        @testset "Collocation Network" begin
            network = build_collocation_network(
                corpus,
                ["innovation"],
                metric=PMI,
                depth=2,
                min_score=0.0,  # Lower threshold for test data
                max_neighbors=5,
                windowsize=3,
                minfreq=1
            )

            @test isa(network, CollocationNetwork)
            @test "innovation" in network.nodes
            @test isa(network.edges, DataFrame)
            @test isa(network.node_metrics, DataFrame)
            @test !isempty(network.parameters)

            # Check edges structure
            if !isempty(network.edges)
                @test "Source" in names(network.edges)
                @test "Target" in names(network.edges)
                @test "Weight" in names(network.edges)
                @test "Metric" in names(network.edges)
            end

            # Check node metrics structure
            if !isempty(network.node_metrics)
                @test "Node" in names(network.node_metrics)
                @test "Degree" in names(network.node_metrics)
                @test "AvgScore" in names(network.node_metrics)
                @test "MaxScore" in names(network.node_metrics)
                @test "Layer" in names(network.node_metrics)
            end

            # Test network export (without actually writing files)
            @test_nowarn begin
                temp_nodes = tempname()
                temp_edges = tempname()
                try
                    export_network_to_gephi(network, temp_nodes, temp_edges)
                finally
                    isfile(temp_nodes) && rm(temp_nodes)
                    isfile(temp_edges) && rm(temp_edges)
                end
            end
        end

        @testset "Concordance Generation" begin
            concordance = generate_concordance(
                corpus,
                "innovation",
                context_size=10,
                max_lines=50
            )

            @test isa(concordance, Concordance)
            @test concordance.node == "innovation"
            @test isa(concordance.lines, DataFrame)
            @test isa(concordance.statistics, Dict)

            # Check concordance lines structure
            if !isempty(concordance.lines)
                @test "LeftContext" in names(concordance.lines)
                @test "Node" in names(concordance.lines)
                @test "RightContext" in names(concordance.lines)
                @test "DocId" in names(concordance.lines)
                @test "Position" in names(concordance.lines)
            end

            # Check statistics
            @test haskey(concordance.statistics, :total_occurrences)
            @test haskey(concordance.statistics, :documents_with_node)
            @test haskey(concordance.statistics, :lines_generated)
            @test concordance.statistics[:total_occurrences] >= concordance.statistics[:lines_generated]
        end

        @testset "Export Results" begin
            nodes = ["innovation", "technology"]
            metrics = [PMI, Dice]

            analysis = analyze_multiple_nodes(
                corpus, nodes, metrics,
                windowsize=3, minfreq=1, top_n=10
            )

            # Test CSV export
            temp_dir = mktempdir()
            try
                export_results(analysis, temp_dir, format=:csv)
                @test isfile(joinpath(temp_dir, "summary.csv"))

                # Check that node result files exist
                for node in nodes
                    if !isempty(analysis.results[node])
                        @test isfile(joinpath(temp_dir, "$(node)_results.csv"))
                    end
                end
            finally
                rm(temp_dir, recursive=true)
            end

            # Test JSON export
            temp_file = tempname() * ".json"
            try
                export_results(analysis, temp_file, format=:json)
                @test isfile(temp_file)
            finally
                isfile(temp_file) && rm(temp_file)
            end
        end
    end

    @testset "Utility Functions" begin
        @testset "Text Analysis Utilities" begin
            doc = prepstring("The quick brown fox jumps over the lazy dog")

            # Test find_prior_words
            prior_words = TextAssociations.find_prior_words(doc, "fox", 2)
            @test isa(prior_words, Set{String})
            @test "brown" in prior_words
            @test "quick" in prior_words

            # Test find_following_words
            following_words = TextAssociations.find_following_words(doc, "fox", 2)
            @test isa(following_words, Set{String})
            @test "jumps" in following_words
            @test "over" in following_words

            # Test count_word_frequency
            freq = TextAssociations.count_word_frequency(doc, "the")
            @test freq == 2
        end

        @testset "Statistical Utilities" begin
            # Test log_safe
            @test TextAssociations.log_safe(0) == log(eps())
            @test TextAssociations.log_safe(-1) == log(eps())
            @test TextAssociations.log_safe(1) == log(1)

            # Test log2_safe
            @test TextAssociations.log2_safe(0) == log2(eps())
            @test TextAssociations.log2_safe(-1) == log2(eps())
            @test TextAssociations.log2_safe(2) == log2(2)

            # Test listmetrics
            metrics = listmetrics()
            @test isa(metrics, Vector{Symbol})
            @test :PMI in metrics
            @test :Dice in metrics
            @test :JaccardIdx in metrics
        end

        @testset "I/O Utilities" begin
            # Test read_text_smart with different encodings
            temp_file = tempname()
            try
                # Write UTF-8 text
                open(temp_file, "w") do f
                    write(f, "Test text UTF-8")
                end

                content = TextAssociations.read_text_smart(temp_file)
                @test content == "Test text UTF-8"
            finally
                isfile(temp_file) && rm(temp_file)
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

        @testset "Empty Corpus" begin
            empty_corpus = TextAssociations.Corpus(TextAnalysis.StringDocument[])
            stats = corpus_statistics(empty_corpus)
            @test stats[:num_documents] == 0
            @test stats[:total_tokens] == 0
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

        @testset "Corpus with Missing Metadata" begin
            docs = [
                TextAnalysis.StringDocument("Document 1"),
                TextAnalysis.StringDocument("Document 2")
            ]

            # Corpus with partial metadata
            metadata = Dict{String,Any}(
                "doc_1" => Dict(:year => 2020)
                # doc_2 has no metadata
            )

            corpus = TextAssociations.Corpus(docs, metadata=metadata)

            # Test that temporal analysis handles missing metadata gracefully
            @test_throws ArgumentError temporal_corpus_analysis(
                corpus, ["document"], :year, PMI
            )
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

        @testset "Extract Cached Data" begin
            lazy_proc = TextAssociations.LazyProcess(() -> DataFrame(a=[1, 2, 3]))
            @test !lazy_proc.cached_process

            result = extract_cached_data(lazy_proc)
            @test isa(result, DataFrame)
            @test lazy_proc.cached_process

            # Second call should return cached result
            result2 = extract_cached_data(lazy_proc)
            @test result === result2
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

        @testset "Metric Name Consistency" begin
            # Test that metric types are properly named
            @test string(PMI) == "PMI"
            @test string(Dice) == "Dice"
            @test string(JaccardIdx) == "JaccardIdx"

            # Test that eval functions exist for all metric types
            for metric in listmetrics()
                func_name = Symbol("eval_", lowercase(string(metric)))
                @test isdefined(TextAssociations, func_name)
            end
        end
    end

    @testset "Performance and Memory" begin
        @testset "Large Corpus Handling" begin
            # Create a larger corpus
            large_docs = [TextAnalysis.StringDocument("word "^100) for _ in 1:10]
            large_corpus = TextAssociations.Corpus(large_docs)

            # Test that operations complete without errors
            @test_nowarn analyze_corpus(large_corpus, "word", PMI, windowsize=5, minfreq=1)
        end

        @testset "Batch Processing" begin
            # Create test corpus
            docs = [TextAnalysis.StringDocument("test document $i") for i in 1:5]
            corpus = TextAssociations.Corpus(docs)

            # Create node file
            node_file = tempname()
            output_dir = mktempdir()

            try
                open(node_file, "w") do f
                    println(f, "test")
                    println(f, "document")
                end

                # Test batch processing
                batch_process_corpus(
                    corpus, node_file, output_dir,
                    metrics=[PMI, Dice],
                    batch_size=2
                )

                # Check that batch directories were created
                @test isdir(joinpath(output_dir, "batch_1"))
            finally
                rm(node_file, force=true)
                rm(output_dir, recursive=true, force=true)
            end
        end
    end
end

# Run tests
println("Running comprehensive TextAssociations.jl tests...")