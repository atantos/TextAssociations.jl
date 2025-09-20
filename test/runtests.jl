# =====================================
# File: test/runtests.jl
# Comprehensive test suite for TextAssociations.jl (Updated)
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

            # Test with preprocessing and accent stripping
            ct_prep = ContingencyTable(text, "test", 3, 1, auto_prep=true, strip_accents=false)
            @test !isnothing(ct_prep.input_ref)

            # Test with accent stripping enabled
            text_accents = "Ένα τεστ με τόνους"
            ct_accents = ContingencyTable(text_accents, "τεστ", 3, 1, strip_accents=true)
            @test ct_accents.node == "τεστ"

            # Test error handling
            @test_throws ArgumentError ContingencyTable(text, "test", -1, 1)
            @test_throws ArgumentError ContingencyTable(text, "test", 3, -1)
            @test_throws ArgumentError ContingencyTable(text, "", 3, 1)
        end

        @testset "Text Preprocessing" begin
            text = "Hello, World! This is a TEST."

            # Test without accent stripping (default)
            doc = prepstring(text, strip_accents=false)
            processed_text = TextAnalysis.text(doc)
            @test !occursin(",", processed_text)  # Punctuation removed
            @test !occursin("!", processed_text)
            @test !occursin("TEST", processed_text)  # Lowercased
            @test occursin("test", processed_text)

            # Test with accent stripping
            text_greek = "Ένα τεστ με τόνους και διαλυτικά"
            doc_stripped = prepstring(text_greek, strip_accents=true)
            @test occursin("ενα", TextAnalysis.text(doc_stripped))
            @test !occursin("ένα", TextAnalysis.text(doc_stripped))
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

    @testset "New API with AssociationDataFormat" begin
        text = """
        The cat sat on the mat. The cat played with the ball.
        The dog sat on the mat. The dog played with the cat.
        The mat was comfortable. The ball was red.
        """

        ct = ContingencyTable(text, "the", 3, 1)

        @testset "DataFrame vs Scores-only Output" begin
            # Test DataFrame output (default)
            result_df = evalassoc(PMI, ct, scores_only=false)
            @test isa(result_df, DataFrame)
            @test "Node" in names(result_df)
            @test "Collocate" in names(result_df)
            @test "Frequency" in names(result_df)
            @test "PMI" in names(result_df)
            @test all(result_df.Node .== "the")

            # Test scores-only output
            result_scores = evalassoc(PMI, ct, scores_only=true)
            @test isa(result_scores, Vector{Float64})
            @test length(result_scores) == nrow(result_df)
        end

        @testset "Multiple Metrics with New API" begin
            metrics = [PMI, Dice, JaccardIdx]

            # Test DataFrame output for multiple metrics
            result_df = evalassoc(metrics, ct, scores_only=false)
            @test isa(result_df, DataFrame)
            @test "Node" in names(result_df)
            @test "Collocate" in names(result_df)
            @test "Frequency" in names(result_df)
            @test "PMI" in names(result_df)
            @test "Dice" in names(result_df)
            @test "JaccardIdx" in names(result_df)

            # Test scores-only output for multiple metrics
            result_dict = evalassoc(metrics, ct, scores_only=true)
            @test isa(result_dict, Dict{String,Vector{Float64}})
            @test haskey(result_dict, "PMI")
            @test haskey(result_dict, "Dice")
            @test haskey(result_dict, "JaccardIdx")
        end
    end

    @testset "Metrics" begin
        text = """
        The cat sat on the mat. The cat played with the ball.
        The dog sat on the mat. The dog played with the cat.
        The mat was comfortable. The ball was red.
        """

        ct = ContingencyTable(text, "the", 3, 1)

        @testset "Information Theoretic Metrics" begin
            # Test all return DataFrames by default
            @test isa(evalassoc(PMI, ct), DataFrame)
            @test isa(evalassoc(PMI², ct), DataFrame)
            @test isa(evalassoc(PMI³, ct), DataFrame)

            ppmi_result = evalassoc(PPMI, ct)
            @test isa(ppmi_result, DataFrame)
            if nrow(ppmi_result) > 0
                @test all(ppmi_result.PPMI .>= 0)  # PPMI should be non-negative
            end

            @test isa(evalassoc(LLR, ct), DataFrame)
            @test isa(evalassoc(LLR², ct), DataFrame)
        end

        @testset "Statistical Metrics" begin
            @test isa(evalassoc(ChiSquare, ct), DataFrame)
            @test isa(evalassoc(Tscore, ct), DataFrame)
            @test isa(evalassoc(Zscore, ct), DataFrame)
            @test isa(evalassoc(PhiCoef, ct), DataFrame)
            @test isa(evalassoc(CramersV, ct), DataFrame)
            @test isa(evalassoc(YuleQ, ct), DataFrame)
            @test isa(evalassoc(YuleOmega, ct), DataFrame)
            @test isa(evalassoc(DeltaPi, ct), DataFrame)
            @test isa(evalassoc(MinSens, ct), DataFrame)
            @test isa(evalassoc(PiatetskyShapiro, ct), DataFrame)
            @test isa(evalassoc(TschuprowT, ct), DataFrame)
            @test isa(evalassoc(ContCoef, ct), DataFrame)
        end

        @testset "Similarity Metrics" begin
            dice_result = evalassoc(Dice, ct)
            @test isa(dice_result, DataFrame)
            if nrow(dice_result) > 0
                dice_scores = dice_result.Dice
                @test all(x -> 0 <= x <= 1 || isnan(x), dice_scores)
            end

            @test isa(evalassoc(LogDice, ct), DataFrame)

            jaccard_result = evalassoc(JaccardIdx, ct)
            @test isa(jaccard_result, DataFrame)
            if nrow(jaccard_result) > 0
                jaccard_scores = jaccard_result.JaccardIdx
                @test all(x -> 0 <= x <= 1 || isnan(x), jaccard_scores)
            end

            @test isa(evalassoc(CosineSim, ct), DataFrame)
            @test isa(evalassoc(OverlapCoef, ct), DataFrame)
            @test isa(evalassoc(OchiaiIdx, ct), DataFrame)
            @test isa(evalassoc(KulczynskiSim, ct), DataFrame)
            @test isa(evalassoc(TanimotoCoef, ct), DataFrame)
            @test isa(evalassoc(RogersTanimotoCoef, ct), DataFrame)
            @test isa(evalassoc(RogersTanimotoCoef2, ct), DataFrame)
            @test isa(evalassoc(HammanSim, ct), DataFrame)
            @test isa(evalassoc(HammanSim2, ct), DataFrame)
            @test isa(evalassoc(GoodmanKruskalIdx, ct), DataFrame)
            @test isa(evalassoc(GowerCoef, ct), DataFrame)
            @test isa(evalassoc(GowerCoef2, ct), DataFrame)
            @test isa(evalassoc(CzekanowskiDiceCoef, ct), DataFrame)
            @test isa(evalassoc(SorgenfreyIdx, ct), DataFrame)
            @test isa(evalassoc(SorgenfreyIdx2, ct), DataFrame)
            @test isa(evalassoc(MountfordCoef, ct), DataFrame)
            @test isa(evalassoc(MountfordCoef2, ct), DataFrame)
            @test isa(evalassoc(SokalSneathIdx, ct), DataFrame)
            @test isa(evalassoc(SokalMichenerCoef, ct), DataFrame)
        end

        @testset "Epidemiological Metrics" begin
            @test isa(evalassoc(RelRisk, ct), DataFrame)
            @test isa(evalassoc(LogRelRisk, ct), DataFrame)
            @test isa(evalassoc(OddsRatio, ct), DataFrame)
            @test isa(evalassoc(LogOddsRatio, ct), DataFrame)
            @test isa(evalassoc(RiskDiff, ct), DataFrame)
            @test isa(evalassoc(AttrRisk, ct), DataFrame)
        end

        @testset "Lexical Gravity" begin
            # Test lexical gravity with different formulas
            lg_result = evalassoc(LexicalGravity, ct)
            @test isa(lg_result, DataFrame)

            # Test scores-only mode
            lg_scores = evalassoc(LexicalGravity, ct, scores_only=true)
            @test isa(lg_scores, Vector{Float64})

            # Verify LazyInput is working
            @test !isnothing(ct.input_ref)
            doc = extract_document(ct.input_ref)
            @test isa(doc, TextAnalysis.StringDocument)
        end

        @testset "Multiple Metrics with Different Output Formats" begin
            metrics = [PMI, Dice, JaccardIdx]

            # Test DataFrame output (default)
            results_df = evalassoc(metrics, ct)
            @test isa(results_df, DataFrame)
            @test all(name in names(results_df) for name in ["Node", "Collocate", "Frequency", "PMI", "Dice", "JaccardIdx"])

            # Test scores-only output
            results_dict = evalassoc(metrics, ct, scores_only=true)
            @test isa(results_dict, Dict{String,Vector{Float64}})
            @test all(haskey(results_dict, string(m)) for m in metrics)

            # Test convenience method from raw text
            results2 = evalassoc(metrics, text, "the", 3, 1)
            @test isa(results2, DataFrame)
        end
    end

    @testset "Corpus Analysis" begin
        # Create test corpus
        docs = [
            TextAnalysis.StringDocument("The cat sat on the mat. The cat was happy."),
            TextAnalysis.StringDocument("The dog sat on the floor. The dog was tired."),
            TextAnalysis.StringDocument("The bird flew over the tree. The bird sang.")
        ]

        metadata = Dict{String,Any}(
            "doc_1" => Dict(:year => 2020, :category => "animals", :author => "Alice"),
            "doc_2" => Dict(:year => 2021, :category => "animals", :author => "Bob"),
            "doc_3" => Dict(:year => 2022, :category => "nature", :author => "Alice")
        )

        corpus = TextAssociations.Corpus(docs, metadata=metadata)

        @testset "Corpus Loading" begin
            @test length(corpus.documents) == 3
            @test !isempty(corpus.vocabulary)

            # Test corpus statistics with accent stripping options
            stats = corpus_statistics(corpus, unicode_form=:NFC, strip_accents=false)
            @test stats[:num_documents] == 3
            @test stats[:total_tokens] > 0
            @test stats[:unique_tokens] > 0
            @test stats[:vocabulary_size] > 0
            @test stats[:avg_doc_length] > 0
        end

        @testset "Single Node Corpus Analysis with Updated API" begin
            # Results now include Node column by default
            results = analyze_corpus(corpus, "the", PMI, windowsize=3, minfreq=1)

            @test isa(results, DataFrame)
            @test "Node" in names(results)
            @test "Collocate" in names(results)
            @test "Score" in names(results)
            @test "Frequency" in names(results)
            @test "DocFrequency" in names(results)
            @test all(results.Node .== "the")
        end

        @testset "Multiple Nodes Analysis with Node Column" begin
            nodes = ["the", "cat", "dog"]
            metrics = [PMI, Dice]

            analysis = analyze_multiple_nodes(corpus, nodes, metrics,
                windowsize=3, minfreq=1, top_n=10)

            @test isa(analysis, MultiNodeAnalysis)
            @test length(analysis.nodes) == length(nodes)

            # Check that each result has Node column
            for node in nodes
                if haskey(analysis.results, node) && !isempty(analysis.results[node])
                    df = analysis.results[node]
                    @test isa(df, DataFrame)
                    @test "Node" in names(df)
                    @test all(df.Node .== node)
                    @test "Collocate" in names(df)
                    @test "Frequency" in names(df)
                    @test "PMI" in names(df)
                    @test "Dice" in names(df)
                end
            end
        end

        @testset "CorpusContingencyTable with New API" begin
            # Test with accent stripping option
            cct = CorpusContingencyTable(corpus, "the", 3, 1, strip_accents=false)

            @test length(cct.tables) > 0
            @test cct.node == "the"
            @test cct.windowsize == 3
            @test cct.minfreq == 1

            # Test evalassoc with CorpusContingencyTable
            # Should return DataFrame by default
            result = evalassoc(PMI, cct)
            @test isa(result, DataFrame)
            if nrow(result) > 0
                @test "Node" in names(result)
                @test "Collocate" in names(result)
                @test "PMI" in names(result)
            end

            # Test scores-only mode
            scores = evalassoc(PMI, cct, scores_only=true)
            @test isa(scores, Vector{Float64})

            # Test multiple metrics
            multi_result = evalassoc([PMI, Dice], cct)
            @test isa(multi_result, DataFrame)
        end

        @testset "Corpus Loading from Different Sources" begin
            df = DataFrame(
                text=["Document 1 text", "Document 2 text", "Document 3 text"],
                author=["Author A", "Author B", "Author C"],
                year=[2020, 2021, 2022]
            )

            corpus_from_df = load_corpus_df(
                df,
                text_column=:text,
                metadata_columns=[:author, :year],
                preprocess=true
            )

            @test length(corpus_from_df.documents) == 3
            @test !isempty(corpus_from_df.metadata)
            @test haskey(corpus_from_df.metadata, "doc_1")
        end
    end

    @testset "Advanced Corpus Features" begin
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

        @testset "Temporal Analysis" begin
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
            if nrow(keywords) > 0
                @test "Keyword" in names(keywords)
                @test "TFIDF" in names(keywords)
                @test "DocFreq" in names(keywords)
                @test "DocFreqRatio" in names(keywords)
            end

            @test_throws ArgumentError extract_keywords(corpus, method=:unknown)
        end

        @testset "Collocation Network" begin
            network = build_collocation_network(
                corpus,
                ["innovation"],
                metric=PMI,
                depth=2,
                min_score=0.0,
                max_neighbors=5,
                windowsize=3,
                minfreq=1
            )

            @test isa(network, CollocationNetwork)
            @test "innovation" in network.nodes
            @test isa(network.edges, DataFrame)
            @test isa(network.node_metrics, DataFrame)

            # Test export without actually writing files
            temp_nodes = tempname()
            temp_edges = tempname()
            try
                export_network_to_gephi(network, temp_nodes, temp_edges)
                @test isfile(temp_nodes)
                @test isfile(temp_edges)
            finally
                isfile(temp_nodes) && rm(temp_nodes)
                isfile(temp_edges) && rm(temp_edges)
            end
        end

        @testset "Concordance" begin
            concordance = concord(
                corpus,
                "innovation",
                context_size=10,
                max_lines=50
            )

            @test isa(concordance, Concordance)
            @test concordance.node == "innovation"
            @test isa(concordance.lines, DataFrame)
            @test isa(concordance.statistics, Dict)
        end
    end

    @testset "AssociationDataFormat Interface" begin
        text = "Test text for interface testing. Test again."
        ct = ContingencyTable(text, "test", 3, 1)

        @testset "Accessor Functions" begin
            # Test accessor functions work for ContingencyTable
            df = TextAssociations.assoc_df(ct)
            @test isa(df, DataFrame)

            node = TextAssociations.assoc_node(ct)
            @test node == "test"

            ws = TextAssociations.assoc_ws(ct)
            @test ws == 3

            tokens = TextAssociations.assoc_tokens(ct)
            @test isa(tokens, Vector{String})
        end

        @testset "CorpusContingencyTable Accessors" begin
            docs = [TextAnalysis.StringDocument("test document")]
            corpus = TextAssociations.Corpus(docs)
            cct = CorpusContingencyTable(corpus, "test", 3, 1)

            df = TextAssociations.assoc_df(cct)
            @test isa(df, DataFrame)

            node = TextAssociations.assoc_node(cct)
            @test node == "test"

            ws = TextAssociations.assoc_ws(cct)
            @test ws == 3

            # CCT returns nothing for tokens by default
            tokens = TextAssociations.assoc_tokens(cct)
            @test isnothing(tokens)
        end
    end

    @testset "Utility Functions" begin
        @testset "Text Analysis Utilities" begin
            doc = prepstring("The quick brown fox jumps over the lazy dog")

            prior_words = TextAssociations.find_prior_words(doc, "fox", 2)
            @test isa(prior_words, Set{String})
            @test "brown" in prior_words

            following_words = TextAssociations.find_following_words(doc, "fox", 2)
            @test isa(following_words, Set{String})
            @test "jumps" in following_words

            freq = TextAssociations.count_word_frequency(doc, "the")
            @test freq == 2
        end

        @testset "Statistical Utilities" begin
            @test TextAssociations.log_safe(0) == log(eps())
            @test TextAssociations.log_safe(-1) == log(eps())
            @test TextAssociations.log_safe(1) == log(1)

            @test TextAssociations.log2_safe(0) == log2(eps())
            @test TextAssociations.log2_safe(-1) == log2(eps())
            @test TextAssociations.log2_safe(2) == log2(2)

            metrics = listmetrics()
            @test isa(metrics, Vector{Symbol})
            @test :PMI in metrics
            @test :Dice in metrics
            @test :LexicalGravity in metrics
        end
    end

    @testset "Edge Cases and Error Handling" begin
        @testset "Empty Results" begin
            # Word not in text
            ct = ContingencyTable("This is a test", "missing", 5, 1)
            result = evalassoc(PMI, ct)
            @test isa(result, DataFrame)
            @test nrow(result) == 0

            # Scores-only mode with empty results
            scores = evalassoc(PMI, ct, scores_only=true)
            @test isempty(scores)

            # Very high minimum frequency
            ct = ContingencyTable("word "^10, "word", 5, 100)
            result = evalassoc(PMI, ct)
            @test isa(result, DataFrame)
            @test nrow(result) == 0
        end

        @testset "Single Word Text" begin
            ct = ContingencyTable("word", "word", 5, 1)
            result = evalassoc(PMI, ct)
            @test isa(result, DataFrame)
            @test nrow(result) == 0
        end

        @testset "Empty Corpus" begin
            empty_corpus = TextAssociations.Corpus(TextAnalysis.StringDocument[])
            stats = corpus_statistics(empty_corpus)
            @test stats[:num_documents] == 0
            @test stats[:total_tokens] == 0
        end

        @testset "Unknown Metrics" begin
            text = "Test text for edge cases."
            ct = ContingencyTable(text, "test", 3, 1)

            # Test that unknown metric types throw appropriate errors
            struct UnknownMetric <: AssociationMetric end
            @test_throws ArgumentError evalassoc(UnknownMetric, ct)
        end
    end

    @testset "LazyProcess and LazyInput" begin
        text = "Test text "^100
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
            doc = extract_document(ct.input_ref)
            @test isa(doc, TextAnalysis.StringDocument)

            ct2 = ContingencyTable(text, "text", 5, 1)
            @test !isnothing(ct2.input_ref)
        end

        @testset "Extract Cached Data" begin
            lazy_proc = TextAssociations.LazyProcess(() -> DataFrame(a=[1, 2, 3]))
            @test !lazy_proc.cached_process

            result = extract_cached_data(lazy_proc)
            @test isa(result, DataFrame)
            @test lazy_proc.cached_process

            result2 = extract_cached_data(lazy_proc)
            @test result === result2
        end
    end

    @testset "Unicode and Accent Handling" begin
        @testset "Greek Text Processing" begin
            # Test with Greek text containing tonos
            greek_text = "Το ελληνικό κείμενο με τόνους και διαλυτικά"

            # Test without accent stripping (preserve tonos)
            ct_with_tonos = ContingencyTable(greek_text, "με", 3, 1, strip_accents=false)
            result_with = evalassoc(PMI, ct_with_tonos)
            @test isa(result_with, DataFrame)

            # Test with accent stripping
            ct_no_tonos = ContingencyTable(greek_text, "με", 3, 1, strip_accents=true)
            result_without = evalassoc(PMI, ct_no_tonos)
            @test isa(result_without, DataFrame)
        end

        @testset "Unicode Normalization" begin
            text = "Café naïve résumé"

            # Test different normalization forms
            doc_nfc = prepstring(text, unicode_form=:NFC)
            doc_nfd = prepstring(text, unicode_form=:NFD)

            @test isa(doc_nfc, TextAnalysis.StringDocument)
            @test isa(doc_nfd, TextAnalysis.StringDocument)

            # Test strip_diacritics function
            stripped = TextAssociations.strip_diacritics("café")
            @test stripped == "cafe"
        end
    end

    @testset "Export and Batch Processing" begin
        docs = [TextAnalysis.StringDocument("test document $i") for i in 1:5]
        corpus = TextAssociations.Corpus(docs)

        @testset "Export Results with Node Column" begin
            nodes = ["test", "document"]
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
                @test isfile(joinpath(temp_dir, "all_results_combined.csv"))

                # Check combined results has Node column
                combined = CSV.read(joinpath(temp_dir, "all_results_combined.csv"), DataFrame)
                @test "Node" in names(combined)
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

        @testset "Batch Processing" begin
            node_file = tempname()
            output_dir = mktempdir()

            try
                open(node_file, "w") do f
                    println(f, "test")
                    println(f, "document")
                end

                batch_process_corpus(
                    corpus, node_file, output_dir,
                    metrics=[PMI, Dice],
                    batch_size=2
                )

                @test isdir(joinpath(output_dir, "batch_1"))
                @test isfile(joinpath(output_dir, "all_batches_combined.csv"))

                # Check combined results include Node column
                if isfile(joinpath(output_dir, "all_batches_combined.csv"))
                    combined = CSV.read(joinpath(output_dir, "all_batches_combined.csv"), DataFrame)
                    @test "Node" in names(combined)
                end
            finally
                rm(node_file, force=true)
                rm(output_dir, recursive=true, force=true)
            end
        end
    end

    @testset "Performance Options" begin
        text = "Test text "^50
        ct = ContingencyTable(text, "test", 3, 1)

        @testset "Scores-only Performance Mode" begin
            # Single metric performance mode
            scores = evalassoc(PMI, ct, scores_only=true)
            @test isa(scores, Vector{Float64})

            # Compare with full DataFrame output
            df = evalassoc(PMI, ct, scores_only=false)
            @test length(scores) == nrow(df)
            if nrow(df) > 0
                @test scores == df.PMI
            end

            # Multiple metrics performance mode
            metrics = [PMI, Dice, JaccardIdx]
            scores_dict = evalassoc(metrics, ct, scores_only=true)
            @test isa(scores_dict, Dict{String,Vector{Float64}})

            df_multi = evalassoc(metrics, ct, scores_only=false)
            for m in metrics
                metric_name = string(m)
                @test scores_dict[metric_name] == df_multi[!, Symbol(metric_name)]
            end
        end

        @testset "Large Corpus Performance" begin
            # Create larger corpus
            large_docs = [TextAnalysis.StringDocument("word "^100) for _ in 1:10]
            large_corpus = TextAssociations.Corpus(large_docs)

            # Test scores-only mode saves memory
            cct = CorpusContingencyTable(large_corpus, "word", 5, 1)

            # Scores only - more efficient
            scores = evalassoc(PMI, cct, scores_only=true)
            @test isa(scores, Vector{Float64})

            # DataFrame output - includes metadata
            df = evalassoc(PMI, cct, scores_only=false)
            @test isa(df, DataFrame)
            @test "Node" in names(df)
        end
    end

    @testset "API Consistency and Compatibility" begin
        text = "Consistency test text with repeated words test."

        @testset "All evalassoc Signatures" begin
            ct = ContingencyTable(text, "test", 3, 1)

            # Single metric, ContingencyTable, default (DataFrame)
            r1 = evalassoc(PMI, ct)
            @test isa(r1, DataFrame)

            # Single metric, ContingencyTable, scores only
            r2 = evalassoc(PMI, ct, scores_only=true)
            @test isa(r2, Vector{Float64})

            # Single metric from raw text
            r3 = evalassoc(PMI, text, "test", 3, 1)
            @test isa(r3, DataFrame)

            # Single metric from raw text with scores_only
            r4 = evalassoc(PMI, text, "test", 3, 1, scores_only=true)
            @test isa(r4, Vector{Float64})

            # Multiple metrics, default output
            r5 = evalassoc([PMI, Dice], ct)
            @test isa(r5, DataFrame)

            # Multiple metrics, scores_only
            r6 = evalassoc([PMI, Dice], ct, scores_only=true)
            @test isa(r6, Dict{String,Vector{Float64}})

            # Vector{DataType} compatibility
            r7 = evalassoc(Vector{DataType}([PMI, Dice]), ct)
            @test isa(r7, DataFrame)
        end

        @testset "Corpus-level API" begin
            docs = [TextAnalysis.StringDocument("test document")]
            corpus = TextAssociations.Corpus(docs)
            cct = CorpusContingencyTable(corpus, "test", 3, 1)

            # Single metric on CCT
            r1 = evalassoc(PMI, cct)
            @test isa(r1, DataFrame)

            # Single metric on CCT, scores_only
            r2 = evalassoc(PMI, cct, scores_only=true)
            @test isa(r2, Vector{Float64})

            # Multiple metrics on CCT
            r3 = evalassoc([PMI, Dice], cct)
            @test isa(r3, DataFrame)

            # Multiple metrics on CCT, scores_only
            r4 = evalassoc([PMI, Dice], cct, scores_only=true)
            @test isa(r4, Dict{String,Vector{Float64}})
        end
    end

    @testset "Coverage Summary and Statistics" begin
        docs = [
            TextAnalysis.StringDocument("word1 word2 word3 word4"),
            TextAnalysis.StringDocument("word1 word2 word5 word6"),
            TextAnalysis.StringDocument("word1 word7 word8 word9")
        ]
        corpus = TextAssociations.Corpus(docs)

        @testset "Vocabulary Coverage" begin
            coverage = vocab_coverage(corpus, percentiles=0.25:0.25:1.0)
            @test isa(coverage, DataFrame)
            @test "Percentile" in names(coverage)
            @test "WordsNeeded" in names(coverage)
            @test "ProportionOfVocab" in names(coverage)
            @test nrow(coverage) == 4
        end

        @testset "Token Distribution" begin
            dist = token_distribution(corpus)
            @test isa(dist, DataFrame)
            @test "Token" in names(dist)
            @test "Frequency" in names(dist)
            @test "DocFrequency" in names(dist)
            @test "DocFrequencyRatio" in names(dist)
            @test "IDF" in names(dist)
            @test "TFIDF" in names(dist)
        end

        @testset "Coverage Summary Display" begin
            stats = corpus_statistics(corpus)
            # Test that coverage_summary doesn't error
            @test_nowarn coverage_summary(stats)
        end
    end

    @testset "Stream Processing" begin
        # Create test files
        temp_dir = mktempdir()
        try
            for i in 1:5
                file = joinpath(temp_dir, "doc_$i.txt")
                open(file, "w") do f
                    write(f, "test document $i with some words")
                end
            end

            # Test streaming (basic test - full implementation would need more setup)
            pattern = joinpath(temp_dir, "*.txt")
            @test_nowarn begin
                # This would need actual implementation testing
                # For now just verify the function exists
                @test isdefined(TextAssociations, :stream_corpus_analysis)
            end
        finally
            rm(temp_dir, recursive=true)
        end
    end

    @testset "Metric Evaluation Functions" begin
        # Test that all metrics have their eval_ functions
        text = "Test text for metric functions"
        ct = ContingencyTable(text, "test", 3, 1)

        for metric in listmetrics()
            func_name = Symbol("eval_", lowercase(string(metric)))
            @test isdefined(TextAssociations, func_name)

            # Test that the function can be called
            if !isempty(extract_cached_data(ct.con_tbl))
                @test_nowarn TextAssociations.eval(func_name)(ct)
            end
        end
    end

    @testset "DataFrame Construction from Lazy Process" begin
        # Test the new ContingencyTable constructor from DataFrame
        df = DataFrame(
            Collocate=[:word1, :word2],
            a=[5, 3],
            b=[2, 4],
            c=[1, 2],
            d=[10, 8],
            m=[7, 7],
            n=[11, 10],
            k=[6, 5],
            l=[12, 12],
            N=[18, 17],
            E₁₁=[2.3, 2.1],
            E₁₂=[4.7, 4.9],
            E₂₁=[3.7, 2.9],
            E₂₂=[7.3, 8.1]
        )

        ct = ContingencyTable(df, "test", 5, 2)
        @test ct.node == "test"
        @test ct.windowsize == 5
        @test ct.minfreq == 2

        # Test that metrics work with this constructed table
        result = evalassoc(PMI, ct)
        @test isa(result, DataFrame)
        @test nrow(result) == 2
    end
end

println("All tests completed successfully!")