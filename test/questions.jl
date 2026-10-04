@testset "Question types and answers" begin
    choice = ChoiceQuestion(["a" => "A", "b" => "B", "c" => "C"])
    @test option_count(choice) == 3
    @test choice.criteria == ["a" => "A", "b" => "B", "c" => "C"]
    noul = NoulQuestion()
    @test option_count(noul) == 2
    score = ScoreQuestion(["low", "medium", "high"])
    @test option_count(score) == 3

    @test_throws ArgumentError ChoiceQuestion(Pair{String,String}[])
    @test_throws ArgumentError ChoiceQuestion(["same" => "A", "same" => "B"])
    @test_throws ArgumentError ScoreQuestion(["one"])
    @test_throws ArgumentError ScoreQuestion(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"])

    # Softmax over present options; a single option is certain.
    @test probabilities(Float32[0, log(9)], 1.0) ≈ [0.1, 0.9]
    @test probabilities(Float32[42], 1.0) ≈ [1.0]
    # Temperature 0 is not a calibration value the checkpoint stores.
    @test_throws ArgumentError probabilities(Float32[NaN, 0], 1.0)
    @test_throws ArgumentError probabilities(Float32[Inf, 0], 1.0)

    values = probabilities(Float32[0, log(3), log(2)], 1.0)
    choice_answer = answer(choice, values)
    @test choice_answer.type == "choice"
    @test choice_answer.choice == "b"
    @test choice_answer.probabilities["b"] ≈ 0.5
    @test choice_answer.confidence ≈ 0.25 atol = 1e-6

    @test answer(noul, probabilities(Float32[0, log(3)], 1.0)).noul ≈ 0.75

    score_answer = answer(score, probabilities(Float32[0, log(2), log(3)], 1.0))
    @test score_answer.type == "score"
    @test score_answer.score ≈ (0 * 1 + 1 * 2 + 2 * 3) / 6
    @test score_answer.legend == Dict("0" => "low", "1" => "medium", "2" => "high")

    singleton = answer(ChoiceQuestion(["only" => "Only"]), [1.0])
    @test singleton.choice == "only"
    @test singleton.confidence == 1.0

    tied = answer(
        ChoiceQuestion(["first" => "First", "second" => "Second"]),
        [0.5, 0.5],
    )
    @test tied.choice == "first"
    @test tied.confidence == 0.0
end
