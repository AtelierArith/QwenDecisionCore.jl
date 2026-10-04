# Shared contract for decision backends. A concrete backend scores prepared
# inputs and returns one answer per question (see questions.jl). The Qwen
# backbone, not the head, is what this package owns.
abstract type AbstractDecisionBackend end
