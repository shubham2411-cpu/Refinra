# ============================================================
# CONFIGURATION
# ============================================================

MODE = "gemini"


# ============================================================
# MAIN PIPELINE ROUTER
# ============================================================

def run_pipeline(question):

    if MODE == "mock":
        return run_mock_pipeline(question)

    elif MODE == "gemini":
        from gemini_pipeline import run_gemini_pipeline
        return run_gemini_pipeline(question)

    else:
        raise ValueError("Invalid mode. Use 'mock' or 'gemini'.")


# ============================================================
# MOCK SOLVER
# ============================================================

def solve(question, scenario="wrong_calculation"):

    if "*" in question:

        numbers = question.split("*")

        a = int(numbers[0].strip())
        b = int(numbers[1].strip())

        correct_answer = a * b

        if scenario == "correct":
            return f"The answer is {correct_answer}."

        elif scenario == "wrong_calculation":
            wrong_answer = correct_answer + 10
            return f"The answer is {wrong_answer}."

        elif scenario == "incomplete":
            return (
                f"The first number is {a} "
                f"and the second number is {b}."
            )

        elif scenario == "wrong_assumption":
            return (
                f"Assuming the numbers should be added instead, "
                f"the answer is {a + b}."
            )

    return "I don't know how to solve this question yet."


# ============================================================
# MOCK VERIFIER
# ============================================================

def verify(question, solver_answer):

    if "*" in question:

        numbers = question.split("*")

        a = int(numbers[0].strip())
        b = int(numbers[1].strip())

        correct_answer = a * b

        if str(correct_answer) in solver_answer:

            return (
                f"Verdict: PASS\n"
                f"Confidence: 1.00\n"
                f"Explanation: The Solver's answer is correct. "
                f"{a} × {b} = {correct_answer}."
            )

        else:

            return (
                f"Verdict: FAIL\n"
                f"Confidence: 1.00\n"
                f"Explanation: The Solver's answer is incorrect. "
                f"The correct result is {a} × {b} = {correct_answer}."
            )

    return (
        "Verdict: FAIL\n"
        "Confidence: 0.00\n"
        "Explanation: Unable to verify this question yet."
    )


# ============================================================
# MOCK CRITIC
# ============================================================

def criticize(question, solver_answer, verification):

    if "Verdict: FAIL" in verification:

        if "Assuming" in solver_answer:

            return (
                "The Solver used an incorrect assumption. "
                "The question requires multiplication, not addition."
            )

        elif "first number" in solver_answer:

            return (
                "The Solver did not provide the requested calculation "
                "or final answer."
            )

        else:

            return (
                "The Solver made an incorrect calculation. "
                "The Verifier identified the correct result."
            )

    return "NO SIGNIFICANT ISSUES"


# ============================================================
# MOCK FINALIZER
# ============================================================

def finalize(question, solver_answer, verification, critique):

    if "Verdict: FAIL" in verification:

        correct_answer = (
            verification
            .split("=")[-1]
            .strip()
            .rstrip(".")
        )

        return f"The correct answer is {correct_answer}."

    return solver_answer


# ============================================================
# MOCK PIPELINE
# ============================================================

def run_mock_pipeline(question, scenario="wrong_calculation"):

    solver_answer = solve(
        question,
        scenario
    )

    verification = verify(
        question,
        solver_answer
    )

    critique = criticize(
        question,
        solver_answer,
        verification
    )

    final_answer = finalize(
        question,
        solver_answer,
        verification,
        critique
    )

    return {
        "solver": solver_answer,
        "verifier": verification,
        "critic": critique,
        "final": final_answer
    }


# ============================================================
# AUTOMATED MOCK TESTS
# ============================================================

def run_tests():

    test_cases = [
        ("16*25", "correct", "400", "PASS"),
        ("17*24", "wrong_calculation", "408", "FAIL"),
        ("100*7", "incomplete", "700", "FAIL"),
        ("12*12", "wrong_assumption", "144", "FAIL"),
        ("25*4", "wrong_calculation", "100", "FAIL")
    ]

    print("\n==============================")
    print("       RUNNING TESTS")
    print("==============================")

    for question, scenario, expected_answer, expected_verdict in test_cases:

        result = run_mock_pipeline(
            question,
            scenario
        )

        print(f"\nQuestion: {question}")
        print(f"Scenario: {scenario}")

        print(f"Solver:   {result['solver']}")
        print(f"Verifier: {result['verifier']}")
        print(f"Critic:   {result['critic']}")
        print(f"Final:    {result['final']}")

        # ----------------------------------------------------
        # Assertions
        # ----------------------------------------------------

        assert expected_answer in result["final"], (
            f"Final answer test failed for {question}. "
            f"Expected {expected_answer}, got: {result['final']}"
        )

        assert expected_verdict in result["verifier"], (
            f"Verifier test failed for {question}. "
            f"Expected {expected_verdict}, got: {result['verifier']}"
        )

    print("\n==============================")
    print("   ALL TESTS PASSED")
    print("==============================")


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    if MODE == "mock":

        run_tests()

    elif MODE == "gemini":

        question = input("\nEnter your question: ")

        result = run_pipeline(question)

        print("\nFinal Answer:")
        print(result)

    else:

        raise ValueError("Invalid mode. Use 'mock' or 'gemini'.")