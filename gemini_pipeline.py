from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel
import json
import math
from dataclasses import dataclass


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

MODEL_NAME = "gemini-3.8-flash"


# Retry only temporary server/network conditions.
# 429 is intentionally excluded because it represents
# quota/rate-limit conditions that should be surfaced immediately.

RETRY_OPTIONS = types.HttpRetryOptions(
    attempts=3,
    initial_delay=2.0,
    max_delay=8.0,
    http_status_codes=[
        408,
        500,
        502,
        503,
        504
    ]
)


client = genai.Client(
    http_options=types.HttpOptions(
        retry_options=RETRY_OPTIONS
    )
)


# ============================================================
# STRUCTURED VERIFIER SCHEMA
# ============================================================

class VerifierSchema(BaseModel):
    verdict: str
    confidence: float
    explanation: str


# ============================================================
# STRUCTURED VERIFIER RESULT
# ============================================================

@dataclass
class VerifierResult:
    verdict: str          # "PASS" or "FAIL"
    confidence: float     # numeric, 0.0 .. 1.0
    explanation: str      # non-empty short rationale


# ============================================================
# GEMINI REQUEST HANDLER
# ============================================================

def ask_gemini(prompt):

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text


# ============================================================
# SOLVER
# ============================================================

def solve(question):

    return ask_gemini(f"""
You are the Solver in an AI verification system.

Solve the user's question carefully.

Show your reasoning clearly.
State assumptions when necessary.
Check calculations, equations, units, and important details.

User question:
{question}

Provide a complete answer.
""")


# ============================================================
# VERIFIER VALIDATION
# ============================================================

def _parse_verifier_output(text):
    """
    Validate the Verifier's structured JSON output.

    Expects exactly:
        verdict
        confidence
        explanation

    Raises ValueError if validation fails.
    """

    if text is None:

        raise ValueError(
            "Verifier returned no text."
        )

    try:

        obj = json.loads(text)

    except json.JSONDecodeError as e:

        raise ValueError(
            f"Verifier output was not valid JSON: "
            f"{text!r} ({e})"
        )

    if not isinstance(obj, dict):

        raise ValueError(
            f"Verifier output JSON must be an object, "
            f"got {type(obj).__name__}."
        )

    # --------------------------------------------------------
    # Exact key set
    # --------------------------------------------------------

    required_keys = (
        "verdict",
        "confidence",
        "explanation"
    )

    missing_keys = [
        key
        for key in required_keys
        if key not in obj
    ]

    if missing_keys:

        raise ValueError(
            f"Verifier output JSON is missing required key(s): "
            f"{missing_keys}."
        )

    extra_keys = [
        key
        for key in obj
        if key not in required_keys
    ]

    if extra_keys:

        raise ValueError(
            f"Verifier output JSON contains unexpected key(s): "
            f"{extra_keys}. Allowed keys are exactly: "
            f"{list(required_keys)}."
        )

    # --------------------------------------------------------
    # Verdict validation
    # --------------------------------------------------------

    raw_verdict = obj.get("verdict")

    if not isinstance(raw_verdict, str):

        raise ValueError(
            f"Verifier 'verdict' must be a string, got "
            f"{type(raw_verdict).__name__}: {raw_verdict!r}."
        )

    verdict = raw_verdict.strip().upper()

    if verdict not in ("PASS", "FAIL"):

        raise ValueError(
            f"Verifier 'verdict' must be exactly "
            f"'PASS' or 'FAIL', got {raw_verdict!r}."
        )

    # --------------------------------------------------------
    # Confidence validation
    # --------------------------------------------------------

    raw_confidence = obj.get("confidence")

    if isinstance(raw_confidence, bool) or not isinstance(
        raw_confidence,
        (int, float)
    ):

        raise ValueError(
            f"Verifier 'confidence' must be numeric, got "
            f"{type(raw_confidence).__name__}: {raw_confidence!r}."
        )

    confidence = float(raw_confidence)

    if not math.isfinite(confidence):

        raise ValueError(
            f"Verifier 'confidence' must be a finite number, "
            f"got {confidence}."
        )

    if confidence < 0.0 or confidence > 1.0:

        raise ValueError(
            f"Verifier 'confidence' must be between "
            f"0.0 and 1.0 inclusive, got {confidence}."
        )

    # --------------------------------------------------------
    # Explanation validation
    # --------------------------------------------------------

    raw_explanation = obj.get("explanation")

    if not isinstance(raw_explanation, str):

        raise ValueError(
            f"Verifier 'explanation' must be a string, got "
            f"{type(raw_explanation).__name__}: {raw_explanation!r}."
        )

    explanation = raw_explanation.strip()

    if not explanation:

        raise ValueError(
            "Verifier 'explanation' must be a non-empty string."
        )

    return VerifierResult(
        verdict=verdict,
        confidence=confidence,
        explanation=explanation,
    )


# ============================================================
# VERIFIER
# ============================================================

def verify(question, solver_answer):

    prompt = f"""
You are the Verifier in an AI verification system.

Independently check the Solver's answer.

Determine whether the Solver's answer is correct.

Rules:

- verdict must be PASS if the Solver's answer is correct.
- verdict must be FAIL if the Solver's answer contains a
  meaningful error, incorrect calculation, unsupported assumption,
  missing required answer, or other important problem.
- confidence must be a number from 0.0 to 1.0 representing
  confidence in your verdict.
- explanation must be one short sentence under 30 words.

User question:
{question}

Solver's answer:
{solver_answer}
"""

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": VerifierSchema,
            }
        )

        raw_output = response.text

        # ----------------------------------------------------
        # Server-side validation remains as a second layer.
        # ----------------------------------------------------

        return _parse_verifier_output(raw_output)

    except ValueError as e:

        raise RuntimeError(
            f"Failed to validate Verifier output: {e}"
        ) from e


# ============================================================
# FORMAT VERIFIER RESULT
# ============================================================

def _format_verifier(verification):

    return (
        f"Verdict: {verification.verdict}\n"
        f"Confidence: {verification.confidence:.2f}\n"
        f"Explanation: {verification.explanation}"
    )


# ============================================================
# CRITIC
# ============================================================

def criticize(question, solver_answer, verification):

    verification_text = _format_verifier(verification)

    return ask_gemini(f"""
You are the Critic in an AI verification system.

Review the Solver's answer and the Verifier's assessment.

Identify any remaining problems, missing reasoning, incorrect
assumptions, calculation errors, or weaknesses.

User question:
{question}

Solver's answer:
{solver_answer}

Verifier's assessment:
{verification_text}

Give a concise critique.
""")


# ============================================================
# FINALIZER
# ============================================================

def finalize(
    question,
    solver_answer,
    verification,
    critique
):

    verification_text = _format_verifier(verification)

    return ask_gemini(f"""
You are the Finalizer in an AI verification system.

Produce the final answer to the user's question.

Use the Solver's answer as the starting point, but incorporate
the Verifier's assessment and the Critic's analysis.

Correct any errors identified during verification.

Do not mention the internal verification process unless necessary.

User question:
{question}

Solver's answer:
{solver_answer}

Verifier's assessment:
{verification_text}

Critic's analysis:
{critique}

Return the final answer clearly and accurately.
""")


# ============================================================
# FULL GEMINI PIPELINE
# ============================================================

def run_gemini_pipeline(question):

    # --------------------------------------------------------
    # Solver
    # --------------------------------------------------------

    print("\n--- SOLVER ---")

    solver_answer = solve(question)

    print(solver_answer)

    # --------------------------------------------------------
    # Verifier
    # --------------------------------------------------------

    print("\n--- VERIFIER ---")

    verification = verify(
        question,
        solver_answer
    )

    print(f"Verdict:     {verification.verdict}")
    print(f"Confidence:  {verification.confidence:.2f}")
    print(f"Explanation: {verification.explanation}")

    # --------------------------------------------------------
    # Critic
    # --------------------------------------------------------

    print("\n--- CRITIC ---")

    critique = criticize(
        question,
        solver_answer,
        verification
    )

    print(critique)

    # --------------------------------------------------------
    # Finalizer
    # --------------------------------------------------------

    print("\n--- FINAL ANSWER ---")

    final_answer = finalize(
        question,
        solver_answer,
        verification,
        critique
    )

    print(final_answer)

    # --------------------------------------------------------
    # API-compatible response
    # --------------------------------------------------------

    return {
        "solver": solver_answer,
        "verifier": _format_verifier(verification),
        "critic": critique,
        "final": final_answer
    }