from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pipeline import run_pipeline


# ============================================================
# CONFIGURATION
# ============================================================

MAX_QUESTION_LENGTH = 10_000


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Refinra",
    description=(
        "Refinra - multi-agent AI answer verification system. "
        "Four-stage pipeline: Solver, Verifier, Critic, Finalizer."
    ),
    version="1.0.0"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUESTION_LENGTH
    )


# ============================================================
# SERVE FRONTEND
# ============================================================

app.mount(
    "/app",
    StaticFiles(directory="frontend", html=True),
    name="frontend"
)


# ============================================================
# BASIC ROUTES
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Refinra backend is running.",
        "frontend": "/app"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


# ============================================================
# VERIFY ENDPOINT
# ============================================================

@app.post("/verify")
def verify_question(request: QuestionRequest):

    question = request.question.strip()

    # --------------------------------------------------------
    # Reject whitespace-only questions
    # --------------------------------------------------------

    if not question:

        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_question",
                "detail": "Question cannot be empty."
            }
        )

    try:

        result = run_pipeline(question)

        return result

    except Exception as e:

        error_message = str(e)

        # ----------------------------------------------------
        # Gemini rate-limit / quota error
        # ----------------------------------------------------

        if (
            "429" in error_message
            or "RESOURCE_EXHAUSTED" in error_message.upper()
            or "rate limit" in error_message.lower()
        ):

            return JSONResponse(
                status_code=429,
                content={
                    "error": "gemini_rate_limit_exceeded",
                    "detail": (
                        "Gemini API rate limit or quota was reached. "
                        "Try again later or use an API key with "
                        "available quota."
                    )
                }
            )

        # ----------------------------------------------------
        # Expected pipeline/Gemini failure
        # ----------------------------------------------------

        return JSONResponse(
            status_code=500,
            content={
                "error": "verification_failed",
                "detail": (
                    "Refinra could not complete the verification "
                    "pipeline. Please try again."
                )
            }
        )