"""FastAPI application for the multimodal troubleshooting graph."""

from __future__ import annotations

import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from io import BytesIO

from app.graph import MAX_IMAGE_BYTES, troubleshooting_graph

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

ALLOWED_MIME_TYPES = {
    "image/png": "image/png",
    "image/jpeg": "image/jpeg",
}
ALLOWED_PIL_FORMATS = {"PNG", "JPEG"}


class ChatQueryRequest(BaseModel):
    """JSON payload for text-only requests."""

    question: str = Field(min_length=1, max_length=4000)


class SourceRecord(BaseModel):
    """A CSV file + record attribution returned to the frontend."""

    file: str
    record: str
    match_type: str


class ChatResponse(BaseModel):
    """Stable API response exposed by both chat endpoints."""

    answer: str
    detected_error_code: str | None = None
    sources: list[SourceRecord] = Field(default_factory=list)
    evidence_found: bool = False
    image_uncertain: bool = False
    synthetic_data: bool = True


def _validate_uploaded_image(filename: str | None, content_type: str | None, data: bytes) -> str:
    """Validate MIME type, size, and actual image bytes.

    Checking both the HTTP content type and the decoded image prevents a caller
    from simply renaming an arbitrary file to ``.png`` or ``.jpg``.
    """

    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 5 MB or smaller.")

    if content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=415, detail="Only PNG and JPG/JPEG images are supported.")

    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
        with Image.open(BytesIO(data)) as image:
            actual_format = (image.format or "").upper()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=415, detail="The uploaded file is not a valid image.") from exc

    if actual_format not in ALLOWED_PIL_FORMATS:
        raise HTTPException(status_code=415, detail="Only PNG and JPG/JPEG images are supported.")

    # Guard against a valid image supplied under an unexpected content type.
    expected_mime = "image/png" if actual_format == "PNG" else "image/jpeg"
    if content_type != expected_mime:
        raise HTTPException(
            status_code=415,
            detail=f"File content does not match its declared type. Expected {expected_mime}.",
        )

    return expected_mime


def _to_response(result: dict) -> ChatResponse:
    """Map internal graph state into the documented API contract."""

    return ChatResponse(
        answer=result.get("answer", ""),
        detected_error_code=result.get("detected_error_code"),
        sources=[SourceRecord(**source) for source in result.get("sources", [])],
        evidence_found=bool(result.get("evidence_found")),
        image_uncertain=bool(result.get("image_uncertain")),
        synthetic_data=True,
    )


app = FastAPI(
    title="Multimodal LangGraph Troubleshooting",
    version="1.0.0",
    description="Synthetic-data-only troubleshooting assistant with text and image routing.",
)

# Same-origin deployment is the default. CORS is configurable for local frontend
# development or a separately hosted static frontend.
cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:8000").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Serve the single-page frontend."""

    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    """Basic liveness endpoint."""

    return {"status": "ok"}


@app.post("/api/chat/query", response_model=ChatResponse)
async def chat_query(payload: ChatQueryRequest) -> ChatResponse:
    """Process a text-only question through the LangGraph workflow."""

    try:
        result = await troubleshooting_graph.ainvoke({"question": payload.question})
        print("called api/chat/query")
        return _to_response(result)
    except HTTPException:
        raise
    except Exception as exc:
        # Do not leak provider internals or stack traces to end users.
        print("called api/chat/query request failed")
        raise HTTPException(status_code=500, detail=f"Troubleshooting request failed: {exc}") from exc


@app.post("/api/chat/query-with-image", response_model=ChatResponse)
async def chat_query_with_image(
    question: str = Form(default=""),
    image: UploadFile = File(...),
) -> ChatResponse:
    """Validate an uploaded image, base64-encode it, and invoke the graph."""

    data = await image.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 5 MB or smaller.")

    mime = _validate_uploaded_image(image.filename, image.content_type, data)
    image_b64 = base64.b64encode(data).decode("ascii")

    try:
        result = await troubleshooting_graph.ainvoke(
            {
                "question": question,
                "image_base64": image_b64,
                "image_mime_type": mime,
            }
        )
        return _to_response(result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image troubleshooting request failed: {exc}") from exc
