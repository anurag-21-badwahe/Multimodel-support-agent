"""LangGraph workflow for grounded multimodal troubleshooting.

The graph intentionally has two convergence paths:

Text:
    START -> understand_query -> extract_error_code -> lookup_data -> evidence_check

Image:
    START -> validate_image -> extract_error_code_from_image -> lookup_data -> evidence_check

Both paths share the same deterministic Pandas lookup and grounding gate.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
from io import BytesIO
from typing import Any, TypedDict

from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from app.data_lookup import get_error_catalog, load_datasets, lookup_by_error_code
from app.llm_factory import get_llm

# Load .env before reading model/image configuration because app.main imports this module.
load_dotenv()

# Easy-to-swap model constants. Environment variables allow deployment-specific
# overrides without changing source code.
FAST_TEXT_MODEL = os.getenv("FAST_TEXT_MODEL", "cohere/north-mini-code:free")
VISION_MODEL = os.getenv("VISION_MODEL", "inclusionai/ling-3.0-flash-vl:free")


# FAST_TEXT_MODEL = "google/gemma-4-26b-a4b-it:free"
# VISION_MODEL =  "inclusionai/ling-3.0-flash-vl:free"

MAX_IMAGE_BYTES = int(float(os.getenv("MAX_IMAGE_SIZE_MB", "5")) * 1024 * 1024)
SUPPORTED_IMAGE_FORMATS = {"PNG", "JPEG"}
ERROR_CODE_PATTERN = re.compile(r"\bE-\d{3}\b")


class TroubleshootingState(TypedDict, total=False):
    """State passed between graph nodes.

    ``image_base64`` and ``detected_error_code`` are intentionally optional so
    the same state shape supports both the text and multimodal workflows.
    """

    question: str
    image_base64: str
    image_mime_type: str
    normalized_question: str
    detected_error_code: str | None
    image_uncertain: bool
    evidence: dict[str, Any] | None
    evidence_found: bool
    sources: list[dict[str, Any]]
    answer: str
    error_message: str | None


def _message_text(response: Any) -> str:
    """Normalize LangChain AIMessage content into plain text."""

    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()

    return str(content).strip()


def _known_error_codes() -> set[str]:
    """Return codes that actually exist in the primary synthetic table."""

    return {
        row["error_code"]
        for row in get_error_catalog()
        if row.get("error_code")
    }


def _first_code(text: str) -> str | None:
    """Extract the first syntactically valid E-xxx code from model output."""

    match = ERROR_CODE_PATTERN.search(text or "")
    return match.group(0) if match else None


def _validate_known_code(code: str | None) -> str | None:
    """Only accept LLM-derived codes that are present in the synthetic catalog."""

    if code and code in _known_error_codes():
        return code
    return None


def _build_catalog_prompt() -> str:
    catalog = get_error_catalog()
    lines = [json.dumps(item, ensure_ascii=False) for item in catalog]
    return "\n".join(lines)


async def understand_query(state: TroubleshootingState) -> dict[str, Any]:
    """Normalize and validate the user's text request.

    This node deliberately does not retrieve data. It is the graph's clear text
    entry stage; code interpretation happens in the dedicated extraction node.
    """

    question = (state.get("question") or "").strip()
    if not question:
        return {
            "normalized_question": "",
            "error_message": "Please enter a troubleshooting question.",
        }

    return {
        "normalized_question": " ".join(question.split()),
        "error_message": None,
    }


async def extract_error_code(state: TroubleshootingState) -> dict[str, Any]:
    """Extract an explicit or catalog-supported error code from text.

    A syntactically explicit code such as E-999 is preserved exactly and passed
    into the lookup gate; the evidence check then determines that it is unsupported.
    For natural language, the LLM must select only from codes present in the CSV.
    """

    question = state.get("normalized_question") or state.get("question", "")
    if not question:
        return {"detected_error_code": None}

    explicit_code = _first_code(question)
    if explicit_code:
        return {
            "detected_error_code": explicit_code,
            "image_uncertain": False,
            "error_message": None,
        }

    catalog = _build_catalog_prompt()
    prompt = f"""You are the error-code extraction step in a troubleshooting system.
You may select an error code ONLY from the synthetic catalog below.
Do not invent, normalize, or modify a code. If the question cannot be mapped to one
catalog entry with reasonable confidence, return exactly NONE.

Synthetic catalog:
{catalog}

User question:
{question}

Return exactly one token: a catalog error code such as E-104, or NONE.
"""

    model = get_llm(FAST_TEXT_MODEL)
    response = await model.ainvoke(prompt)
    code = _validate_known_code(_first_code(_message_text(response)))

    return {
        "detected_error_code": code,
        "image_uncertain": False,
        "error_message": None if code else "No supported synthetic error code could be identified from the question.",
    }


async def validate_image(state: TroubleshootingState) -> dict[str, Any]:
    """Validate base64 image content before the vision model sees it."""

    image_b64 = state.get("image_base64")
    if not image_b64:
        return {
            "image_uncertain": True,
            "error_message": "No image was supplied. Please upload a PNG or JPG image.",
        }

    try:
        raw = base64.b64decode(image_b64, validate=True)
    except (ValueError, binascii.Error):
        return {
            "image_uncertain": True,
            "error_message": "The uploaded image payload is not valid base64 data.",
        }

    if len(raw) > MAX_IMAGE_BYTES:
        return {
            "image_uncertain": True,
            "error_message": "The image exceeds the 5 MB maximum size.",
        }

    try:
        with Image.open(BytesIO(raw)) as image:
            image.verify()
        with Image.open(BytesIO(raw)) as image:
            image_format = (image.format or "").upper()
    except (UnidentifiedImageError, OSError, ValueError):
        return {
            "image_uncertain": True,
            "error_message": "The uploaded file is not a valid PNG or JPG image.",
        }

    if image_format not in SUPPORTED_IMAGE_FORMATS:
        return {
            "image_uncertain": True,
            "error_message": "Only PNG and JPG/JPEG images are supported.",
        }

    return {
        "image_uncertain": False,
        "error_message": None,
    }


async def extract_error_code_from_image(state: TroubleshootingState) -> dict[str, Any]:
    """Use the dedicated heavy multimodal model exclusively for image extraction."""

    if state.get("image_uncertain"):
        return {}

    image_b64 = state.get("image_base64", "")
    mime = state.get("image_mime_type", "image/png")
    if not image_b64:
        return {
            "image_uncertain": True,
            "error_message": "No image content is available for visual extraction.",
        }

    data_url = f"data:{mime};base64,{image_b64}"
    prompt = """Read the error code shown in this synthetic troubleshooting image.
The image should contain one E-xxx code. Return exactly the visible code and nothing else.
If the code cannot be read clearly, return UNCERTAIN.
Never invent a code.
"""

    try:
        model = get_llm(VISION_MODEL)
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        )
        response = await model.ainvoke([message])
        raw_output = _message_text(response)
    except Exception as exc:  # Model/API failures are operational uncertainty, not evidence.
        return {
            "detected_error_code": None,
            "image_uncertain": True,
            "error_message": f"Image analysis could not be completed: {exc}",
        }

    code = _validate_known_code(_first_code(raw_output))
    if not code:
        return {
            "detected_error_code": None,
            "image_uncertain": True,
            "error_message": "The error code could not be read with sufficient confidence. Please upload a clearer image.",
        }

    return {
        "detected_error_code": code,
        "image_uncertain": False,
        "error_message": None,
    }


async def lookup_data(state: TroubleshootingState) -> dict[str, Any]:
    """Run the single deterministic evidence lookup shared by both graph paths."""

    code = state.get("detected_error_code")
    evidence = lookup_by_error_code(code or "")

    return {
        "evidence": evidence,
        "evidence_found": bool(evidence.get("found")),
        "sources": evidence.get("sources", []),
    }


def evidence_check(state: TroubleshootingState) -> dict[str, Any]:
    """Record the grounding decision; routing is handled by ``route_evidence``."""

    found = bool(state.get("evidence_found"))
    return {"evidence_found": found}


def route_evidence(state: TroubleshootingState) -> str:
    """Route only evidence-backed requests to answer generation."""

    return "generate_answer" if state.get("evidence_found") else "insufficient_information"


async def generate_answer(state: TroubleshootingState) -> dict[str, Any]:
    """Generate a grounded response using only the retrieved synthetic records."""

    evidence = state.get("evidence") or {}
    code = state.get("detected_error_code") or evidence.get("error_code")
    question = state.get("normalized_question") or state.get("question", "")

    evidence_json = json.dumps(evidence, ensure_ascii=False, indent=2)
    prompt = f"""You are the final answer writer for a synthetic equipment-troubleshooting knowledge base.

STRICT GROUNDING RULES:
1. Use ONLY facts present in the RETRIEVED CSV EVIDENCE below.
2. Do not add causes, components, instructions, safety advice, or codes from memory.
3. Do not infer facts that are not explicitly present in the evidence.
4. Preserve the exact error code spelling from the evidence.
5. Clearly state that the data is synthetic.
6. Give a concise practical answer that addresses the user's question.
7. Mention the relevant troubleshooting steps and safety notes only when they exist in the evidence.

User question:
{question}

Detected error code:
{code}

RETRIEVED CSV EVIDENCE (the only allowed knowledge source):
{evidence_json}

Write the final answer now.
"""

    model = get_llm(FAST_TEXT_MODEL)
    response = await model.ainvoke(prompt)
    answer = _message_text(response)

    # A lightweight deterministic safeguard: if the model dropped the exact code,
    # prepend it. We never fabricate a missing code; it comes from the validated state.
    if code and code not in answer:
        answer = f"Error {code}: {answer}"

    return {"answer": answer}


async def insufficient_information(state: TroubleshootingState) -> dict[str, Any]:
    """Return a hardcoded refusal when evidence is missing or the image is unclear."""

    if state.get("image_uncertain"):
        message = "The error code could not be read with sufficient confidence. Please upload a clearer image."
    else:
        code = state.get("detected_error_code")
        if code:
            message = f"The synthetic knowledge base does not contain information about {code}."
        else:
            message = "The synthetic knowledge base does not contain enough information to answer this question."

    return {
        "answer": message,
        "error_message": message,
        "sources": [],
    }


def route_entry(state: TroubleshootingState) -> str:
    """Conditional graph entry required by the assignment."""

    return "validate_image" if state.get("image_base64") else "understand_query"


def build_graph():
    """Construct and compile the complete troubleshooting graph."""

    workflow = StateGraph(TroubleshootingState)

    workflow.add_node("understand_query", understand_query)
    workflow.add_node("extract_error_code", extract_error_code)
    workflow.add_node("validate_image", validate_image)
    workflow.add_node("extract_error_code_from_image", extract_error_code_from_image)
    workflow.add_node("lookup_data", lookup_data)
    workflow.add_node("evidence_check", evidence_check)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("insufficient_information", insufficient_information)

    # Explicit conditional entry point: image state takes the image path;
    # otherwise the request takes the text path.
    workflow.set_conditional_entry_point(
        route_entry,
        {
            "validate_image": "validate_image",
            "understand_query": "understand_query",
        },
    )

    # Text path.
    workflow.add_edge("understand_query", "extract_error_code")
    workflow.add_edge("extract_error_code", "lookup_data")

    # Image path.
    workflow.add_edge("validate_image", "extract_error_code_from_image")
    workflow.add_edge("extract_error_code_from_image", "lookup_data")

    # Shared evidence path.
    workflow.add_edge("lookup_data", "evidence_check")
    workflow.add_conditional_edges(
        "evidence_check",
        route_evidence,
        {
            "generate_answer": "generate_answer",
            "insufficient_information": "insufficient_information",
        },
    )

    workflow.add_edge("generate_answer", END)
    workflow.add_edge("insufficient_information", END)

    return workflow.compile()


troubleshooting_graph = build_graph()

# Prevent an unused-import warning in environments that lint strictly while still
# validating the data contract once at module import.
load_datasets()
