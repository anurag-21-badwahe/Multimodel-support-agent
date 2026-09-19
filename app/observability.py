"""Optional Langfuse tracing for API requests and LangChain generations."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from dotenv import load_dotenv

load_dotenv()


def langfuse_configured() -> bool:
    """Return whether all credentials required for Langfuse tracing exist."""

    return all(
        os.getenv(name, "").strip()
        for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL")
    )


@dataclass
class RequestTrace:
    """Request-scoped callback state shared by graph model calls."""

    callbacks: list[Any]
    root: Any

    def complete(self, result: dict[str, Any]) -> None:
        """Record a small, non-sensitive request result on the root span."""

        if self.root is not None:
            self.root.update(
                output={
                    "status": "success",
                    "detected_error_code": result.get("detected_error_code"),
                    "evidence_found": bool(result.get("evidence_found")),
                    "image_uncertain": bool(result.get("image_uncertain")),
                }
            )


@contextmanager
def request_trace(question: str, has_image: bool) -> Iterator[RequestTrace]:
    """Create a root trace when configured, otherwise provide a no-op context."""

    if not langfuse_configured():
        yield RequestTrace(callbacks=[], root=None)
        return

    from langfuse import get_client
    from langfuse.langchain import CallbackHandler

    client = get_client()
    handler = CallbackHandler()
    with client.start_as_current_observation(
        as_type="span",
        name="chat-response",
        input={"question": question, "has_image": has_image},
    ) as root:
        trace = RequestTrace(callbacks=[handler], root=root)
        try:
            yield trace
        except Exception as exc:
            root.update(
                output={"status": "error", "error_type": type(exc).__name__},
            )
            raise
        finally:
            client.flush()