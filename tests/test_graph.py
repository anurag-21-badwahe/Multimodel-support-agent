import asyncio
import base64
from io import BytesIO

from PIL import Image
from langchain_core.messages import AIMessage

import app.graph as graph_module


class FakeModel:
    def __init__(self, response_text):
        self.response_text = response_text
        self.prompts = []

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        return AIMessage(content=self.response_text)


def synthetic_png_base64(text_color=(20, 40, 70)):
    image = Image.new("RGB", (480, 280), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


async def run_graph(state):
    return await graph_module.troubleshooting_graph.ainvoke(state)


def test_text_explicit_code_routes_without_text_model_for_extraction(monkeypatch):
    answer_model = FakeModel("E-104 indicates low coolant pressure based only on the supplied synthetic evidence.")
    calls = []

    def fake_factory(model_name):
        calls.append(model_name)
        return answer_model

    monkeypatch.setattr(graph_module, "get_llm", fake_factory)
    result = asyncio.run(run_graph({"question": "What does E-104 mean?"}))

    assert result["detected_error_code"] == "E-104"
    assert result["evidence_found"] is True
    assert "E-104" in result["answer"]
    assert calls == [graph_module.FAST_TEXT_MODEL]


def test_unknown_explicit_code_is_refused(monkeypatch):
    calls = []

    def fake_factory(model_name):
        calls.append(model_name)
        raise AssertionError("The unsupported explicit code should not need an LLM call")

    monkeypatch.setattr(graph_module, "get_llm", fake_factory)
    result = asyncio.run(run_graph({"question": "What does E-999 mean?"}))

    assert result["detected_error_code"] == "E-999"
    assert result["evidence_found"] is False
    assert result["answer"] == "The synthetic knowledge base does not contain information about E-999."
    assert calls == []


def test_image_path_uses_vision_model_then_shared_lookup(monkeypatch):
    image_model = FakeModel("E-104")
    answer_model = FakeModel("Check the coolant level and valve V2 using only the synthetic records.")

    def fake_factory(model_name):
        return image_model if model_name == graph_module.VISION_MODEL else answer_model

    monkeypatch.setattr(graph_module, "get_llm", fake_factory)

    result = asyncio.run(
        run_graph(
            {
                "question": "How do I resolve the issue?",
                "image_base64": synthetic_png_base64(),
                "image_mime_type": "image/png",
            }
        )
    )

    assert result["detected_error_code"] == "E-104"
    assert result["evidence_found"] is True
    assert any(source["file"] == "troubleshooting.csv" for source in result["sources"])


def test_unclear_image_returns_clearer_image_refusal(monkeypatch):
    image_model = FakeModel("UNCERTAIN")
    monkeypatch.setattr(graph_module, "get_llm", lambda _: image_model)

    result = asyncio.run(
        run_graph(
            {
                "question": "How do I resolve this?",
                "image_base64": synthetic_png_base64(),
                "image_mime_type": "image/png",
            }
        )
    )

    assert result["evidence_found"] is False
    assert result["image_uncertain"] is True
    assert "clearer image" in result["answer"]
