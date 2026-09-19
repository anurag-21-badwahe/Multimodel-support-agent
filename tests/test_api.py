import base64
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

import app.main as main_module
from app.main import app


def make_png_bytes():
    image = Image.new("RGB", (240, 140), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_image_endpoint_rejects_non_image():
    client = TestClient(app)
    response = client.post(
        "/api/chat/query-with-image",
        files={"image": ("bad.txt", b"not an image", "text/plain")},
        data={"question": "What is happening?"},
    )
    assert response.status_code == 415


def test_image_endpoint_accepts_png_and_invokes_graph(monkeypatch):
    client = TestClient(app)
    captured = {}

    class FakeGraph:
        async def ainvoke(self, state):
            captured.update(state)
            return {
                "answer": "Synthetic answer",
                "detected_error_code": "E-104",
                "sources": [
                    {
                        "file": "error_codes.csv",
                        "record": "E-104",
                        "match_type": "exact error_code match",
                    }
                ],
                "evidence_found": True,
                "image_uncertain": False,
            }

    monkeypatch.setattr(main_module, "troubleshooting_graph", FakeGraph())
    png = make_png_bytes()
    response = client.post(
        "/api/chat/query-with-image",
        files={"image": ("e104.png", png, "image/png")},
        data={"question": "How do I resolve it?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["detected_error_code"] == "E-104"
    assert captured["question"] == "How do I resolve it?"
    assert base64.b64decode(captured["image_base64"]) == png
