(() => {
  "use strict";

  const question = document.getElementById("question");
  const imageInput = document.getElementById("image");
  const previewWrap = document.getElementById("previewWrap");
  const preview = document.getElementById("preview");
  const removeImage = document.getElementById("removeImage");
  const sendButton = document.getElementById("sendButton");
  const sendLabel = document.getElementById("sendLabel");
  const spinner = document.getElementById("spinner");
  const clearButton = document.getElementById("clearButton");
  const status = document.getElementById("status");
  const answerContent = document.getElementById("answerContent");
  const detectedCode = document.getElementById("detectedCode");
  const imageStatus = document.getElementById("imageStatus");
  const sourcesList = document.getElementById("sourcesList");
  const evidencePill = document.getElementById("evidencePill");

  const MAX_IMAGE_BYTES = 5 * 1024 * 1024;
  let previewUrl = null;

  function setStatus(message = "", type = "") {
    status.textContent = message;
    status.className = `status ${type}`.trim();
  }

  function resetPreview() {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      previewUrl = null;
    }
    imageInput.value = "";
    preview.removeAttribute("src");
    previewWrap.hidden = true;
    imageStatus.textContent = "No image analysis yet.";
  }

  function renderSources(sources) {
    sourcesList.innerHTML = "";
    if (!sources.length) {
      const empty = document.createElement("p");
      empty.className = "placeholder";
      empty.textContent = "No CSV evidence was used.";
      sourcesList.appendChild(empty);
      return;
    }

    for (const source of sources) {
      const item = document.createElement("div");
      item.className = "source-item";

      const file = document.createElement("div");
      file.className = "source-file";
      file.textContent = source.file;

      const meta = document.createElement("div");
      meta.className = "source-meta";
      meta.textContent = `Record: ${source.record} · ${source.match_type}`;

      item.append(file, meta);
      sourcesList.appendChild(item);
    }
  }

  function renderResult(data) {
    answerContent.textContent = data.answer || "No answer returned.";
    detectedCode.textContent = data.detected_error_code || "—";

    if (data.image_uncertain) {
      imageStatus.textContent = "Image result was uncertain; no unsupported code was used.";
    } else if (data.detected_error_code) {
      imageStatus.textContent = "The graph validated the detected code before CSV lookup.";
    } else {
      imageStatus.textContent = "No error code was detected from the text request.";
    }

    evidencePill.textContent = data.evidence_found ? "Evidence found" : "Insufficient evidence";
    renderSources(data.sources || []);
  }

  function setLoading(loading) {
    sendButton.disabled = loading;
    clearButton.disabled = loading;
    imageInput.disabled = loading;
    question.disabled = loading;
    spinner.hidden = !loading;
    sendLabel.textContent = loading ? "Processing..." : "Send request";
  }

  async function parseError(response) {
    try {
      const body = await response.json();
      return body.detail || "Request failed.";
    } catch {
      return `Request failed with HTTP ${response.status}.`;
    }
  }

  imageInput.addEventListener("change", () => {
    setStatus("");
    const file = imageInput.files?.[0];
    if (!file) {
      resetPreview();
      return;
    }

    if (!['image/png', 'image/jpeg'].includes(file.type)) {
      resetPreview();
      setStatus("Only PNG and JPG/JPEG images are supported.", "error");
      return;
    }

    if (file.size > MAX_IMAGE_BYTES) {
      resetPreview();
      setStatus("The selected image exceeds the 5 MB limit.", "error");
      return;
    }

    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    previewWrap.hidden = false;
    imageStatus.textContent = `${file.name} selected.`;
  });

  removeImage.addEventListener("click", resetPreview);

  clearButton.addEventListener("click", () => {
    question.value = "";
    resetPreview();
    answerContent.innerHTML = '<p class="placeholder">Your grounded troubleshooting response will appear here.</p>';
    detectedCode.textContent = "—";
    evidencePill.textContent = "No result yet";
    sourcesList.innerHTML = '<p class="placeholder">No sources yet.</p>';
    setStatus("");
    question.focus();
  });

  sendButton.addEventListener("click", async () => {
    const text = question.value.trim();
    const file = imageInput.files?.[0] || null;

    if (!text && !file) {
      setStatus("Enter a question or upload an image first.", "error");
      return;
    }

    if (file && file.size > MAX_IMAGE_BYTES) {
      setStatus("The selected image exceeds the 5 MB limit.", "error");
      return;
    }

    setLoading(true);
    setStatus("Running the LangGraph workflow...", "success");

    try {
      let response;
      if (file) {
        const form = new FormData();
        form.append("question", text);
        form.append("image", file);
        response = await fetch("/api/chat/query-with-image", {
          method: "POST",
          body: form,
        });
      } else {
        response = await fetch("/api/chat/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: text }),
        });
      }

      if (!response.ok) {
        throw new Error(await parseError(response));
      }

      const data = await response.json();
      renderResult(data);
      setStatus(data.evidence_found ? "Grounded response generated from synthetic CSV evidence." : data.answer, data.evidence_found ? "success" : "error");
    } catch (error) {
      setStatus(error?.message || "Network request failed. Please try again.", "error");
    } finally {
      setLoading(false);
    }
  });
})();
