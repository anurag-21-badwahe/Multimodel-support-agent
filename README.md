# Multimodal LangGraph Troubleshooting Assignment

A small production-oriented FastAPI + LangGraph application that answers troubleshooting questions from **synthetic CSV data only** and can read an error code from an uploaded PNG/JPG image.

The assignment explicitly requires the text path and image path to converge on one lookup node, evidence checking before answer generation, source attribution, and a hard refusal when the synthetic knowledge base has no matching record. This implementation follows those requirements.

## Architecture

### Text flow

```text
START
  |
  v
understand_query
  |
  v
extract_error_code
  |
  v
lookup_data  <-- exact Pandas match across CSVs
  |
  v
evidence_check
  |------------------------|
  v                        v
generate_answer       insufficient_information
  |                        |
  v                        v
 END                      END
```

### Image flow

```text
START
  |
  v
validate_image
  |
  v
extract_error_code_from_image   <-- heavy multimodal model only here
  |
  v
lookup_data                     <-- same deterministic lookup as text path
  |
  v
evidence_check
  |------------------------|
  v                        v
generate_answer       insufficient_information
  |                        |
  v                        v
 END                      END
```

`app/graph.py` uses `StateGraph.set_conditional_entry_point()` to choose `validate_image` whenever `image_base64` is present; otherwise it enters `understand_query`.

## Project structure

```text
.
├── app/
│   ├── __init__.py
│   ├── data_lookup.py
│   ├── graph.py
│   ├── llm_factory.py
│   └── main.py
├── data/
│   ├── components.csv
│   ├── error_codes.csv
│   └── troubleshooting.csv
├── static/
│   ├── app.js
│   ├── index.html
│   ├── styles.css
│   └── images/
│       ├── E-101.png
│       ├── E-102.png
│       ├── E-103.png
│       ├── E-104.png
│       └── E-105.png
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_data_lookup.py
│   └── test_graph.py
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

## Requirements

Python 3.11+ is recommended.

Install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

Create the local environment file:

```bash
copy .env.example .env
```

Then add at least the key needed by the models you select.

## Model factory

`app/llm_factory.py` provides the required `get_llm(model_name)` function.

- `google/gemini-2.5-pro` -> `ChatOpenAI` pointed to OpenRouter.
- `hf:<repo-id>` -> `ChatHuggingFace` + `HuggingFaceEndpoint`.
- `gpt-*` -> direct `ChatOpenAI`.
- `claude-*` -> `ChatAnthropic`.
- `gemini-*` -> `ChatGoogleGenerativeAI`.

The assignment specifically asks for the OpenRouter path to use `ChatOpenAI` with `https://openrouter.ai/api/v1`, so the factory intentionally follows that contract.

## Running the application

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000

## API examples

### Health

```bash
curl http://localhost:8000/health
```

### Text query

```bash
curl -X POST http://localhost:8000/api/chat/query \\
  -H "Content-Type: application/json" \\
  -d "{\"question\":\"What does E-104 mean?\"}"
```

### Image query

```bash
curl -X POST http://localhost:8000/api/chat/query-with-image \\
  -F "question=How should I resolve this?" \\
  -F "image=@static/images/E-104.png"
```

## Grounding rules

1. `data_lookup.py` loads exactly three CSVs with Pandas.
2. The primary support condition is an exact `error_code` match in `error_codes.csv`.
3. Component and troubleshooting rows are exact matches on their code columns.
4. No vector store or embedding search is used.
5. The final answer prompt receives only the retrieved records.
6. `evidence_check` prevents `generate_answer` from running when no primary evidence exists.
7. Unsupported codes such as `E-999` return the hardcoded refusal:

```text
The synthetic knowledge base does not contain information about E-999.
```

8. An unclear image returns a hardcoded clearer-image message instead of guessing.

## Synthetic images

The repository includes five generated PNG images for E-101 through E-105. Each image shows a large, readable error code and includes the required text:

```text
Synthetic image for training purposes.
```

These are test fixtures, not real equipment photographs.

## Testing

Run the tests with:

```bash
pytest -q
```

The suite covers:

- Exact CSV lookup and unknown-code behavior.
- Text graph routing and grounded answer generation.
- Image graph routing and vision-model separation.
- Unclear image fallback.
- FastAPI health endpoint.
- Image MIME/content validation.
- Base64 image handoff into the graph.

## Production notes

For a real deployment, add structured request IDs/logging, centralized secret management, rate limiting, authentication, provider-specific observability, and a persistence layer if multi-user chat history is introduced. None of those are required by the assignment, so they are intentionally not mixed into the core workflow.
