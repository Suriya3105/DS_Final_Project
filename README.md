# DS_Final_Project RAG-Based Insurance Policy Assistant using Gemini

## Overview
An educational Retrieval-Augmented Generation (RAG) chatbot that answers questions about a sample health-insurance knowledge base. It uses JSON policy records, Sentence Transformers embeddings, FAISS similarity search, Google Gemini generation, and a Streamlit interface.

**Important:** The JSON records are sample project data, not official policy wording. Verify all coverage and claim details with the relevant insurer.

## Technology
- Python and Streamlit
- Google Gemini (`gemini-3.8-flash` by default; configurable with `GEMINI_MODEL`)
- `sentence-transformers/all-MiniLM-L6-v2` embeddings (384 dimensions)
- FAISS vector search
- JSON knowledge base (20 records)
- NumPy, pandas, scikit-learn, python-dotenv, google-genai

## Architecture
1. Load and validate `data/raw/knowledge_base.json`.
2. Chunk policy text and preserve metadata.
3. Generate semantic embeddings using Sentence Transformers.
4. Build and save a FAISS index.
5. Embed the user's question and retrieve relevant chunks.
6. Pass retrieved context and the question to Gemini with grounding instructions.
7. Display the answer and source records in Streamlit.

## Project structure
```text
Insurance_policy_assistant_6/
├── app.py
├── config.py
├── requirements.txt
├── .env
├── data/
│   ├── raw/knowledge_base.json
│   └── processed/
│       ├── chunks_metadata.json
│       └── embeddings.npy
├── src/
│   ├── document_loader.py
│   ├── text_chunker.py
│   ├── embeddings.py
│   ├── faiss_index.py
│   ├── retriever.py
│   ├── gemini_generator.py
│   └── evaluate_retrieval.py
└── vectorstore/insurance_faiss.index
```

## Setup
Activate your Conda environment, open a terminal in the project root, and install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.8-flash
```

Keep `.env` private. Never upload or share your API key.

## Build the pipeline
Run from the project root:

```bash
python -m src.document_loader
python -m src.text_chunker
python -m src.embeddings
python -m src.faiss_index
```

Rebuild the pipeline if the JSON knowledge base changes.

## Run the app
```bash
streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`).

Use the sidebar's retrieval testing mode to inspect search results without calling Gemini. Turn that mode off to generate an answer. Review the retrieved source records for supporting evidence.

Example questions:
- What is the waiting period for pre-existing diseases?
- How does the cashless claim process work?
- What expenses are covered before and after hospitalization?
- What is a critical illness rider?

## Retrieval evaluation
Run:

```bash
python -m src.evaluate_retrieval
```

Recorded results from the 20-question evaluation:
| Metric | Result |
|---|---:|
| Test questions | 20 |
| Passed | 20/20 |
| Precision@5 | 0.2000 |
| Recall@5 | 1.0000 |
| Hit Rate@5 | 100% |
| Mean Reciprocal Rank | 0.9750 |
| Top-K | 5 |
| Similarity threshold | 0.30 |

The expected relevant document appeared in the top five for all 20 test questions. MRR 0.975 means the expected document was ranked first for 19 questions and second for one. Precision@5 of 0.20 reflects one expected relevant document among five retrieved results per question.

## Limitations and future work
The evaluation measures retrieval, not the factual correctness of every generated answer. The knowledge base is limited sample data and the application does not connect to live insurer systems or execute claims. Potential future enhancements include BM25 hybrid retrieval, cross-encoder reranking, broader human-reviewed evaluation, PII redaction, and authorized insurer API integrations.

## Responsible use
This is an educational prototype, not an authoritative insurance advice or claims system. Consult the official policy wording and insurer before making coverage, medical, financial, or claim decisions.
