
import json
import numpy as np
import faiss

from pathlib import Path
from sentence_transformers import SentenceTransformer


# --------------------------------
# PROJECT DIRECTORIES
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

VECTORSTORE_DIR = BASE_DIR / "vectorstore"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INDEX_PATH = VECTORSTORE_DIR / "insurance_faiss.index"
METADATA_PATH = PROCESSED_DIR / "chunks_metadata.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------
# INSURANCE RETRIEVER
# --------------------------------

class InsuranceRetriever:

    def __init__(self):

        print("Loading embedding model...")

        self.model = SentenceTransformer(MODEL_NAME)

        print("Loading FAISS index...")

        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {INDEX_PATH}"
            )

        self.index = faiss.read_index(str(INDEX_PATH))

        print("Loading JSON policy metadata...")

        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {METADATA_PATH}"
            )

        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            self.metadata = json.load(file)

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                "FAISS index and metadata counts do not match."
            )

        # Verify that all indexed documents are JSON.
        for item in self.metadata:

            if item.get("source_type") != "json":
                raise ValueError(
                    "Non-JSON document found in metadata."
                )

        print("Total JSON vectors:", self.index.ntotal)

        print("Retriever initialized successfully!")


    # --------------------------------
    # SEARCH POLICY DOCUMENTS
    # --------------------------------

    def search(
        self,
        query,
        top_k=5,
        similarity_threshold=0.30
    ):

        if not query or not query.strip():
            raise ValueError(
                "Search query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        if not 0 <= similarity_threshold <= 1:
            raise ValueError(
                "Similarity threshold must be between 0 and 1."
            )

        # Generate query embedding.
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        query_embedding = np.ascontiguousarray(
            query_embedding.astype("float32")
        )

        # Search the JSON-only FAISS index.
        scores, indices = self.index.search(
            query_embedding,
            min(top_k, self.index.ntotal)
        )

        results = []

        for score, index_position in zip(
            scores[0],
            indices[0]
        ):

            if index_position == -1:
                continue

            score = float(score)

            if score < similarity_threshold:
                continue

            result = self.metadata[index_position].copy()

            result["similarity_score"] = score

            results.append(result)

        return results


# --------------------------------
# RETRIEVER TESTING
# --------------------------------

if __name__ == "__main__":

    retriever = InsuranceRetriever()

    questions = [
        "What is hospitalization coverage?",
        "What is the waiting period for pre-existing diseases?",
        "How does the cashless claim process work?",
        "How are health insurance premiums calculated?",
        "What is a critical illness rider?"
    ]

    for question in questions:

        print("\n" + "=" * 65)

        print("USER QUESTION:", question)

        results = retriever.search(
            question,
            top_k=5,
            similarity_threshold=0.30
        )

        if not results:

            print("No relevant JSON policy documents found.")

            continue

        for rank, result in enumerate(results, start=1):

            print("\nRESULT:", rank)

            print("Document ID:", result["doc_id"])

            print("Source Type:", result["source_type"])

            print("Title:", result["title"])

            print(
                "Similarity:",
                round(result["similarity_score"], 4)
            )

            print("Content:", result["text"])

    print("\nJSON-only retrieval testing completed!")
