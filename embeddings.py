
import json
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.text_chunker import create_chunks
from src.document_loader import load_documents


# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"

EMBEDDING_PATH = PROCESSED_DIR / "embeddings.npy"

METADATA_PATH = PROCESSED_DIR / "chunks_metadata.json"


# Embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def generate_embeddings(chunks):

    print("\nLoading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    # Extract text from chunks
    texts = [chunk["text"] for chunk in chunks]

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    return embeddings


def save_embeddings(embeddings, chunks):

    # Save numerical vectors
    np.save(EMBEDDING_PATH, embeddings)

    # Save corresponding metadata
    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=4, ensure_ascii=False)

    print("\nEmbeddings saved successfully!")

    print(f"Embedding path: {EMBEDDING_PATH}")
    print(f"Metadata path: {METADATA_PATH}")


if __name__ == "__main__":

    # Step 1: Load original documents
    documents = load_documents()

    print(f"Documents loaded: {len(documents)}")

    # Step 2: Create chunks
    chunks = create_chunks(documents)

    print(f"Chunks created: {len(chunks)}")

    # Step 3: Generate embeddings
    embeddings = generate_embeddings(chunks)

    print("\nEmbedding information:")
    print("Shape:", embeddings.shape)
    print("Data type:", embeddings.dtype)

    # Step 4: Save embeddings and metadata
    save_embeddings(embeddings, chunks)

    print("\nFirst embedding:")
    print(embeddings[0][:10])

    print("\nEmbedding generation completed!")
