
import numpy as np
import faiss

from pathlib import Path


# --------------------------------
# PROJECT CONFIGURATION
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

EMBEDDING_PATH = PROCESSED_DIR / "embeddings.npy"
INDEX_PATH = VECTORSTORE_DIR / "insurance_faiss.index"


# --------------------------------
# BUILD FAISS INDEX
# --------------------------------

def build_faiss_index():

    print("Loading embeddings...")

    embeddings = np.load(EMBEDDING_PATH)

    embeddings = np.ascontiguousarray(
        embeddings.astype("float32")
    )

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid values.")

    print("Embedding shape:", embeddings.shape)

    dimension = embeddings.shape[1]

    print("\nCreating FAISS index...")

    # Inner product on normalized embeddings
    # is equivalent to cosine similarity.
    index = faiss.IndexFlatIP(dimension)

    print("Adding embeddings to FAISS...")

    index.add(embeddings)

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Saving FAISS index...")

    faiss.write_index(
        index,
        str(INDEX_PATH)
    )

    print("\nFAISS index created successfully!")

    print("Total vectors:", index.ntotal)
    print("Vector dimensions:", index.d)
    print("Index path:", INDEX_PATH)


if __name__ == "__main__":

    build_faiss_index()
