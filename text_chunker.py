
from src.document_loader import load_documents


def create_chunks(documents, chunk_size=120, overlap=20):
    """
    Split documents into smaller text chunks.

    chunk_size: Maximum number of words in a chunk.
    overlap: Number of words repeated between consecutive chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    chunks = []

    for document in documents:

        text = (
            document["title"]
            + ". "
            + document["content"]
        )

        words = text.split()

        start = 0
        chunk_number = 1

        while start < len(words):

            end = start + chunk_size

            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunk = {
                "chunk_id": (
                    document["doc_id"]
                    + "_CHUNK_"
                    + str(chunk_number)
                ),
                "doc_id": document["doc_id"],
                "category": document["category"],
                "title": document["title"],
                "source_type": document["source_type"],
                "source_file": document["source_file"],
                "text": chunk_text
            }

            chunks.append(chunk)

            chunk_number += 1

            if end >= len(words):
                break

            start = end - overlap

    return chunks


if __name__ == "__main__":

    documents = load_documents()

    chunks = create_chunks(documents)

    print("=" * 50)
    print("INSURANCE TEXT CHUNKING")
    print("=" * 50)

    print(f"Total documents: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")

    print("\nFirst chunk metadata:")

    for field in [
        "chunk_id",
        "doc_id",
        "category",
        "title",
        "source_type",
        "source_file"
    ]:
        print(f"{field}: {chunks[0][field]}")

    print("\nFirst chunk text preview:")
    print(chunks[0]["text"][:300])

    print("\nChunking completed successfully!")
