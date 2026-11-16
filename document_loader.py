
import json
from pathlib import Path


# --------------------------------
# PROJECT CONFIGURATION
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "knowledge_base.json"


# --------------------------------
# JSON DOCUMENT LOADER
# --------------------------------

def load_json_documents():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Knowledge base not found: {DATA_PATH}"
        )

    with open(DATA_PATH, "r", encoding="utf-8") as file:

        documents = json.load(file)

    if not isinstance(documents, list):

        raise ValueError(
            "Expected a list of documents in JSON."
        )

    required_fields = {
        "doc_id",
        "category",
        "title",
        "content"
    }

    seen_ids = set()

    for index, document in enumerate(documents):

        if not isinstance(document, dict):

            raise ValueError(
                f"Record {index} must be a dictionary."
            )

        missing_fields = required_fields - document.keys()

        if missing_fields:

            raise ValueError(
                f"Record {index} is missing fields: "
                f"{missing_fields}"
            )

        for field in required_fields:

            if (
                not isinstance(document[field], str)
                or not document[field].strip()
            ):

                raise ValueError(
                    f"Invalid field '{field}' in record {index}."
                )

        if document["doc_id"] in seen_ids:

            raise ValueError(
                f"Duplicate document ID: {document['doc_id']}"
            )

        seen_ids.add(document["doc_id"])

        document["source_type"] = "json"
        document["source_file"] = "knowledge_base.json"

    print(f"JSON documents loaded: {len(documents)}")

    return documents


# --------------------------------
# JSON DOCUMENT PIPELINE
# --------------------------------

def load_documents():

    print("=" * 60)
    print("JSON INSURANCE DOCUMENT LOADING PIPELINE")
    print("=" * 60)

    documents = load_json_documents()

    print("\n" + "=" * 60)
    print("DOCUMENT LOADING SUMMARY")
    print("=" * 60)

    print("Total JSON documents:", len(documents))

    return documents


# --------------------------------
# TESTING
# --------------------------------

if __name__ == "__main__":

    documents = load_documents()

    print("\nDocument details:")

    for document in documents:

        print("\n" + "-" * 50)

        print("Document ID:", document["doc_id"])
        print("Title:", document["title"])
        print("Category:", document["category"])
        print("Source Type:", document["source_type"])
        print("Source File:", document["source_file"])

        print("Characters:", len(document["content"]))

    print("\nJSON document loading completed successfully!")
