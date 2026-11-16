
from src.retriever import InsuranceRetriever


# --------------------------------
# EVALUATION SETTINGS
# --------------------------------

TOP_K = 5
SIMILARITY_THRESHOLD = 0.30


# --------------------------------
# TEST DATASET
# Expected document IDs are our
# manually verified ground truth.
# --------------------------------

TEST_CASES = [
    {
        "question": "What expenses are covered during hospitalization?",
        "expected_doc_ids": ["KB001"]
    },
    {
        "question": "What expenses are covered before and after hospitalization?",
        "expected_doc_ids": ["KB002"]
    },
    {
        "question": "Which treatments are covered as day care procedures?",
        "expected_doc_ids": ["KB003"]
    },
    {
        "question": "How much road ambulance expense is reimbursed?",
        "expected_doc_ids": ["KB004"]
    },
    {
        "question": "What is the waiting period for pre-existing diseases?",
        "expected_doc_ids": ["KB005"]
    },
    {
        "question": "What is the initial waiting period for illnesses?",
        "expected_doc_ids": ["KB006"]
    },
    {
        "question": "Which specific diseases have a 24-month waiting period?",
        "expected_doc_ids": ["KB007"]
    },
    {
        "question": "Which treatments are permanently excluded?",
        "expected_doc_ids": ["KB008"]
    },
    {
        "question": "How does the cashless claim process work?",
        "expected_doc_ids": ["KB009"]
    },
    {
        "question": "How can I submit a reimbursement claim?",
        "expected_doc_ids": ["KB010"]
    },
    {
        "question": "How long does the insurer take to settle a claim?",
        "expected_doc_ids": ["KB011"]
    },
    {
        "question": "What are the common reasons for claim rejection?",
        "expected_doc_ids": ["KB012"]
    },
    {
        "question": "Which factors affect health insurance premiums?",
        "expected_doc_ids": ["KB013"]
    },
    {
        "question": "What is the grace period for policy renewal?",
        "expected_doc_ids": ["KB014"]
    },
    {
        "question": "What is a No Claim Bonus?",
        "expected_doc_ids": ["KB015"]
    },
    {
        "question": "Can I transfer my health insurance to another insurer?",
        "expected_doc_ids": ["KB016"]
    },
    {
        "question": "What does the maternity cover add-on provide?",
        "expected_doc_ids": ["KB017"]
    },
    {
        "question": "What is a critical illness rider?",
        "expected_doc_ids": ["KB018"]
    },
    {
        "question": "How can I find network hospitals?",
        "expected_doc_ids": ["KB019"]
    },
    {
        "question": "What is the free look period in health insurance?",
        "expected_doc_ids": ["KB020"]
    }
]


# --------------------------------
# EVALUATION METRICS
# --------------------------------

def calculate_metrics(retrieved_ids, expected_ids):

    retrieved_set = set(retrieved_ids)
    expected_set = set(expected_ids)

    relevant_count = len(retrieved_set & expected_set)

    precision = relevant_count / TOP_K

    recall = relevant_count / len(expected_set)

    hit_rate = 1 if relevant_count > 0 else 0

    reciprocal_rank = 0

    for rank, doc_id in enumerate(retrieved_ids, start=1):

        if doc_id in expected_set:
            reciprocal_rank = 1 / rank
            break

    return precision, recall, hit_rate, reciprocal_rank


# --------------------------------
# MAIN EVALUATION
# --------------------------------

def main():

    print("\nInitializing Insurance Retriever...")

    retriever = InsuranceRetriever()

    precision_scores = []
    recall_scores = []
    hit_scores = []
    reciprocal_ranks = []

    print("\nStarting Retrieval Evaluation")
    print("=" * 80)

    for number, test in enumerate(TEST_CASES, start=1):

        question = test["question"]
        expected_ids = test["expected_doc_ids"]

        results = retriever.search(
            question,
            top_k=TOP_K,
            similarity_threshold=SIMILARITY_THRESHOLD
        )

        retrieved_ids = [
            result["doc_id"] for result in results
        ]

        precision, recall, hit, mrr = calculate_metrics(
            retrieved_ids,
            expected_ids
        )

        precision_scores.append(precision)
        recall_scores.append(recall)
        hit_scores.append(hit)
        reciprocal_ranks.append(mrr)

        print(f"\nTEST {number}: {question}")

        print("Expected:", expected_ids)
        print("Retrieved:", retrieved_ids)

        print(f"Precision@{TOP_K}: {precision:.2f}")
        print(f"Recall@{TOP_K}:    {recall:.2f}")
        print(f"Hit Rate@{TOP_K}:  {hit}")
        print(f"MRR@{TOP_K}:       {mrr:.2f}")

        if hit:
            print("Result: PASS")
        else:
            print("Result: MISS")

    # --------------------------------
    # FINAL EVALUATION SUMMARY
    # --------------------------------

    total = len(TEST_CASES)

    avg_precision = sum(precision_scores) / total
    avg_recall = sum(recall_scores) / total
    avg_hit_rate = sum(hit_scores) / total
    avg_mrr = sum(reciprocal_ranks) / total

    print("\n")
    print("=" * 80)
    print("FINAL RETRIEVAL EVALUATION REPORT")
    print("=" * 80)

    print(f"Total test questions: {total}")
    print(f"Top K: {TOP_K}")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")

    print(f"\nAverage Precision@{TOP_K}: {avg_precision:.4f}")
    print(f"Average Recall@{TOP_K}:    {avg_recall:.4f}")
    print(f"Hit Rate@{TOP_K}:          {avg_hit_rate:.4f}")
    print(f"Mean Reciprocal Rank:      {avg_mrr:.4f}")

    print("\nRetrieval evaluation completed!")


if __name__ == "__main__":
    main()
