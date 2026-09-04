import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Paths
# --------------------------------------------------

PAPERS_PATH = "data/processed/papers_clean.csv"
EMBEDDINGS_PATH = "data/processed/paper_embeddings.npy"


# --------------------------------------------------
# Embedding model
# --------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------
# Retrieval configuration
# --------------------------------------------------

DEFAULT_TOP_K = 5


# --------------------------------------------------
# Load paper database
# --------------------------------------------------

print("Loading paper database...")

try:

    papers = pd.read_csv(
        PAPERS_PATH
    )

except Exception as e:

    raise RuntimeError(
        f"Could not load paper database "
        f"from '{PAPERS_PATH}': {e}"
    ) from e


# --------------------------------------------------
# Load embeddings
# --------------------------------------------------

try:

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

except Exception as e:

    raise RuntimeError(
        f"Could not load paper embeddings "
        f"from '{EMBEDDINGS_PATH}': {e}"
    ) from e


print(
    f"Loaded {len(papers)} papers"
)

print(
    f"Embeddings shape: {embeddings.shape}"
)


# --------------------------------------------------
# Validate database
# --------------------------------------------------

if len(papers) != len(embeddings):

    raise RuntimeError(
        "The number of papers does not match "
        "the number of embeddings."
    )


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    MODEL_NAME
)

print(
    "Embedding model loaded"
)


# --------------------------------------------------
# Retrieve relevant papers
# --------------------------------------------------

def retrieve_papers(
    question,
    top_k=DEFAULT_TOP_K
):
    """
    Retrieve the most semantically relevant
    scientific papers for a research question.

    Returns a pandas DataFrame containing the
    retrieved papers and their similarity scores.
    """

    # --------------------------------------------------
    # Validate question
    # --------------------------------------------------

    if not question or not question.strip():

        raise ValueError(
            "Research question cannot be empty."
        )


    # --------------------------------------------------
    # Validate top_k
    # --------------------------------------------------

    if top_k is None:

        top_k = DEFAULT_TOP_K


    try:

        top_k = int(top_k)

    except (TypeError, ValueError):

        raise ValueError(
            "top_k must be an integer."
        )


    if top_k <= 0:

        raise ValueError(
            "top_k must be greater than zero."
        )


    # --------------------------------------------------
    # Prevent requesting more papers than exist
    # --------------------------------------------------

    top_k = min(
        top_k,
        len(papers)
    )


    # --------------------------------------------------
    # Create embedding for the question
    # --------------------------------------------------

    query_embedding = embedding_model.encode(
        [question.strip()],
        normalize_embeddings=True
    )


    # --------------------------------------------------
    # Compare question against paper embeddings
    # --------------------------------------------------

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]


    # --------------------------------------------------
    # Get highest-scoring papers
    # --------------------------------------------------

    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]


    # --------------------------------------------------
    # Build result DataFrame
    # --------------------------------------------------

    results = papers.iloc[
        top_indices
    ].copy()


    # --------------------------------------------------
    # Add similarity score
    # --------------------------------------------------

    results["similarity"] = similarities[
        top_indices
    ]


    # --------------------------------------------------
    # Reset index
    # --------------------------------------------------

    results = results.reset_index(
        drop=True
    )


    return results


# --------------------------------------------------
# Test retrieval
# --------------------------------------------------

if __name__ == "__main__":

    question = (
        "How is deep learning used for "
        "medical image analysis?"
    )

    results = retrieve_papers(
        question,
        top_k=5
    )

    print(
        "\nTOP PAPERS\n"
    )

    print(
        "=" * 70
    )

    for i, (_, paper) in enumerate(
        results.iterrows(),
        start=1
    ):

        print(
            f"[{i}] "
            f"{paper.get('title', '')} "
            f"({paper.get('publication_year', '')}) "
            f"Similarity: "
            f"{paper['similarity']:.3f}"
        )