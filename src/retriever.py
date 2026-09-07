from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PAPERS_PATH = PROJECT_ROOT / "data" / "processed" / "papers_clean.csv"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "paper_embeddings.npy"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5

# Hugging Face dataset
HF_BASE_URL = (
    "https://huggingface.co/datasets/"
    "deekshavijayakumxr/"
    "scientific-literature-research-assistant-data/"
    "resolve/main/"
)


# ---------------------------------------------------------
# Download dataset files if they are missing
# ---------------------------------------------------------

def download_file(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {destination.name}...")

    try:
        urllib.request.urlretrieve(url, destination)
    except Exception as e:
        if destination.exists():
            destination.unlink()

        raise RuntimeError(
            f"Could not download {destination.name} from Hugging Face: {e}"
        ) from e


def ensure_data_files():
    if not PAPERS_PATH.exists():
        download_file(
            HF_BASE_URL + "papers_clean.csv",
            PAPERS_PATH
        )

    if not EMBEDDINGS_PATH.exists():
        download_file(
            HF_BASE_URL + "paper_embeddings.npy",
            EMBEDDINGS_PATH
        )


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

ensure_data_files()

try:
    PAPERS = pd.read_csv(PAPERS_PATH)
except Exception as e:
    raise RuntimeError(
        f"Could not load paper database from '{PAPERS_PATH}': {e}"
    ) from e


try:
    EMBEDDINGS = np.load(EMBEDDINGS_PATH)
except Exception as e:
    raise RuntimeError(
        f"Could not load paper embeddings from '{EMBEDDINGS_PATH}': {e}"
    ) from e


if len(PAPERS) != len(EMBEDDINGS):
    raise RuntimeError(
        f"Paper/embedding mismatch: "
        f"{len(PAPERS)} papers but {len(EMBEDDINGS)} embeddings."
    )


# ---------------------------------------------------------
# Embedding model
# ---------------------------------------------------------

MODEL = SentenceTransformer(MODEL_NAME)


# ---------------------------------------------------------
# Retrieval
# ---------------------------------------------------------

def retrieve_papers(question, top_k=DEFAULT_TOP_K):
    """
    Retrieve the most semantically similar papers for a question.
    """

    query_embedding = MODEL.encode(
        [question],
        normalize_embeddings=True
    )

    similarities = cosine_similarity(
        query_embedding,
        EMBEDDINGS
    )[0]

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = PAPERS.iloc[top_indices].copy()
    results["similarity"] = similarities[top_indices]

    return results.reset_index(drop=True)


# ---------------------------------------------------------
# Terminal test
# ---------------------------------------------------------

if __name__ == "__main__":
    question = "How is deep learning used in medical imaging?"

    results = retrieve_papers(question, top_k=5)

    print("\nTop retrieved papers:\n")

    for i, row in results.iterrows():
        print(
            f"{i + 1}. {row.get('title', 'Unknown title')}"
            f" | similarity={row['similarity']:.4f}"
        )
