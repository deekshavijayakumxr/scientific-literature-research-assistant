import re

from .retriever import retrieve_papers
from .generator import generate_answer


# --------------------------------------------------
# Query preprocessing
# --------------------------------------------------

def preprocess_question(question):
    """
    Clean the research question before retrieval.

    This does not attempt to automatically correct
    scientific terminology. Spelling normalization
    is handled separately in the Streamlit layer.
    """

    if not question or not question.strip():

        raise ValueError(
            "Research question cannot be empty."
        )

    cleaned = " ".join(
        question.strip().split()
    )

    return cleaned


# --------------------------------------------------
# Sentence splitting
# --------------------------------------------------

def split_into_sentences(text):
    """
    Split abstract text into reasonably clean
    sentences.
    """

    text = text.replace(
        "\n",
        " "
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    if not text:

        return []

    # Basic sentence splitting while avoiding
    # extremely small fragments.
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    cleaned_sentences = []

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:

            continue

        # Remove trailing punctuation noise.
        sentence = sentence.rstrip()

        if len(sentence.split()) < 8:

            continue

        # Skip obvious extraction artifacts.
        if sentence.endswith(
            (
                "-",
                "[",
                "(",
                "/",
                "…",
                "..."
            )
        ):

            continue

        if sentence.endswith(
            "-e"
        ):

            continue

        cleaned_sentences.append(
            sentence
        )

    return cleaned_sentences


# --------------------------------------------------
# Normalize text for duplicate detection
# --------------------------------------------------

def normalize_for_comparison(text):
    """
    Normalize text so repeated evidence can be detected.
    """

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    return text.strip()


# --------------------------------------------------
# Extract evidence
# --------------------------------------------------

def extract_evidence(
    retrieved_papers,
    question,
    top_evidence=10
):
    """
    Extract the strongest and most diverse
    evidence sentences from retrieved papers.
    """

    from sklearn.metrics.pairwise import cosine_similarity
    from .retriever import embedding_model


    if retrieved_papers is None:

        return []


    if retrieved_papers.empty:

        return []


    # --------------------------------------------------
    # Embed research question
    # --------------------------------------------------

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )


    evidence_candidates = []


    # --------------------------------------------------
    # Extract candidate evidence from papers
    # --------------------------------------------------

    for paper_number, (_, paper) in enumerate(
        retrieved_papers.iterrows(),
        start=1
    ):

        abstract = str(
            paper.get(
                "abstract",
                ""
            )
        ).strip()


        if not abstract:

            continue


        sentences = split_into_sentences(
            abstract
        )


        if not sentences:

            continue


        # --------------------------------------------------
        # Embed candidate sentences
        # --------------------------------------------------

        sentence_embeddings = embedding_model.encode(
            sentences,
            normalize_embeddings=True
        )


        similarities = cosine_similarity(
            query_embedding,
            sentence_embeddings
        )[0]


        # --------------------------------------------------
        # Store candidates
        # --------------------------------------------------

        for sentence, similarity in zip(
            sentences,
            similarities
        ):

            evidence_candidates.append(
                {
                    "paper_number": paper_number,

                    "title": paper.get(
                        "title",
                        ""
                    ),

                    "year": paper.get(
                        "publication_year",
                        ""
                    ),

                    "sentence": sentence,

                    "similarity": float(
                        similarity
                    )
                }
            )


    # --------------------------------------------------
    # Rank candidates
    # --------------------------------------------------

    evidence_candidates.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )


    # --------------------------------------------------
    # Select evidence
    # --------------------------------------------------

    selected_evidence = []

    paper_counts = {}

    seen_sentences = set()


    # Slightly lower threshold than the old 0.60
    # so valid evidence is not discarded too aggressively.
    MIN_SIMILARITY = 0.52

    MAX_SENTENCES_PER_PAPER = 2


    for item in evidence_candidates:

        paper_number = item[
            "paper_number"
        ]

        similarity = item[
            "similarity"
        ]


        if similarity < MIN_SIMILARITY:

            continue


        # ----------------------------------------------
        # Duplicate protection
        # ----------------------------------------------

        normalized_sentence = normalize_for_comparison(
            item["sentence"]
        )


        if normalized_sentence in seen_sentences:

            continue


        # ----------------------------------------------
        # Prevent one paper dominating
        # ----------------------------------------------

        current_count = paper_counts.get(
            paper_number,
            0
        )


        if current_count >= MAX_SENTENCES_PER_PAPER:

            continue


        selected_evidence.append(
            item
        )


        seen_sentences.add(
            normalized_sentence
        )


        paper_counts[
            paper_number
        ] = current_count + 1


        if len(selected_evidence) >= top_evidence:

            break


    # --------------------------------------------------
    # Fallback if threshold was too restrictive
    # --------------------------------------------------

    if not selected_evidence:

        fallback_count = 0

        seen_sentences = set()

        paper_counts = {}


        for item in evidence_candidates:

            normalized_sentence = normalize_for_comparison(
                item["sentence"]
            )


            if normalized_sentence in seen_sentences:

                continue


            paper_number = item[
                "paper_number"
            ]


            current_count = paper_counts.get(
                paper_number,
                0
            )


            if current_count >= 2:

                continue


            selected_evidence.append(
                item
            )


            seen_sentences.add(
                normalized_sentence
            )


            paper_counts[
                paper_number
            ] = current_count + 1


            fallback_count += 1


            if fallback_count >= min(
                top_evidence,
                5
            ):

                break


    # --------------------------------------------------
    # Terminal display
    # --------------------------------------------------

    print(
        "\nSELECTED EVIDENCE"
    )

    print(
        "=" * 70
    )


    for item in selected_evidence:

        print(
            f"[{item['paper_number']}] "
            f"Similarity: "
            f"{item['similarity']:.3f}\n"
            f"{item['sentence']}\n"
        )


    return selected_evidence


# --------------------------------------------------
# Build evidence text
# --------------------------------------------------

def build_evidence_text(
    evidence
):
    """
    Build clearly numbered evidence
    for the LLM.
    """

    evidence_blocks = []


    for item in evidence:

        block = (
            f"[{item['paper_number']}] "
            f"{item['title']} "
            f"({item['year']})\n"
            f"- {item['sentence']}"
        )

        evidence_blocks.append(
            block
        )


    return "\n\n".join(
        evidence_blocks
    )


# --------------------------------------------------
# Complete RAG pipeline
# --------------------------------------------------

def answer_research_question(
    question,
    top_papers=8,
    top_evidence=10
):
    """
    Complete research pipeline:

    Question
        ↓
    Query preprocessing
        ↓
    Semantic paper retrieval
        ↓
    Sentence-level evidence extraction
        ↓
    Evidence filtering + deduplication
        ↓
    LLM generation
        ↓
    Sources
    """


    # --------------------------------------------------
    # 0. Validate + preprocess
    # --------------------------------------------------

    question = preprocess_question(
        question
    )


    # --------------------------------------------------
    # 1. Retrieve papers
    # --------------------------------------------------

    retrieved_papers = retrieve_papers(
        question,
        top_k=top_papers
    )


    # --------------------------------------------------
    # 2. Extract evidence
    # --------------------------------------------------

    evidence = extract_evidence(
        retrieved_papers,
        question,
        top_evidence=top_evidence
    )


    # --------------------------------------------------
    # 3. Handle missing evidence
    # --------------------------------------------------

    if not evidence:

        raise RuntimeError(
            "No usable evidence was found in "
            "the retrieved papers."
        )


    # --------------------------------------------------
    # 4. Build evidence for Llama
    # --------------------------------------------------

    evidence_text = build_evidence_text(
        evidence
    )


    # --------------------------------------------------
    # 5. Generate answer
    # --------------------------------------------------

    answer = generate_answer(
        question,
        evidence_text
    )


    # --------------------------------------------------
    # 6. Build source list
    # --------------------------------------------------

    sources = []


    for number, (_, paper) in enumerate(
        retrieved_papers.iterrows(),
        start=1
    ):

        sources.append(
            {
                "number": number,

                "title": paper.get(
                    "title",
                    ""
                ),

                "year": paper.get(
                    "publication_year",
                    ""
                ),

                "citations": paper.get(
                    "cited_by_count",
                    0
                ),

                "similarity": float(
                    paper["similarity"]
                )
            }
        )


    # --------------------------------------------------
    # 7. Return structured result
    # --------------------------------------------------

    return {
        "question": question,

        "answer": answer,

        "sources": sources,

        "evidence": evidence
    }


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    question = (
        "How is deep learning used "
        "for medical image analysis?"
    )


    print(
        "\nRunning RAG pipeline...\n"
    )


    result = answer_research_question(
        question,
        top_papers=8,
        top_evidence=10
    )


    print(
        "=" * 70
    )

    print(
        "RESEARCH ANSWER"
    )

    print(
        "=" * 70
    )

    print(
        result["answer"]
    )


    print(
        "\n"
    )

    print(
        "=" * 70
    )

    print(
        "SOURCES"
    )

    print(
        "=" * 70
    )


    for source in result["sources"]:

        print(
            f"[{source['number']}] "
            f"{source['title']} "
            f"({source['year']}) "
            f"Similarity: "
            f"{source['similarity']:.3f}"
        )