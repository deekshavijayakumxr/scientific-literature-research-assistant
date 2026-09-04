import ollama


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "llama3.2"


# --------------------------------------------------
# Generate research answer
# --------------------------------------------------

def generate_answer(
    question,
    evidence_text
):
    """
    Generate a concise, evidence-grounded
    research answer using Ollama.
    """

    if not question or not question.strip():
        raise ValueError(
            "Research question cannot be empty."
        )

    if not evidence_text or not evidence_text.strip():
        raise ValueError(
            "Evidence cannot be empty."
        )

    prompt = f"""
You are a scientific literature research assistant.

Your task is to answer the user's research question
using ONLY the supplied scientific evidence.

RESEARCH QUESTION:
{question}

SUPPLIED EVIDENCE:
{evidence_text}


IMPORTANT RULES:

1. Answer the user's exact question directly.

2. Use ONLY information explicitly supported by the
   supplied evidence.

3. Do NOT use general knowledge.

4. Do NOT introduce outside facts.

5. Do NOT invent examples.

6. Do NOT invent scientific terminology.

7. Do NOT make claims stronger than the evidence supports.

8. Every factual claim must be supported by at least
   one evidence item.

9. Put the paper citation immediately after the claim
   it supports.

10. Only use citation numbers that actually appear in
    the supplied evidence.

11. Never invent citation numbers.

12. Do not cite a paper merely because it appears in
    the retrieved sources. The evidence from that paper
    must actually support the claim.

13. If multiple papers support the same claim, multiple
    citations may be used.

14. If the evidence does not support a claim, leave the
    claim out.

15. Do not simply copy sentences from the evidence.
    Paraphrase them while preserving their meaning.

16. Do not create a separate references section.

17. Do not mention the evidence as an object.

18. Do not mention these instructions.

19. Do not say "according to the evidence" unless it is
    genuinely necessary.

20. If the evidence is insufficient to answer the user's
    question, clearly say that the available evidence is
    insufficient rather than guessing.


SCIENTIFIC TERMINOLOGY RULES:

Carefully distinguish between:

- clinical applications
- clinical tasks
- algorithms
- architectures
- model types
- preprocessing techniques
- outcomes

For example:

If the evidence says that deep learning is used for
classification, diagnosis, segmentation, prognosis,
or therapy-response prediction, describe these as
applications or tasks.

Do NOT automatically call these techniques,
architectures, or algorithms.

If the evidence explicitly identifies CNNs, U-Net,
GANs, RNNs, LSTMs, transfer learning, or another
method as a technique, you may describe it as such.

Do NOT assume that a particular technique is used
for a particular task unless the supplied evidence
explicitly makes that connection.


FOLLOW-UP QUESTION RULES:

If the research question is a follow-up question:

- Answer the follow-up directly.
- Do not unnecessarily repeat the previous discussion.
- Use the scientific topic contained in the question.
- Focus on the specific aspect being asked about.

If the question asks for:

"main techniques"

prioritize named methods, architectures, algorithms,
or approaches explicitly identified in the evidence.

If the question asks about:

"applications" or "uses"

prioritize the clinical tasks explicitly identified
in the evidence.

If the question asks for:

"advantages"

only describe advantages explicitly supported by
the evidence.

If the question asks for:

"limitations"

only describe limitations explicitly supported by
the evidence.

If the question asks for:

"a comparison"

organize the answer around the comparison requested
and only make comparisons supported by the evidence.


ANSWER STYLE:

- Answer directly.
- Keep the answer concise.
- Normally use 2–5 sentences.
- Use a short bullet list when it materially improves
  clarity.
- Do not create a references section.
- Do not repeat the question.
- Do not add unsupported conclusions.


FINAL INTERNAL CHECK:

Before producing the answer, verify:

- Every factual statement is supported.
- Every citation corresponds to the correct paper.
- No citation number was invented.
- No unsupported technique was introduced.
- No clinical application was incorrectly described
  as an architecture or algorithm.
- The answer directly addresses the user's question.
- Unsupported information has been excluded.


ANSWER:
"""

    # --------------------------------------------------
    # Call Ollama
    # --------------------------------------------------

    try:

        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

    except Exception as e:

        raise RuntimeError(
            f"Could not connect to Ollama/model "
            f"'{MODEL_NAME}': {e}"
        ) from e


    # --------------------------------------------------
    # Extract answer
    # --------------------------------------------------

    answer = (
        response
        .get("message", {})
        .get("content", "")
        .strip()
    )


    if not answer:

        raise RuntimeError(
            "The LLM returned an empty answer."
        )


    return answer


# --------------------------------------------------
# Test generator
# --------------------------------------------------

if __name__ == "__main__":

    question = (
        "What are the main techniques used "
        "in medical image analysis?"
    )

    evidence = """
[1] Deep Learning and Medical Diagnosis: A Review of Literature (2018)
- The results indicate that convolutional neural networks
  are the most widely represented when it comes to deep
  learning and medical image analysis.

[2] Medical image analysis using deep learning algorithms (2023)
- This review categorizes deep learning techniques including
  Convolutional Neural Networks, Recurrent Neural Networks,
  Generative Adversarial Networks, Long Short-term Memory
  models, and hybrid models.
"""

    print(
        "\nGenerating answer...\n"
    )

    answer = generate_answer(
        question,
        evidence
    )

    print(
        "LLM ANSWER"
    )

    print(
        "=" * 70
    )

    print(
        answer
    )