# 🔬 Scientific Literature Research Assistant

> **An AI-powered research assistant for exploring 10,000+ scientific papers using Retrieval-Augmented Generation (RAG).**

## 🚀 LIVE DEMO

### 👉 [TRY THE SCIENTIFIC LITERATURE RESEARCH ASSISTANT](https://scientific-literature-research-assistant-ec9d8kvnqim565dyeibim.streamlit.app/)

**Ask a research question → retrieve relevant literature → extract evidence → generate a citation-grounded answer.**

---

## 📌 Overview

The **Scientific Literature Research Assistant** is an end-to-end AI application designed to help users explore scientific literature through natural-language questions.

Instead of relying solely on an LLM's internal knowledge, the system retrieves relevant scientific papers from a corpus of **10,260 papers**, identifies the most relevant evidence from those papers, and provides that evidence to the language model before generating an answer.

The application also supports **context-aware conversational research**, allowing users to ask follow-up questions without restating the entire research topic.

### Example

```text
User:
How is deep learning used in medical imaging?

Assistant:
[Research answer grounded in retrieved scientific evidence]

User:
What about diagnosis specifically?

Assistant:
[Follow-up answer using the previous conversation context]
```

---

# 🧠 End-to-End System Architecture

```text
                         USER QUESTION
                              │
                              ▼
                    Query Preprocessing
                              │
                              ▼
                 Conversational Query
                       Rewriting
                              │
                              ▼
                 Sentence Transformer
                 all-MiniLM-L6-v2
                              │
                              ▼
                    Query Embedding
                              │
                              ▼
                    Semantic Retrieval
                              │
                              ▼
                 Top Relevant Papers
                              │
                              ▼
                Sentence-Level Evidence
                       Extraction
                              │
                              ▼
                 Evidence Similarity
                       Ranking
                              │
                              ▼
                Evidence Filtering &
                    Deduplication
                              │
                              ▼
                   Evidence Context
                              │
                              ▼
                    LLM Generation
                              │
                              ▼
                Citation-Grounded Answer
                              │
                              ▼
                     Streamlit UI
                              │
                              ▼
                   Conversation History
```

---

# 🔄 Project Development & LLM Migration

One of the important parts of this project was separating the **RAG pipeline** from the **LLM inference layer**.

This allowed the retrieval architecture to remain stable while the inference infrastructure changed from a local prototype to a cloud deployment.

## Phase 1 — Local RAG Prototype

The project was initially developed and tested locally using:

```text
Llama 3.2
    │
    ▼
Ollama
    │
    ▼
Local LLM Inference
```

Ollama provided a convenient way to run the Llama model locally while developing the application.

During this phase, the RAG system was built and validated, including:

* Scientific paper retrieval
* Sentence Transformer embeddings
* Semantic similarity search
* Sentence-level evidence extraction
* Evidence filtering
* Evidence deduplication
* Citation-grounded prompting
* Conversational query rewriting
* Streamlit application development
* Persistent chat history using SQLite

The goal was to validate the **retrieval and generation workflow locally before deployment**.

---

## Phase 2 — Preparing the Application for Cloud Deployment

Once the RAG pipeline was working, the application needed to be deployed as a publicly accessible web application.

The local Ollama setup was tied to the development environment and was therefore not suitable as the inference layer for a public cloud application.

Instead of rebuilding the RAG pipeline, the system was structured so that the **LLM generation component could be replaced independently**.

This meant the following components could remain unchanged:

```text
Paper Dataset
      ↓
Embeddings
      ↓
Semantic Retrieval
      ↓
Evidence Extraction
      ↓
Evidence Filtering
      ↓
RAG Context
```

Only the final generation component needed to change.

---

## Phase 3 — Cloud LLM Inference

The local inference layer was migrated from:

```text
Llama 3.2
     ↓
Ollama
     ↓
Local inference
```

to:

```text
GPT-OSS-120B
     ↓
Groq API
     ↓
Hosted inference
```

The RAG architecture itself remained the same.

This separation makes the system more flexible because the retrieval layer does not depend on a specific LLM provider.

### Final Architecture

```text
                  RAG SYSTEM
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
   RETRIEVAL LAYER         GENERATION LAYER
          │                       │
   Paper embeddings        Cloud LLM inference
   Semantic search                 │
   Evidence extraction             │
   Evidence ranking                │
   Deduplication           GPT-OSS-120B
          │                       │
          │                    Groq API
          │                       │
          └───────────┬───────────┘
                      ▼
             Citation-Grounded
                   Answer
```

---

# 🔍 How the RAG Pipeline Works

## 1. User Question

The user enters a natural-language research question through the Streamlit interface.

For example:

```text
How is deep learning used in medical imaging?
```

---

## 2. Conversational Query Rewriting

If the user asks a follow-up question, the system uses the previous conversation to understand the context.

For example:

```text
Question 1:
How is deep learning used in medical imaging?

Question 2:
What about diagnosis specifically?
```

The second question depends on the first question for its meaning.

The application therefore uses an LLM-based query-rewriting step to transform follow-up questions into standalone research queries before retrieval.

This allows the semantic search system to retrieve the correct literature for the conversation.

---

## 3. Query Embedding

The standalone research question is converted into a vector representation using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The same embedding model was used to create embeddings for the scientific paper corpus.

The resulting vectors represent the semantic meaning of the text.

---

## 4. Semantic Paper Retrieval

The query embedding is compared with the precomputed paper embeddings using **cosine similarity**.

The system retrieves the papers with the highest semantic similarity to the research question.

This allows the application to find relevant literature even when the user's wording does not exactly match the wording used in the papers.

---

## 5. Sentence-Level Evidence Extraction

Rather than providing entire retrieved documents to the LLM, the application breaks the retrieved paper abstracts into individual sentences.

Each sentence is compared with the research question using the same embedding-based similarity approach.

The most relevant sentences are selected as evidence.

This produces a more focused context for the language model.

---

## 6. Evidence Filtering

Candidate evidence is filtered using a similarity threshold.

The system also limits the number of selected evidence sentences from each paper.

This helps prevent the final context from being dominated by a single source.

---

## 7. Evidence Deduplication

Highly similar evidence sentences are removed so that the final context contains useful and distinct information.

The resulting evidence set is then passed to the generation layer.

---

## 8. Citation-Grounded Generation

The selected evidence is provided to the LLM with a strict evidence-grounding prompt.

The generation layer is instructed to:

* Use the supplied evidence
* Avoid unsupported claims
* Avoid inventing citations
* Keep citations close to the claims they support
* Distinguish evidence from interpretation
* Keep answers concise
* Acknowledge limitations when supported by the retrieved literature

The final answer is therefore generated from the retrieved research evidence rather than relying entirely on the model's general knowledge.

---

# 📚 Scientific Paper Corpus

The application currently uses:

```text
10,260 scientific papers
10,260 corresponding embeddings
384-dimensional embeddings
```

The paper corpus and embeddings are stored separately from the GitHub source code because the dataset is too large to conveniently maintain inside the repository.

### Dataset

The processed paper dataset and embeddings are hosted on Hugging Face:

👉 [Scientific Literature Research Assistant Dataset](https://huggingface.co/datasets/deekshavijayakumxr/scientific-literature-research-assistant-data)

The dataset contains:

```text
papers_clean.csv
paper_embeddings.npy
```

The application automatically retrieves these files when required.

---

# 🤖 Embedding Model

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model is used for:

1. Creating the paper embeddings
2. Creating query embeddings
3. Comparing research questions with papers
4. Ranking individual evidence sentences

The embedding dimension is:

```text
384
```

---

# 💬 Conversational Research

The application is designed to support multi-turn research conversations.

For example:

```text
User:
How is deep learning used in medical imaging?

Assistant:
[Answer with citations]

User:
What about diagnosis specifically?

Assistant:
[Context-aware answer]

User:
What are the limitations?

Assistant:
[Context-aware answer]
```

Instead of treating every message as an isolated question, the application uses previous conversation context to interpret follow-up questions.

---

# 🗃️ Persistent Chat History

The application uses **SQLite** to persist chat sessions and messages.

The structure is conceptually:

```text
Chats
 │
 └── Messages
       ├── User questions
       └── Assistant responses
```

This allows users to create and continue research conversations within the application.

---

# 🖥️ Streamlit Application

The user interface is built using **Streamlit**.

The application provides:

* Research question input
* Conversational chat interface
* New chat functionality
* Persistent chat history
* Context-aware follow-up questions
* Citation-grounded answers
* Evidence inspection
* Source information

---

# ☁️ Cloud Deployment

The final application is publicly deployed using **Streamlit Community Cloud**.

### Deployment architecture

```text
GitHub Repository
       │
       ▼
Streamlit Community Cloud
       │
       ├── Install Python dependencies
       │
       ├── Load Sentence Transformer
       │
       ├── Retrieve paper dataset
       │
       ├── Load embeddings
       │
       ├── Connect to Groq API
       │
       └── Start Streamlit Application
```

### Runtime data flow

```text
User
 │
 ▼
Streamlit Cloud
 │
 ▼
Query Rewriting
 │
 ▼
Semantic Retrieval
 │
 ▼
Evidence Extraction
 │
 ▼
Groq / GPT-OSS-120B
 │
 ▼
Citation-Grounded Answer
```

The Groq API key is stored as a deployment secret and is not included in the GitHub repository.

---

# 🛠️ Tech Stack

## Programming & Data

* Python
* Pandas
* NumPy
* scikit-learn

## NLP / AI

* Retrieval-Augmented Generation (RAG)
* Large Language Models (LLMs)
* Sentence Transformers
* Text Embeddings
* Semantic Search
* Cosine Similarity
* Evidence Retrieval
* Prompt Engineering

## LLM Inference

### Initial prototype

* Llama 3.2
* Ollama

### Production / cloud deployment

* Groq API
* GPT-OSS-120B

## Application & Storage

* Streamlit
* SQLite

## Data / Deployment

* Hugging Face
* Streamlit Community Cloud
* Git
* GitHub

---

# 📁 Project Structure

```text
scientific-literature-research-assistant/
│
├── app/
│   └── main.py
│
├── src/
│   ├── retriever.py
│   ├── rag.py
│   └── generator.py
│
├── data/
│   └── processed/
│
├── notebooks/
│   └── final_papers_10260.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

### `app/main.py`

Responsible for:

* Streamlit UI
* Chat interface
* Conversation handling
* Follow-up query rewriting
* SQLite chat persistence
* Displaying answers and sources

### `src/retriever.py`

Responsible for:

* Loading the scientific paper corpus
* Loading paper embeddings
* Loading the Sentence Transformer model
* Generating query embeddings
* Performing semantic retrieval

### `src/rag.py`

Responsible for:

* Query preprocessing
* Paper retrieval
* Sentence-level evidence extraction
* Evidence ranking
* Similarity filtering
* Evidence deduplication
* Source construction
* Coordinating the RAG pipeline

### `src/generator.py`

Responsible for:

* Building the evidence-grounded prompt
* Calling the LLM
* Generating the final research answer
* Enforcing evidence-based generation

---

# ⚙️ Run Locally

## 1. Clone the repository

```bash
git clone https://github.com/deekshavijayakumxr/scientific-literature-research-assistant.git
cd scientific-literature-research-assistant
```

## 2. Create a virtual environment

```bash
python -m venv venv
```

### Windows

```powershell
venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure the Groq API key

Windows PowerShell:

```powershell
$env:GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

Do not commit API keys to GitHub.

## 5. Run the application

```bash
streamlit run app\main.py
```

---

# 🔐 Security

API credentials are intentionally excluded from the repository.

The following should never be committed:

```text
GROQ_API_KEY
.env
API tokens
Private credentials
```

The application reads the Groq API key through an environment variable:

```text
GROQ_API_KEY
```

For cloud deployment, the key is stored using Streamlit's secrets management.

---

# 📊 Current Capabilities

| Capability                         | Status |
| ---------------------------------- | ------ |
| 10K+ scientific papers             | ✅      |
| 10,260 paper corpus                | ✅      |
| Semantic paper retrieval           | ✅      |
| Sentence Transformer embeddings    | ✅      |
| Sentence-level evidence extraction | ✅      |
| Evidence similarity ranking        | ✅      |
| Evidence filtering                 | ✅      |
| Evidence deduplication             | ✅      |
| Citation-grounded generation       | ✅      |
| Conversational query rewriting     | ✅      |
| Context-aware follow-ups           | ✅      |
| Persistent chat history            | ✅      |
| SQLite storage                     | ✅      |
| Streamlit interface                | ✅      |
| Cloud LLM inference                | ✅      |
| Hugging Face dataset hosting       | ✅      |
| Public cloud deployment            | ✅      |

---

# 🚧 Future Improvements

Potential improvements include:

* Full-text paper retrieval instead of abstract-only evidence
* Hybrid keyword + semantic retrieval
* Cross-encoder reranking
* Metadata filtering by publication year, journal, or topic
* Improved source and evidence visualization
* Retrieval evaluation benchmarks
* Answer faithfulness evaluation
* Automated retrieval-quality metrics
* Larger and continuously updated literature collections
* More advanced citation verification

---

# 🎓 Key Learning Outcomes

This project demonstrates practical experience with:

* Designing an end-to-end RAG architecture
* Working with a 10K+ scientific literature corpus
* Generating and managing vector embeddings
* Implementing semantic information retrieval
* Ranking sentence-level evidence
* Building evidence-grounded LLM prompts
* Implementing conversational query rewriting
* Separating retrieval and generation components
* Migrating from local LLM inference to cloud inference
* Integrating hosted LLM APIs
* Building an interactive AI application
* Managing persistent application state with SQLite
* Hosting large datasets separately from application code
* Deploying an AI application to the cloud

---

# 👩‍💻 Author

**Deeksha Vijayakumar**

### GitHub

👉 [View the source code](https://github.com/deekshavijayakumxr/scientific-literature-research-assistant)

### Live Application

🚀 **[Launch the Scientific Literature Research Assistant](https://scientific-literature-research-assistant-ec9d8kvnqim565dyeibim.streamlit.app/)**

### Dataset

📚 [View the scientific literature dataset](https://huggingface.co/datasets/deekshavijayakumxr/scientific-literature-research-assistant-data)
