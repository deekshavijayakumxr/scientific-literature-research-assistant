import sys
import sqlite3
from pathlib import Path

# --------------------------------------------------
# Make project root available to Python
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import ollama

from src.rag import answer_research_question


# --------------------------------------------------
# Database
# --------------------------------------------------

DB_PATH = PROJECT_ROOT / "data" / "research_assistant.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    conn = get_connection()

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            evidence TEXT,
            sources TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id) REFERENCES chats(id)
        )
        """
    )

    conn.commit()

    # Create first chat if database is empty
    existing_chat = conn.execute(
        "SELECT id FROM chats ORDER BY id LIMIT 1"
    ).fetchone()

    if existing_chat is None:
        conn.execute(
            "INSERT INTO chats (name) VALUES (?)",
            ("Chat 1",)
        )
        conn.commit()

    conn.close()


def get_chats():
    conn = get_connection()

    rows = conn.execute(
        """
        SELECT id, name
        FROM chats
        ORDER BY id
        """
    ).fetchall()

    conn.close()

    return rows


def create_chat():
    conn = get_connection()

    count = conn.execute(
        "SELECT COUNT(*) FROM chats"
    ).fetchone()[0]

    chat_number = count + 1
    chat_name = f"Chat {chat_number}"

    # Prevent accidental duplicate names
    while conn.execute(
        "SELECT id FROM chats WHERE name = ?",
        (chat_name,)
    ).fetchone():

        chat_number += 1
        chat_name = f"Chat {chat_number}"

    cursor = conn.execute(
        "INSERT INTO chats (name) VALUES (?)",
        (chat_name,)
    )

    conn.commit()

    chat_id = cursor.lastrowid

    conn.close()

    return chat_id, chat_name


def get_messages(chat_id):
    conn = get_connection()

    rows = conn.execute(
        """
        SELECT role, content, evidence, sources
        FROM messages
        WHERE chat_id = ?
        ORDER BY id
        """,
        (chat_id,)
    ).fetchall()

    conn.close()

    messages = []

    for row in rows:

        message = {
            "role": row["role"],
            "content": row["content"]
        }

        if row["evidence"]:
            import json
            message["evidence"] = json.loads(row["evidence"])
        else:
            message["evidence"] = []

        if row["sources"]:
            import json
            message["sources"] = json.loads(row["sources"])
        else:
            message["sources"] = []

        messages.append(message)

    return messages


def save_message(
    chat_id,
    role,
    content,
    evidence=None,
    sources=None
):
    import json

    conn = get_connection()

    conn.execute(
        """
        INSERT INTO messages (
            chat_id,
            role,
            content,
            evidence,
            sources
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            chat_id,
            role,
            content,
            json.dumps(evidence or []),
            json.dumps(sources or [])
        )
    )

    conn.commit()
    conn.close()


# --------------------------------------------------
# Initialize database
# --------------------------------------------------

initialize_database()


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Scientific Literature Research Assistant",
    page_icon="🔬",
    layout="wide"
)


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "current_chat_id" not in st.session_state:

    first_chat = get_chats()[0]

    st.session_state.current_chat_id = first_chat["id"]


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.title("💬 Research Chats")

    # --------------------------------------------------
    # New Chat
    # --------------------------------------------------

    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):

        new_chat_id, new_chat_name = create_chat()

        st.session_state.current_chat_id = new_chat_id

        st.rerun()

    st.divider()

    st.subheader("Chat History")

    chats = get_chats()

    for chat in chats:

        chat_id = chat["id"]
        chat_name = chat["name"]

        is_current = (
            chat_id == st.session_state.current_chat_id
        )

        button_label = (
            f"🟢 {chat_name}"
            if is_current
            else chat_name
        )

        if st.button(
            button_label,
            key=f"chat_{chat_id}",
            use_container_width=True
        ):

            st.session_state.current_chat_id = chat_id

            st.rerun()


# --------------------------------------------------
# Current chat
# --------------------------------------------------

current_chat_id = st.session_state.current_chat_id

conn = get_connection()

current_chat = conn.execute(
    """
    SELECT id, name
    FROM chats
    WHERE id = ?
    """,
    (current_chat_id,)
).fetchone()

conn.close()

if current_chat is None:

    chats = get_chats()

    current_chat = chats[0]

    st.session_state.current_chat_id = current_chat["id"]

    current_chat_id = current_chat["id"]


current_chat_name = current_chat["name"]

messages = get_messages(current_chat_id)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    """
    <div style="
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    ">
        🔬 Scientific Literature Research Assistant
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="
        font-size: 1.05rem;
        color: #666;
        margin-bottom: 2rem;
    ">
        Ask research questions, explore scientific literature,
        and continue your conversation using evidence from
        retrieved papers.
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Current chat title
# --------------------------------------------------

st.subheader(f"📚 {current_chat_name}")


# --------------------------------------------------
# Display previous conversation
# --------------------------------------------------

for message in messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # --------------------------------------------------
        # Evidence
        # --------------------------------------------------

        if message.get("evidence"):

            with st.expander("🧠 Selected Evidence"):

                for item in message["evidence"]:

                    st.markdown(
                        f"**[{item['paper_number']}] "
                        f"{item['title']} "
                        f"({item['year']})**"
                    )

                    st.write(item["sentence"])

                    st.caption(
                        f"Evidence similarity: "
                        f"{item['similarity']:.3f}"
                    )

        # --------------------------------------------------
        # Sources
        # --------------------------------------------------

        if message.get("sources"):

            with st.expander("📚 Sources"):

                for source in message["sources"]:

                    st.markdown(
                        f"**[{source['number']}] "
                        f"{source['title']} "
                        f"({source['year']})**"
                    )

                    st.caption(
                        f"Similarity: "
                        f"{source['similarity']:.3f}"
                    )

                    st.caption(
                        f"Citations: "
                        f"{source['citations']}"
                    )


# --------------------------------------------------
# Chat input
# --------------------------------------------------

question = st.chat_input(
    "Ask a research question..."
)


# --------------------------------------------------
# Process new question
# --------------------------------------------------

if question:

    # --------------------------------------------------
    # Save user message permanently
    # --------------------------------------------------

    save_message(
        chat_id=current_chat_id,
        role="user",
        content=question
    )

    # --------------------------------------------------
    # Display user question
    # --------------------------------------------------

    with st.chat_message("user"):

        st.markdown(question)


    # --------------------------------------------------
    # Build conversation history
    # --------------------------------------------------

    conversation_history = []

    for message in messages:

        if message["role"] == "user":

            conversation_history.append(
                f"USER: {message['content']}"
            )

        elif message["role"] == "assistant":

            conversation_history.append(
                f"ASSISTANT: {message['content']}"
            )


    # --------------------------------------------------
    # Determine research query
    # --------------------------------------------------

    if conversation_history:

        recent_history = conversation_history[-8:]

        conversation_context = "\n\n".join(
            recent_history
        )

        rewrite_prompt = f"""
You are a query-rewriting assistant for a scientific
literature research system.

Convert the CURRENT USER QUESTION into one clear,
standalone scientific research question.

Use the conversation history to understand references
such as "it", "this", "that", "the technique", "the method",
"the disease", "the application", etc.

CONVERSATION HISTORY:

{conversation_context}

CURRENT USER QUESTION:

{question}

RULES:

1. Preserve the scientific topic of the conversation.
2. Resolve vague references using the conversation.
3. Use previous assistant answers when useful.
4. Do not answer the question.
5. Do not introduce unrelated scientific facts.
6. Do not invent terminology.
7. Return ONLY one standalone research question.
8. Make it concise and suitable for semantic search.

STANDALONE RESEARCH QUESTION:
"""

        try:

            rewrite_response = ollama.chat(
                model="llama3.2",
                messages=[
                    {
                        "role": "user",
                        "content": rewrite_prompt
                    }
                ]
            )

            rewritten_question = (
                rewrite_response
                .get("message", {})
                .get("content", "")
                .strip()
            )

            if rewritten_question:

                research_query = rewritten_question

            else:

                research_query = question

        except Exception:

            research_query = question

    else:

        research_query = question


    # --------------------------------------------------
    # Run RAG
    # --------------------------------------------------

    with st.chat_message("assistant"):

        try:

            with st.status(
                "Running literature research...",
                expanded=True
            ):

                st.write(
                    "🔎 Searching the scientific literature..."
                )

                result = answer_research_question(
                    research_query,
                    top_papers=5,
                    top_evidence=8
                )

                st.write(
                    "🧠 Extracting the most relevant evidence..."
                )

                st.write(
                    "✍️ Generating an evidence-based answer..."
                )


            # --------------------------------------------------
            # Answer
            # --------------------------------------------------

            st.markdown(
                result["answer"]
            )


            # --------------------------------------------------
            # Evidence
            # --------------------------------------------------

            if result.get("evidence"):

                with st.expander(
                    "🧠 Selected Evidence"
                ):

                    for item in result["evidence"]:

                        st.markdown(
                            f"**[{item['paper_number']}] "
                            f"{item['title']} "
                            f"({item['year']})**"
                        )

                        st.write(
                            item["sentence"]
                        )

                        st.caption(
                            f"Evidence similarity: "
                            f"{item['similarity']:.3f}"
                        )


            # --------------------------------------------------
            # Sources
            # --------------------------------------------------

            if result.get("sources"):

                with st.expander(
                    "📚 Sources"
                ):

                    for source in result["sources"]:

                        st.markdown(
                            f"**[{source['number']}] "
                            f"{source['title']} "
                            f"({source['year']})**"
                        )

                        st.caption(
                            f"Similarity: "
                            f"{source['similarity']:.3f}"
                        )

                        st.caption(
                            f"Citations: "
                            f"{source['citations']}"
                        )


            # --------------------------------------------------
            # Save assistant response permanently
            # --------------------------------------------------

            save_message(
                chat_id=current_chat_id,
                role="assistant",
                content=result["answer"],
                evidence=result.get("evidence", []),
                sources=result.get("sources", [])
            )


        except Exception as e:

            error_message = (
                f"Something went wrong: {e}"
            )

            st.error(
                error_message
            )

            save_message(
                chat_id=current_chat_id,
                role="assistant",
                content=error_message
            )