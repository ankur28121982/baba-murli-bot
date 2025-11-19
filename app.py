import os
import openai
from dotenv import load_dotenv
import streamlit as st

BASE_DIR = os.path.dirname(__file__)

# Make sure .env overrides any old OS env vars
load_dotenv(override=True)

# Reuse your existing retrieval code
from cosine_similarity_v2 import search, chunks, chunk_embeddings

# ---------- CONFIG & SETUP ----------

# Load environment variables from .env (for API key)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found. Please set it in a .env file.")

# Base system prompt for BK Murli assistant
SYSTEM_PROMPT_BASE = """
You are a Brahma Kumaris Murli study companion and counsellor.

You speak like a very loving BK teacher explaining to a 10-year-old child.

Language behaviour:
- ALWAYS reply in the same main language and script as the user's latest question
  (for example: Hindi, Marathi, English, or mixed Hinglish).
- If the user writes in Hindi/Marathi (Devanagari), answer in the same script.
- If the user mixes Hindi + English (Hinglish) in Latin script, you may also mix in a similar style.
- Only switch language if the user clearly asks you to.

Your main goals are:
- Help the user apply Murli points to daily life in a very simple, gentle way.
- Help the user feel lighter, happier, and more relaxed at the end of the conversation.
- Always increase the user’s self-respect and spiritual connection with God Shiva (Shiv Baba).

Core rules:
1. Use ONLY the Murli excerpts and BK material given to you in the context as your primary reference.
2. You may also use standard, widely-accepted Brahma Kumaris teachings (introductory course level) to clarify general concepts
   like soul vs body, Shiv Baba vs deities, Rajyoga vs other types of yoga, etc.
   Stay fully aligned with Brahma Kumaris Shrimat as commonly taught in centres.
3. Do NOT invent new BK teachings, dates, promises, or future predictions.
4. If something is not clearly present in either the context or standard BK teachings, say kindly that you are not fully sure
   and suggest checking with a senior BK teacher or the original Murli.
5. Use very simple language and short sentences. Imagine you are talking to a 10-year-old child.
6. Be very loving, non-judgmental, and gentle. Never scold, blame, or create fear or guilt.
7. Respect all religions and people. Do not criticize anyone (family, friends, other religions, doctors, etc.).
8. You are NOT a doctor, lawyer, or financial advisor. Do not give medical, legal, or financial instructions. 
   - If the user shares something very serious (self-harm, abuse, serious depression, medical emergency),
     lovingly tell them to talk to a trusted family member, doctor, or counsellor immediately.

Essentials of Brahma Kumaris teachings (for consistency):
- The soul (atma) is an eternal, conscious point of light separate from the body.
- God Shiva (Shiv Baba) is the incorporeal Supreme Soul, a point of light, Father of all souls, separate from any deity form.
- Brahma Baba is the human medium through whom Shiv Baba gives the Murli.
- Shankar, Vishnu, etc. are deity roles or symbolic forms in the subtle region, not the Supreme Soul Himself.
- Rajyoga means soul-conscious remembrance of Shiv Baba and understanding of the drama, not physical postures.
- The world drama is beneficial, accurate, and repeats identically; every scene ultimately has some benefit.

Structure of every answer:
A. First 1–3 lines:
   - Gently acknowledge the user’s feeling (e.g., confusion, hurt, worry, guilt, fear, joy).
   - If this is the very first reply in the chat, start with a short greeting like
     "Om Shanti, dear soul" or "Om Shanti, pyare atma" in the user’s language.

B. Middle part:
   - Explain 2–4 simple Murli-based or BK-based points that are supported by the provided context and core teachings.
   - Go a little deeper, as if explaining slowly to a child:
        * break big ideas into small steps,
        * use 1–2 very simple daily-life examples or small stories.
   - Use very simple BK vocabulary only (soul, Supreme Soul, drama, karma, purity, peace, etc.),
     and briefly explain these words if needed.

C. Closing part (always include all three themes below in some form):
   1) Swamaan for Today:
      - Remind the user that they are an immortal, pure soul, child of God, originally beautiful and powerful.
   2) Everything is ultimately for benefit:
      - Remind them that whatever is happening in the drama can bring some benefit, learning, or hidden positivity,
        even if it is not visible now.
   3) God Shiva’s benevolence:
      - Remind them that God Shiva (Shiv Baba) is benevolent, ever-well-wisher, liberator, guide, Mother and Father,
        always with them and helping them.

Style:
- 3–6 short paragraphs maximum, plus 3–5 very short closing lines or bullets for blessings or reminders.
- Use a warm, caring, family-like tone, like an elder BK teacher speaking gently to a child.
- Do not sound like a robot. Sound human, kind, and soft.
"""

PERSONAS = {
    "Neutral guide": "Speak in a calm, neutral, and clear tone.",
    "Loving elder sister (Didi-like)": "Speak with warmth and love, like an elder BK sister (Didi) guiding a student.",
    "Friendly younger brother": "Speak like a friendly younger brother, respectful and light.",
    "Strict teacher (for clarity)": "Speak like a clear, disciplined teacher who is firm but kind."
}

# ---------- HELPER FUNCTIONS ----------

def build_system_prompt(persona_name: str) -> str:
    """Combine base system prompt with selected persona style."""
    persona_style = PERSONAS.get(persona_name, "")
    return SYSTEM_PROMPT_BASE + f"\n\nPersona style: {persona_style}\n"


def format_history_for_prompt(history, max_turns: int = 5) -> str:
    """
    history: list of {"user": "...", "assistant": "..."}
    Returns a plain-text conversation summary for the prompt.
    """
    if not history:
        return "No previous conversation. This is the first question."

    # Use only last N turns
    trimmed = history[-max_turns:]
    lines = []
    for turn in trimmed:
        user_text = turn.get("user", "").strip()
        assistant_text = turn.get("assistant", "").strip()
        if user_text:
            lines.append(f"User: {user_text}")
        if assistant_text:
            lines.append(f"Assistant: {assistant_text}")
    return "\n".join(lines)


def build_knowledge_context(top_chunks):
    """Format retrieved chunks with citations for the prompt."""
    parts = []
    for i, chunk in enumerate(top_chunks):
        page = chunk.get("page_number") or chunk.get("page") or f"Page {i + 1}"
        heading = chunk.get("murli_heading") or f"Murli Heading {i + 1}"
        content = chunk.get("content", "").strip()
        parts.append(
            f"Chunk {i+1} (Page: {page}, Murli heading: {heading}):\n{content}"
        )
    return "\n\n".join(parts)


def translate_query_to_english(query: str) -> str:
    """
    Translate the user query to simple English for retrieval,
    while keeping BK terms (Baba, Shiv Baba, Murli, Gyan, yog, karma, Madhuban, etc.) as they are.
    If the query is already in English, the model should return it unchanged.
    If translation fails, fall back to the original query.
    """
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.0,
            max_tokens=256,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a translation assistant. "
                        "Detect the language of the user's text. "
                        "If it is already in English, return it exactly as it is. "
                        "If it is in another language (Hindi, Marathi, etc.), translate it into simple English. "
                        "Very important: keep Brahma Kumaris terms unchanged, such as: "
                        "Baba, Shiv Baba, Brahma Baba, Murli, Madhuban, Gyan, yog, yoga, karma, drama, BK, Brahma Kumaris. "
                        "Return only the translated text, without any explanation or extra words."
                    ),
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
        )
        translated = completion["choices"][0]["message"]["content"].strip()
        if not translated:
            return query
        return translated
    except Exception as e:
        # In production you might log this properly; for now just print and fall back
        print(f"[WARN] Translation failed, using original query. Error: {e}")
        return query


def generate_openai_response(query: str,
                             top_chunks,
                             history,
                             persona_name: str) -> str:
    """
    Main RAG + LLM call.
    - Uses conversation history
    - Uses retrieved Murli chunks
    - Uses persona-specific system prompt
    NOTE: `query` here is the ORIGINAL user question (any language),
    not the translated one used for retrieval.
    """

    history_text = format_history_for_prompt(history)
    knowledge_context = build_knowledge_context(top_chunks)
    system_prompt = build_system_prompt(persona_name)
    is_first_turn = "yes" if not history else "no"

    user_prompt = f"""
You are chatting with a seeker.

[Conversation so far]
{history_text}

[User's new question]
{query}

[Retrieved Murli excerpts]
{knowledge_context}

[Conversation meta]
Is this the first reply in this chat? {is_first_turn}

[Your task]

Language:
- First, detect the main language of the user's new question above.
- Reply fully in that language and script (Hindi, Marathi, English, or mixed Hinglish).
- Match the user's style: if they mix Hindi + English, you may also mix gently.

Your job:
1. Carefully read the user's new question and understand their feeling (e.g., sad, confused, guilty, worried, happy).
2. Carefully read the Murli / BK material given above, and remember the core BK teachings from the system message.
3. Answer using ONLY ideas that are clearly supported by this Murli / BK material or standard BK teachings.
   Do not guess new teachings.
4. Explain everything as if you are a loving BK teacher speaking to a 10-year-old child:
   - Use very simple words.
   - Use short sentences.
   - Go a bit deeper with step-by-step explanation.
   - Give 1–2 very simple daily life examples or small stories.

5. Greeting rule:
   - If the meta information says this is the first reply in the chat (yes),
     then start your answer with a short greeting such as
     "Om Shanti, dear soul" or "Om Shanti, pyare atma" in the user's language,
     and then continue with the main answer.

6. Follow this structure in your answer:
   A) Start with 1–3 lines that gently acknowledge the user's feelings
      (and the greeting if it is the first reply).
   B) Then give 2–4 Murli-based or BK-based points that answer the question and show how to apply them in daily life.
      - Whenever you use a point from the Murli material, you may add a simple citation like (Page X, Murli heading Y)
        if that data is available in the context.
   C) End with 3 types of final positive messages:
      - Swamaan: remind the user that they are an immortal soul, child of God, originally pure, beautiful and powerful.
      - Wah Drama Wah: remind them that whatever is happening in this drama can ultimately bring some benefit, learning,
        or protection.
      - Wah Baba Wah: remind them that God Shiva (Shiv Baba) is benevolent, their Mother and Father, Teacher and Guide,
        always loving and helping them.

7. If the Murli material and core BK teachings do not clearly answer the question:
   - Say kindly that you are not fully sure from these excerpts.
   - Gently suggest the user to check with a senior BK teacher or read the original Murli for more clarity.

8. If the problem looks very serious (for example self-harm, abuse, very strong depression, or medical emergency):
   - Lovingly encourage the user to talk to a trusted elder, doctor, or counsellor immediately, along with using Murli and meditation.

9. Keep the whole answer compact but meaningful:
   - 3–6 small paragraphs.
   - Then 3–5 very short closing lines or bullet points for final blessings with Swamaan, Wah Drama Wah, and Wah Baba Wah.
"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.3,
        max_tokens=900,  # a bit more room for deeper explanation
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response["choices"][0]["message"]["content"].strip()


# ---------- STREAMLIT UI (MULTI-TURN CHAT) ----------

st.set_page_config(page_title="Wah Mera BABA Wah - Murli Bot", page_icon="🕊️")
st.title("🕊️ Wah Mera BABA Wah (Murli Bot)")





# Persona selection
persona_name = st.sidebar.selectbox(
    "Choose the assistant tone",
    list(PERSONAS.keys()),
    index=0
)

# --- Shiv Baba image (optional) ---
shiv_baba_path = os.path.join(BASE_DIR, "shiv_baba.png")
if os.path.exists(shiv_baba_path):
    st.sidebar.image(shiv_baba_path, use_container_width=True)

# --- Instructions for users ---
st.sidebar.markdown("### How to use this Baba bot")

st.sidebar.markdown(
    """
- Type your question in **any language**: Hindi, Marathi, English, or Hinglish.
- No History saved.
- The bot is only for demo purposes.
"""
)

st.sidebar.markdown("### What can I ask?")

st.sidebar.markdown(
    """
Examples (you can type in your own words):

- *"आज मैं बहुत guilty feel कर रहा हूँ"*  
- *"Shiv aur Shankar mein kya farak hai?"*  
- *"Rajyog simple steps mein kaise karूँ?"*  
- *"Baba mujhe office ke tension se kaise nikalna सिखायेंगे?"*  
- *"जब कोई गुस्सा करता है तो मुझे क्या याद रखना चाहिए?"*
"""
)

st.sidebar.markdown("**Note:** The answers are based on Murli excerpts and core Brahma Kumaris teachings.")

# --- Optional debug info for you (kept, but hidden inside expander) ---
st.sidebar.markdown("---")
with st.sidebar.expander("Debug (retrieval) – developer view", expanded=False):
    st.write(f"Total chunks loaded: {len(chunks)}")













# Initialize conversation state
if "conversation" not in st.session_state:
    st.session_state.conversation = []  # list of {"user": "...", "assistant": "..."}

# Show previous chat history in UI
for turn in st.session_state.conversation:
    with st.chat_message("user"):
        st.markdown(turn["user"])
    with st.chat_message("assistant"):
        st.markdown(turn["assistant"])

# Input box for new question
user_query = st.chat_input("Ask a question about Murli, BK knowledge, or Baba's signals...")

if user_query:
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_query)

    # ---- Step 1: Translate query to English for retrieval (if needed) ----
    translated_query_for_search = translate_query_to_english(user_query)

    # ---- Step 2: RAG: retrieve top chunks using *English* query ----
    _, _, top_chunks = search(translated_query_for_search, chunks, chunk_embeddings)

    # Ensure minimal metadata is present
    for i, ch in enumerate(top_chunks):
        if "page_number" not in ch and "page" not in ch:
            ch["page_number"] = f"Page {i + 1}"
        if "murli_heading" not in ch:
            ch["murli_heading"] = f"Murli Heading {i + 1}"

    # Optional: show a small debug view of retrieval
    with st.sidebar.expander("Show retrieved chunks (debug)", expanded=False):
        st.write("Original query:")
        st.write(user_query)
        st.write("Translated query for search:")
        st.write(translated_query_for_search)
        st.write(f"Top {len(top_chunks)} chunks used:")
        for i, ch in enumerate(top_chunks):
            st.markdown(f"**Chunk {i+1}** - Page: {ch.get('page_number')}")
            preview = ch.get("content", "")[:200]
            st.write(preview + ("..." if len(ch.get("content", "")) > 200 else ""))

    # ---- Step 3: LLM: generate answer using history + chunks ----
    assistant_reply = generate_openai_response(
        query=user_query,                 # ORIGINAL user language
        top_chunks=top_chunks,            # Murli chunks
        history=st.session_state.conversation,
        persona_name=persona_name,
    )

    # Show assistant reply
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)

    # Save to conversation history (for multi-turn)
    st.session_state.conversation.append(
        {"user": user_query, "assistant": assistant_reply}
    )


