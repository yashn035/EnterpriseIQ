import os
import requests
import streamlit as st

# Configure Streamlit Page
st.set_page_config(
    page_title="EnterpriseIQ | AI Policy Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics and dark glassmorphism styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.4rem;
        margin-bottom: 0.2rem;
    }

    .sub-header {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .badge-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
        border-radius: 10px;
        padding: 0.5rem 0.9rem;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 600;
        color: #38bdf8;
        font-size: 0.85rem;
        margin-top: 0.4rem;
        margin-bottom: 0.6rem;
    }

    .source-tag {
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(99, 102, 241, 0.4);
        color: #818cf8;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        display: inline-block;
        margin: 4px 4px 4px 0;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/isometric-line/100/processor.png", width=64)
st.sidebar.markdown("## ⚙️ Configuration")

api_endpoint = st.sidebar.text_input("Backend API Endpoint", value=os.getenv("API_URL", "http://localhost:8000/ask"))
api_base_url = api_endpoint.rsplit("/ask", 1)[0]

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Specs")
st.sidebar.markdown("""
- **Embedding Model**: `all-MiniLM-L6-v2`
- **Vector DB**: Pinecone Serverless
- **LLM Engine**: Mistral-7B (TGI)
- **Monitoring**: Wandb & Prometheus
""")

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Title Header
st.markdown('<h1 class="main-header">EnterpriseIQ Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Strict policy & document intelligence powered by RAG and Mistral-7B</p>', unsafe_allow_html=True)

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your EnterpriseIQ strict policy assistant. Ask me anything based on your ingested company documents.",
            "sources": [],
            "latency_ms": None,
            "user_question": None
        }
    ]


def send_feedback(user_q: str, answer_text: str, rating: str):
    """Sends user feedback (Thumbs Up / Down) to backend /feedback API."""
    try:
        res = requests.post(
            f"{api_base_url}/feedback",
            json={
                "question": user_q or "General Inquiry",
                "answer": answer_text,
                "feedback": rating
            },
            timeout=5
        )
        if res.status_code == 200:
            st.toast(f"Thank you for your feedback! ({'👍' if rating == 'positive' else '👎'})")
        else:
            st.toast("Could not save feedback.")
    except Exception as e:
        st.toast(f"Feedback connection error: {e}")


# Render existing chat messages
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("sources"):
            with st.expander("📄 Source Documents"):
                for src in msg["sources"]:
                    st.markdown(f'<div class="source-tag">📄 [Source: {src}]</div>', unsafe_allow_html=True)

        if msg.get("latency_ms") is not None:
            st.markdown(
                f'<div class="badge-card">⚡ Latency: <span>{msg["latency_ms"]} ms</span></div>',
                unsafe_allow_html=True
            )

        # Render Feedback Action Row for Assistant Messages
        if msg["role"] == "assistant" and msg.get("latency_ms") is not None:
            user_q = msg.get("user_question", "")
            btn_col1, btn_col2, _ = st.columns([1, 1, 10])

            with btn_col1:
                if st.button("👍", key=f"up_{idx}"):
                    send_feedback(user_q, msg["content"], "positive")

            with btn_col2:
                if st.button("👎", key=f"down_{idx}"):
                    send_feedback(user_q, msg["content"], "negative")


# Handle User Input
if prompt := st.chat_input("Ask a question about enterprise policies or documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base & generating cited response..."):
            try:
                response = requests.post(
                    api_endpoint,
                    json={"question": prompt},
                    headers={"Content-Type": "application/json"},
                    timeout=60
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer returned.")
                    sources = data.get("sources", [])
                    latency_ms = data.get("latency_ms", 0.0)

                    st.markdown(answer)

                    if sources:
                        with st.expander("📄 Source Documents"):
                            for src in sources:
                                st.markdown(f'<div class="source-tag">📄 [Source: {src}]</div>', unsafe_allow_html=True)

                    st.markdown(
                        f'<div class="badge-card">⚡ Latency: <span>{latency_ms} ms</span></div>',
                        unsafe_allow_html=True
                    )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "latency_ms": latency_ms,
                        "user_question": prompt
                    })
                    st.rerun()

                else:
                    error_msg = f"API Error ({response.status_code}): {response.text}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": [],
                        "latency_ms": None,
                        "user_question": prompt
                    })

            except requests.exceptions.RequestException as e:
                err_text = f"Connection failed to backend API at `{api_endpoint}`. Details: {e}"
                st.error(err_text)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": err_text,
                    "sources": [],
                    "latency_ms": None,
                    "user_question": prompt
                })
