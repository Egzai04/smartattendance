import streamlit as st
import os


def show():
    st.markdown("<h2 style='color:#6C63FF;'>⚙️ Settings & Configuration</h2>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔑 API Keys", "ℹ️ System Info"])

    with tab1:
        st.markdown("### 🔑 API Key Configuration")
        st.info("These keys are stored in session only. For persistence, use a `.env` file.")

        groq_key = st.text_input("Groq API Key",
                                  value=st.session_state.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", "")),
                                  type="password",
                                  help="Get free key at console.groq.com")

        hf_key = st.text_input("HuggingFace API Key",
                                value=st.session_state.get("HF_API_KEY", os.getenv("HUGGINGFACE_API_KEY", "")),
                                type="password",
                                help="Get key at huggingface.co/settings/tokens")

        if st.button("💾 Save API Keys", type="primary"):
            os.environ["GROQ_API_KEY"] = groq_key
            os.environ["HUGGINGFACE_API_KEY"] = hf_key
            st.session_state["GROQ_API_KEY"] = groq_key
            st.session_state["HF_API_KEY"] = hf_key
            st.success("✅ API keys saved for this session!")
            st.info("To persist across sessions, add them to a `.env` file.")

        st.markdown("---")
        st.markdown("### 🧪 Test Connections")
        if st.button("Test Groq Connection"):
            key = groq_key or os.getenv("GROQ_API_KEY", "")
            if not key:
                st.error("No Groq API key configured.")
            else:
                try:
                    from langchain_groq import ChatGroq
                    from langchain.schema import HumanMessage
                    llm = ChatGroq(groq_api_key=key, model_name="llama3-8b-8192", max_tokens=50)
                    r = llm.invoke([HumanMessage(content="Say 'Groq connected!' in 5 words.")])
                    st.success(f"✅ Groq: {r.content}")
                except Exception as e:
                    st.error(f"❌ Groq error: {e}")

        if st.button("Test HuggingFace Connection"):
            key = hf_key or os.getenv("HUGGINGFACE_API_KEY", "")
            if not key:
                st.error("No HuggingFace API key configured.")
            else:
                import requests
                r = requests.get("https://huggingface.co/api/whoami",
                                  headers={"Authorization": f"Bearer {key}"})
                if r.status_code == 200:
                    st.success(f"✅ HuggingFace: Connected as {r.json().get('name', 'user')}")
                else:
                    st.error(f"❌ HuggingFace: {r.status_code}")

    with tab2:
        st.markdown("### ℹ️ System Information")
        st.markdown("""
        | Component | Detail |
        |-----------|--------|
        | **Face Recognition** | DeepFace (Facenet) → HuggingFace DINO → Simulated |
        | **AI Engine** | LangChain + Groq Llama 3 (8B) |
        | **Embeddings** | HuggingFace Inference API |
        | **Database** | Local JSON (upgrade to SQLite/PostgreSQL) |
        | **Frontend** | Streamlit |
        | **Similarity** | Cosine similarity |
        | **Threshold** | 0.75 (configurable) |
        """)

        st.markdown("---")
        st.markdown("### 🚀 .env File Template")
        st.code("""# .env — place in project root
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
HUGGINGFACE_API_KEY=hf_xxxxxxxxxxxxxxxx
""", language="bash")

        st.markdown("### 📦 Requirements")
        st.code("""streamlit>=1.32.0
langchain>=0.1.0
langchain-groq>=0.1.0
deepface>=0.0.93
opencv-python-headless>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0
requests>=2.31.0
python-dotenv>=1.0.0
tf-keras>=2.15.0
""", language="text")
