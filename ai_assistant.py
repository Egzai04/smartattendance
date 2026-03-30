import streamlit as st
from utils.database import get_all_students, get_attendance_records
from utils.ai_engine import chat_with_ai

STARTER_QUESTIONS = [
    "Who has the lowest attendance this month?",
    "Which students are at risk of not meeting the 75% requirement?",
    "What is the overall attendance trend this week?",
    "Give me a summary of today's attendance.",
    "Which subject has the most absences?",
]


def show():
    st.markdown("<h2 style='color:#6C63FF;'>🤖 AI Attendance Assistant</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:linear-gradient(135deg,rgba(108,99,255,0.1),rgba(67,233,123,0.05));
         border:1px solid rgba(108,99,255,0.3); border-radius:12px; padding:1rem; margin-bottom:1.5rem;'>
        Powered by <b>LangChain + Groq (Llama 3)</b> · Ask anything about attendance, students, or trends.
    </div>
    """, unsafe_allow_html=True)

    students = get_all_students()
    records = get_attendance_records()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Quick starters
    st.markdown("**💡 Quick Questions:**")
    cols = st.columns(3)
    for i, q in enumerate(STARTER_QUESTIONS):
        with cols[i % 3]:
            if st.button(q, key=f"starter_{i}", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": q})
                with st.spinner("Thinking..."):
                    reply = chat_with_ai(st.session_state.chat_history, students, records)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.rerun()

    st.markdown("---")

    # Chat history
    for msg in st.session_state.chat_history:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            st.markdown(f"""
            <div style='display:flex; justify-content:flex-end; margin-bottom:0.75rem;'>
                <div style='background:linear-gradient(135deg,#6C63FF,#9B8CFF);
                     border-radius:16px 16px 4px 16px; padding:0.75rem 1.2rem;
                     max-width:75%; color:white;'>
                    {content}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='display:flex; justify-content:flex-start; margin-bottom:0.75rem;'>
                <div style='background:rgba(26,26,46,0.9); border:1px solid rgba(108,99,255,0.3);
                     border-radius:16px 16px 16px 4px; padding:0.75rem 1.2rem;
                     max-width:80%; color:#E8E8F0; line-height:1.6;'>
                    🤖 {content}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Input
    user_input = st.chat_input("Ask about attendance, students, or trends...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("🤖 Thinking..."):
            reply = chat_with_ai(st.session_state.chat_history, students, records)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

    # Clear
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
