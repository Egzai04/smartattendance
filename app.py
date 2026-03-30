import streamlit as st

st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    :root {
        --primary: #6C63FF;
        --secondary: #FF6584;
        --accent: #43E97B;
        --bg-dark: #0D0D1A;
        --bg-card: #1A1A2E;
        --text: #E8E8F0;
        --muted: #8888AA;
    }

    * { font-family: 'Space Grotesk', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0D0D1A 0%, #1A0A2E 50%, #0A1A2E 100%);
        color: var(--text);
    }

    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #6C63FF, #FF6584, #43E97B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -1px;
    }

    .sub-header {
        text-align: center;
        color: var(--muted);
        font-size: 1rem;
        margin-bottom: 2rem;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .metric-card {
        background: linear-gradient(135deg, #1A1A2E, #16213E);
        border: 1px solid rgba(108, 99, 255, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        border-color: var(--primary);
        box-shadow: 0 0 20px rgba(108, 99, 255, 0.2);
        transform: translateY(-2px);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6C63FF, #43E97B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }

    .nav-card {
        background: linear-gradient(135deg, #1A1A2E, #16213E);
        border: 1px solid rgba(108, 99, 255, 0.2);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }

    .nav-card:hover {
        border-color: #6C63FF;
        box-shadow: 0 8px 32px rgba(108, 99, 255, 0.25);
        transform: translateY(-4px);
    }

    .nav-icon { font-size: 2.5rem; margin-bottom: 0.75rem; }
    .nav-title { font-weight: 600; font-size: 1.1rem; color: #E8E8F0; margin-bottom: 0.3rem; }
    .nav-desc { color: var(--muted); font-size: 0.85rem; }

    .status-online {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(67, 233, 123, 0.1);
        border: 1px solid rgba(67, 233, 123, 0.3);
        border-radius: 20px; padding: 4px 12px;
        font-size: 0.8rem; color: #43E97B;
    }

    .pulse { width: 8px; height: 8px; background: #43E97B; border-radius: 50%;
        animation: pulse 2s infinite; }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.3); }
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D0D1A, #1A0A2E) !important;
        border-right: 1px solid rgba(108, 99, 255, 0.2);
    }

    .stButton > button {
        background: linear-gradient(135deg, #6C63FF, #9B8CFF) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(108, 99, 255, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size:3rem;'>🎓</div>
        <div style='font-weight:700; font-size:1.2rem; color:#6C63FF;'>SmartAttend</div>
        <div style='color:#8888AA; font-size:0.75rem; letter-spacing:2px;'>AI-POWERED</div>
    </div>
    <hr style='border-color: rgba(108,99,255,0.2); margin: 1rem 0;'>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='status-online'>
        <div class='pulse'></div> System Online
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**📋 Navigation**")
    page = st.radio("", [
        "🏠 Dashboard",
        "📸 Register Student",
        "✅ Mark Attendance",
        "📊 View Reports",
        "🤖 AI Assistant",
        "⚙️ Settings"
    ], label_visibility="collapsed")

    st.markdown("<br><hr style='border-color:rgba(108,99,255,0.2);'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color:#8888AA; font-size:0.75rem; text-align:center;'>
        Powered by LangChain · Groq · HuggingFace
    </div>
    """, unsafe_allow_html=True)

# Route pages
if "🏠 Dashboard" in page:
    from pages import dashboard
    dashboard.show()
elif "📸 Register Student" in page:
    from pages import register
    register.show()
elif "✅ Mark Attendance" in page:
    from pages import attendance
    attendance.show()
elif "📊 View Reports" in page:
    from pages import reports
    reports.show()
elif "🤖 AI Assistant" in page:
    from pages import ai_assistant
    ai_assistant.show()
elif "⚙️ Settings" in page:
    from pages import settings
    settings.show()
