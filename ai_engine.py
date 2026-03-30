"""
AI Assistant powered by LangChain + Groq.
Provides intelligent Q&A on attendance data and smart insights.
"""

import os
from datetime import date
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Load API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# System prompt
SYSTEM_PROMPT = """You are SmartAttend AI, an intelligent assistant for a university attendance management system.
You have access to attendance data, student records, and analytics.

Your capabilities:
- Answer questions about student attendance patterns
- Identify at-risk students (attendance < 75%)
- Generate attendance reports and summaries
- Provide recommendations for improving attendance
- Help faculty understand trends and anomalies

Current date: {date}

Be concise, data-driven, and helpful. Format numbers clearly.
When discussing students, respect privacy by using IDs where possible.

Attendance data context:
{context}
"""


# Initialize Groq client
def get_groq_client():
    if not GROQ_API_KEY:
        return None
    try:
        return ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama3-8b-8192",
            temperature=0.3,
            max_tokens=1024,
        )
    except Exception:
        return None


# Build context from attendance data
def build_attendance_context(students: dict, attendance_records: list) -> str:
    total = len(students)
    today = str(date.today())

    today_records = [r for r in attendance_records if r["date"] == today]
    present_today = len(set(r["student_id"] for r in today_records))

    at_risk = []
    for sid, s in students.items():
        if s.get("total_classes", 0) > 0:
            pct = s["attended_classes"] / s["total_classes"] * 100
            if pct < 75:
                at_risk.append(f"{s['name']} ({sid}): {pct:.1f}%")

    context = f"""
Total registered students: {total}
Present today: {present_today} / {total}
At-risk students (attendance < 75%): {len(at_risk)}
{chr(10).join(at_risk[:5]) if at_risk else 'None currently'}
Total attendance records: {len(attendance_records)}
"""
    return context.strip()


# Main chat function
def chat_with_ai(messages: list, students: dict, attendance_records: list) -> str:
    """
    messages: [{"role": "user"/"assistant", "content": "..."}]
    """
    llm = get_groq_client()
    if not llm:
        return (
            "⚠️ Groq API key not configured.\n"
            "Add GROQ_API_KEY to your .env file.\n"
            "Get it from: https://console.groq.com"
        )

    context = build_attendance_context(students, attendance_records)

    lc_messages = [
        SystemMessage(
            content=SYSTEM_PROMPT.format(
                date=str(date.today()),
                context=context,
            )
        )
    ]

    # Convert chat history
    for msg in messages:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        else:
            lc_messages.append(AIMessage(content=msg["content"]))

    try:
        response = llm.invoke(lc_messages)
        return response.content
    except Exception as e:
        return f"❌ Error contacting Groq API: {str(e)}"


# Quick dashboard insight
def generate_quick_insight(students: dict, attendance_records: list) -> str:
    llm = get_groq_client()
    if not llm:
        return "⚠️ Configure GROQ_API_KEY to enable AI insights."

    context = build_attendance_context(students, attendance_records)

    prompt = f"""Based on this attendance data, provide a brief 2-3 sentence insight:

{context}

Focus on:
- Overall attendance health
- Any concerns
- One actionable recommendation
"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"❌ Could not generate insight: {str(e)}"


# Individual student analysis
def analyze_student(student: dict, records: list) -> str:
    llm = get_groq_client()
    if not llm:
        return "⚠️ Configure GROQ_API_KEY for AI analysis."

    total = student.get("total_classes", 0)
    attended = student.get("attended_classes", 0)
    pct = (attended / total * 100) if total > 0 else 0

    prompt = f"""Analyze this student's attendance:

Name: {student['name']}
Department: {student['department']}
Year: {student['year']}

Attendance: {attended}/{total} ({pct:.1f}%)
Recent records: {len(records)}

Provide:
- Status assessment
- Risk level
- One recommendation

Keep response under 100 words.
"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"❌ Analysis failed: {str(e)}"