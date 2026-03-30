import streamlit as st
from datetime import date
from utils.database import get_all_students, get_today_stats, get_attendance_records
from utils.ai_engine import generate_quick_insight


def show():
    st.markdown("<h1 class='main-header'>Smart Attendance Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Real-time AI-Powered Attendance Tracking</p>", unsafe_allow_html=True)

    students = get_all_students()
    stats = get_today_stats()
    records = get_attendance_records()

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        (col1, stats["total"], "Total Students", "👥"),
        (col2, stats["present"], "Present Today", "✅"),
        (col3, stats["absent"], "Absent Today", "❌"),
        (col4, f"{stats['percentage']}%", "Attendance Rate", "📊"),
    ]
    for col, val, label, icon in metrics:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size:1.8rem;'>{icon}</div>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("### 📋 Today's Attendance")
        today_records = get_attendance_records(date_filter=str(date.today()))
        if today_records:
            for rec in today_records[:10]:
                student = students.get(rec["student_id"], {})
                name = student.get("name", rec["student_id"])
                conf = int(rec.get("confidence", 1) * 100)
                st.markdown(f"""
                <div style='background:rgba(108,99,255,0.08); border:1px solid rgba(108,99,255,0.2);
                     border-radius:10px; padding:0.75rem 1rem; margin-bottom:0.5rem;
                     display:flex; justify-content:space-between; align-items:center;'>
                    <span>👤 <b>{name}</b> <span style='color:#8888AA;'>({rec["student_id"]})</span></span>
                    <span style='color:#8888AA;'>{rec["time"]}</span>
                    <span style='color:#43E97B; font-size:0.8rem;'>✓ {conf}%</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No attendance marked today yet.")

    with col_right:
        st.markdown("### 🤖 AI Insight")
        with st.spinner("Generating insight..."):
            insight = generate_quick_insight(students, records)
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,rgba(108,99,255,0.1),rgba(67,233,123,0.05));
             border:1px solid rgba(108,99,255,0.3); border-radius:12px; padding:1.2rem;
             font-size:0.9rem; line-height:1.6; color:#E8E8F0;'>
            {insight}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### ⚠️ At-Risk Students")
        at_risk_count = 0
        for sid, s in students.items():
            if s.get("total_classes", 0) > 0:
                pct = s["attended_classes"] / s["total_classes"] * 100
                if pct < 75:
                    at_risk_count += 1
                    st.markdown(f"""
                    <div style='background:rgba(255,101,132,0.1); border:1px solid rgba(255,101,132,0.3);
                         border-radius:8px; padding:0.6rem 1rem; margin-bottom:0.4rem; font-size:0.85rem;'>
                        ⚠️ <b>{s['name']}</b> — {pct:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
        if at_risk_count == 0:
            st.success("✅ No at-risk students!")

    # Quick stats chart
    st.markdown("### 📈 Recent Attendance Trend")
    import pandas as pd
    from collections import Counter
    dates = [r["date"] for r in records]
    if dates:
        counts = Counter(dates)
        df = pd.DataFrame(sorted(counts.items()), columns=["Date", "Present"])
        st.line_chart(df.set_index("Date"), use_container_width=True, height=200)
    else:
        st.info("No data yet. Start marking attendance to see trends.")
