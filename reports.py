import streamlit as st
import pandas as pd
from datetime import date
from utils.database import get_all_students, get_attendance_records, get_student_attendance
from utils.ai_engine import analyze_student

SUBJECTS = ["All", "General", "Mathematics", "Physics", "Chemistry",
            "Programming", "Data Structures", "Machine Learning", "Database", "Networks"]


def show():
    st.markdown("<h2 style='color:#6C63FF;'>📊 Attendance Reports</h2>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📅 Daily Report", "👤 Student Report", "📈 Analytics"])

    students = get_all_students()
    all_records = get_attendance_records()

    # ── TAB 1: Daily ───────────────────────────────────────────────────────────
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            selected_date = st.date_input("Select Date", value=date.today())
        with col2:
            subject_filter = st.selectbox("Subject", SUBJECTS)

        records = get_attendance_records(str(selected_date), subject_filter)

        present_ids = {r["student_id"] for r in records}
        total = len(students)
        present = len(present_ids)

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Students", total)
        m2.metric("Present", present, delta=f"{present/total*100:.1f}%" if total else "0%")
        m3.metric("Absent", total - present)

        # Present
        st.markdown("#### ✅ Present Students")
        if present_ids:
            rows = []
            for sid in present_ids:
                s = students.get(sid, {})
                rec = next((r for r in records if r["student_id"] == sid), {})
                rows.append({
                    "ID": sid,
                    "Name": s.get("name", "Unknown"),
                    "Department": s.get("department", "—"),
                    "Time": rec.get("time", "—"),
                    "Subject": rec.get("subject", "—"),
                    "Confidence": f"{int(rec.get('confidence', 1)*100)}%",
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", csv,
                               f"attendance_{selected_date}.csv", "text/csv")
        else:
            st.info("No attendance records for this date/subject.")

        # Absent
        st.markdown("#### ❌ Absent Students")
        absent_ids = set(students.keys()) - present_ids
        if absent_ids:
            rows = [{"ID": sid, "Name": students[sid]["name"],
                     "Department": students[sid]["department"]}
                    for sid in absent_ids]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.success("All students present! 🎉")

    # ── TAB 2: Student Report ──────────────────────────────────────────────────
    with tab2:
        if not students:
            st.info("No students registered.")
            return

        sid = st.selectbox("Select Student",
                           list(students.keys()),
                           format_func=lambda x: f"{students[x]['name']} ({x})")

        s = students[sid]
        recs = get_student_attendance(sid)

        total_cls = s.get("total_classes", max(len(recs), 1))
        attended = s.get("attended_classes", len(recs))
        pct = (attended / total_cls * 100) if total_cls > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Classes Attended", attended)
        col2.metric("Total Classes", total_cls)
        col3.metric("Attendance %", f"{pct:.1f}%",
                    delta="At Risk" if pct < 75 else "Good",
                    delta_color="inverse" if pct < 75 else "normal")

        # Progress bar
        color = "#43E97B" if pct >= 75 else "#FF6584"
        st.markdown(f"""
        <div style='background:#1A1A2E; border-radius:10px; padding:3px; margin-bottom:1rem;'>
            <div style='background:{color}; border-radius:8px; height:16px; width:{pct}%;
                 transition:width 1s; display:flex; align-items:center; justify-content:center;
                 font-size:0.7rem; color:white; font-weight:700;'>{pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # AI Analysis
        if st.button("🤖 Generate AI Analysis"):
            with st.spinner("Analyzing..."):
                analysis = analyze_student(s, recs)
            st.info(analysis)

        # Records table
        if recs:
            df = pd.DataFrame(recs)[["date", "time", "subject", "confidence"]]
            df["confidence"] = df["confidence"].apply(lambda x: f"{int(x*100)}%")
            df.columns = ["Date", "Time", "Subject", "Confidence"]
            st.dataframe(df.sort_values("Date", ascending=False),
                         use_container_width=True, hide_index=True)
        else:
            st.info("No attendance records for this student.")

    # ── TAB 3: Analytics ───────────────────────────────────────────────────────
    with tab3:
        st.markdown("### 📈 Overall Analytics")

        if not all_records:
            st.info("No attendance data available yet.")
            return

        df = pd.DataFrame(all_records)
        df["date"] = pd.to_datetime(df["date"])

        # Daily trend
        daily = df.groupby("date")["student_id"].nunique().reset_index()
        daily.columns = ["Date", "Students Present"]
        st.markdown("#### Daily Attendance Trend")
        st.area_chart(daily.set_index("Date"), use_container_width=True, height=200)

        col1, col2 = st.columns(2)
        with col1:
            # Subject breakdown
            st.markdown("#### Attendance by Subject")
            subj = df.groupby("subject")["student_id"].count().reset_index()
            subj.columns = ["Subject", "Records"]
            st.bar_chart(subj.set_index("Subject"), use_container_width=True, height=200)

        with col2:
            # Attendance distribution
            st.markdown("#### Student Attendance Distribution")
            buckets = {"<50%": 0, "50-75%": 0, "75-90%": 0, "90-100%": 0}
            for s in students.values():
                t = s.get("total_classes", 0)
                if t == 0:
                    continue
                p = s["attended_classes"] / t * 100
                if p < 50:
                    buckets["<50%"] += 1
                elif p < 75:
                    buckets["50-75%"] += 1
                elif p < 90:
                    buckets["75-90%"] += 1
                else:
                    buckets["90-100%"] += 1
            bdf = pd.DataFrame(list(buckets.items()), columns=["Range", "Students"])
            st.bar_chart(bdf.set_index("Range"), use_container_width=True, height=200)
