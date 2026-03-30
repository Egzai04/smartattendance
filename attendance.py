
import streamlit as st
from PIL import Image
from datetime import date

from utils.database import (
    get_all_students,
    get_face_embeddings,
    mark_attendance,
    get_attendance_records
)

from utils.face_recognition import (
    extract_embedding,
    validate_image_quality,
    recognize_face
)


SUBJECTS = [
    "General", "Mathematics", "Physics", "Chemistry",
    "Programming", "Data Structures", "Machine Learning",
    "Database", "Networks", "Other"
]


def show():
    st.markdown("<h2 style='color:#6C63FF;'>✅ Mark Attendance</h2>", unsafe_allow_html=True)

    st.markdown("### 🤖 Automatic Attendance via Face Recognition")

    col1, col2 = st.columns([2, 1])

    # ── Right Panel (Controls) ─────────────────────────────
    with col2:
        subject = st.selectbox("Subject", SUBJECTS)
        threshold = st.slider("Confidence Threshold", 0.40, 0.95, 0.60, 0.01)
        st.caption("Lower = more tolerant of low-quality images")

    # ── Left Panel (Camera Only) ───────────────────────────
    with col1:
        captured = st.camera_input("📷 Point camera at the student's face")
        image = None

        if captured:
            image = Image.open(captured)

    # ── Processing ─────────────────────────────────────────
    if image:
        with st.spinner("🔍 Detecting face..."):
            valid, msg = validate_image_quality(image)

        if not valid:
            st.error(f"❌ {msg}")
            return

        st.success(f"✅ {msg}")

        embeddings = get_face_embeddings()
        students = get_all_students()

        if not embeddings:
            st.warning("⚠️ No face embeddings found. Register students first.")
            return

        # ── Face Recognition ───────────────────────────────
        with st.spinner("🧠 Recognizing face..."):
            try:
                query_emb, method, recommended_threshold = extract_embedding(image)
            except ValueError as e:
                st.error(f"❌ {e}")
                return

            effective_threshold = max(threshold, recommended_threshold)
            student_id, confidence = recognize_face(
                query_emb,
                embeddings,
                effective_threshold
            )

        # ── Result Section ─────────────────────────────────
        st.markdown("---")
        col_res, col_action = st.columns([2, 1])

        with col_res:
            if student_id and student_id in students:
                s = students[student_id]
                conf_pct = int(confidence * 100)

                color = "#43E97B" if conf_pct >= 80 else "#FFB347"

                st.markdown(f"""
                <div style='background:rgba(67,233,123,0.1);
                            border:2px solid rgba(67,233,123,0.4);
                            border-radius:16px;
                            padding:1.5rem;
                            text-align:center;'>

                    <div style='font-size:2.5rem;'>👤</div>

                    <div style='font-size:1.4rem;
                                font-weight:700;
                                color:#E8E8F0;'>
                        {s['name']}
                    </div>

                    <div style='color:#8888AA;'>
                        {student_id} · {s['department']} · {s['year']}
                    </div>

                    <div style='margin-top:1rem;
                                font-size:1.8rem;
                                font-weight:700;
                                color:{color};'>
                        {conf_pct}% Match
                    </div>

                    <div style='color:#8888AA;
                                font-size:0.8rem;'>
                        via {method}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Action Buttons ─────────────────────────
                with col_action:
                    st.markdown("<br>", unsafe_allow_html=True)

                    today_recs = get_attendance_records(str(date.today()), subject)
                    already_marked = any(r["student_id"] == student_id for r in today_recs)

                    if already_marked:
                        st.warning(f"⚠️ Already marked for **{subject}** today.")
                    else:
                        if st.button("✅ Confirm & Mark", type="primary", use_container_width=True):
                            success, msg2 = mark_attendance(student_id, subject, confidence)

                            if success:
                                st.success(f"🎉 Attendance marked for {s['name']}!")
                                st.balloons()
                            else:
                                st.warning(msg2)

                    if st.button("❌ Not This Student", use_container_width=True):
                        st.info("Recognition rejected. Try again.")

            else:
                st.markdown(f"""
                <div style='background:rgba(255,101,132,0.1);
                            border:2px solid rgba(255,101,132,0.4);
                            border-radius:16px;
                            padding:1.5rem;
                            text-align:center;'>

                    <div style='font-size:2.5rem;'>❓</div>

                    <div style='font-size:1.2rem;
                                font-weight:700;
                                color:#FF6584;'>
                        Student Not Recognized
                    </div>

                    <div style='color:#8888AA;
                                margin-top:0.5rem;'>
                        Confidence: {int(confidence * 100)}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.info("Try adjusting the threshold or register the student.")

