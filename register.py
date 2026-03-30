import streamlit as st
from PIL import Image
import io
from utils.database import add_student, get_all_students, update_face_status, save_face_embedding, delete_student
from utils.face_recognition import extract_embedding, validate_image_quality, detect_faces


DEPARTMENTS = ["Computer Science", "Electronics", "Mechanical", "Civil",
               "Electrical", "Chemical", "Biotechnology", "MBA", "Other"]
YEARS = ["1st Year", "2nd Year", "3rd Year", "4th Year", "5th Year"]


def show():
    st.markdown("<h2 style='color:#6C63FF;'>📸 Student Registration</h2>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["➕ Register New Student", "👥 Manage Students", "📷 Add Face to Existing"])

    # ── TAB 1: Register ────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### Step 1: Enter Student Details")
        col1, col2 = st.columns(2)

        with col1:
            student_id = st.text_input("Student ID *", placeholder="e.g. CS2024001")
            name = st.text_input("Full Name *", placeholder="e.g. Aryan Sharma")
            department = st.selectbox("Department *", DEPARTMENTS)
            year = st.selectbox("Year *", YEARS)

        with col2:
            email = st.text_input("Email *", placeholder="student@university.edu")
            phone = st.text_input("Phone", placeholder="+91 9876543210")

            st.markdown("### Step 2: Upload Face Photo")
            uploaded_file = st.file_uploader(
                "Upload a clear frontal photo",
                type=["jpg", "jpeg", "png", "webp"],
                help="Use a well-lit photo with only one face visible"
            )

        face_embedding = None
        face_method = None

        if uploaded_file:
            image = Image.open(uploaded_file)
            col_img, col_info = st.columns([1, 1])
            with col_img:
                st.image(image, caption="Uploaded Photo", width=250)

            with col_info:
                with st.spinner("🔍 Analyzing face..."):
                    valid, msg = validate_image_quality(image)

                if valid:
                    st.success(f"✅ {msg}")
                    with st.spinner("🧠 Extracting face embedding..."):
                        face_embedding, face_method, _ = extract_embedding(image)
                    st.info(f"🔬 Method: **{face_method}**")
                    st.markdown(f"""
                    <div style='background:rgba(67,233,123,0.1); border:1px solid rgba(67,233,123,0.3);
                         border-radius:8px; padding:0.8rem; font-size:0.85rem;'>
                        ✅ Face encoded as <b>{len(face_embedding)}-dimensional</b> vector<br>
                        Ready to register!
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(f"❌ {msg}")

        st.markdown("---")

        # Camera option
        st.markdown("### Or Capture via Webcam")
        camera_image = st.camera_input("Take a photo with your camera")
        cam_embedding = None
        cam_method = None

        if camera_image:
            image = Image.open(camera_image)
            with st.spinner("🔍 Analyzing captured photo..."):
                valid, msg = validate_image_quality(image)
            if valid:
                st.success(f"✅ {msg}")
                cam_embedding, cam_method, _ = extract_embedding(image)
                st.info(f"🔬 Method: **{cam_method}** | Vector: {len(cam_embedding)}d")
                face_embedding = cam_embedding
                face_method = cam_method
            else:
                st.error(f"❌ {msg}")

        st.markdown("---")

        if st.button("🎓 Register Student", use_container_width=True, type="primary"):
            if not student_id or not name or not email:
                st.error("Please fill in all required fields (marked with *).")
            else:
                success, message = add_student(
                    student_id.strip().upper(), name.strip(),
                    department, year, email.strip(), phone.strip()
                )
                if success:
                    if face_embedding:
                        save_face_embedding(student_id.strip().upper(), face_embedding)
                        update_face_status(student_id.strip().upper(), True)
                        st.success(f"🎉 {message} Face registered using **{face_method}**!")
                    else:
                        st.warning(f"✅ {message} ⚠️ No face registered. Add a face later.")
                    st.balloons()
                else:
                    st.error(message)

    # ── TAB 2: Manage Students ─────────────────────────────────────────────────
    with tab2:
        students = get_all_students()
        st.markdown(f"### 👥 Registered Students ({len(students)})")

        if not students:
            st.info("No students registered yet.")
            return

        search = st.text_input("🔍 Search students", placeholder="Name, ID, or department...")

        for sid, s in students.items():
            if search and search.lower() not in (s["name"] + sid + s["department"]).lower():
                continue

            pct = 0
            if s.get("total_classes", 0) > 0:
                pct = s["attended_classes"] / s["total_classes"] * 100

            face_badge = "✅ Face" if s.get("face_registered") else "⚠️ No Face"
            risk = "🔴 At Risk" if pct < 75 and s.get("total_classes", 0) > 0 else "🟢 Good"

            with st.expander(f"👤 {s['name']} — {sid} | {face_badge} | {risk}"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Department", s["department"])
                c2.metric("Year", s["year"])
                c3.metric("Attendance", f"{pct:.1f}%")
                st.markdown(f"📧 {s['email']} | 📱 {s.get('phone', 'N/A')}")
                st.caption(f"Registered: {s['registered_at'][:10]}")

                if st.button(f"🗑️ Delete {sid}", key=f"del_{sid}"):
                    if delete_student(sid):
                        st.success("Student deleted.")
                        st.rerun()

    # ── TAB 3: Add Face to Existing ────────────────────────────────────────────
    with tab3:
        students = get_all_students()
        no_face = {sid: s for sid, s in students.items() if not s.get("face_registered")}
        all_ids = list(students.keys())

        st.markdown("### 📷 Add/Update Face for Existing Student")
        target_id = st.selectbox("Select Student", all_ids,
                                 format_func=lambda x: f"{students[x]['name']} ({x})")

        if target_id:
            st.info(f"Selected: **{students[target_id]['name']}**")
            src = st.radio("Image Source", ["Upload File", "Use Camera"])

            img = None
            if src == "Upload File":
                f = st.file_uploader("Upload face photo", type=["jpg", "jpeg", "png"])
                if f:
                    img = Image.open(f)
            else:
                cam = st.camera_input("Capture face")
                if cam:
                    img = Image.open(cam)

            if img:
                st.image(img, width=200)
                with st.spinner("Analyzing..."):
                    valid, msg = validate_image_quality(img)
                if valid:
                    st.success(msg)
                    if st.button("💾 Save Face", type="primary"):
                        try:
                            emb, method, _ = extract_embedding(img)
                            save_face_embedding(target_id, emb)
                            update_face_status(target_id, True)
                            st.success(f"✅ Face updated using {method}!")
                            st.rerun()
                        except ValueError as e:
                            st.error(f"❌ {e}")
                else:
                    st.error(msg)
