import json
import os
from datetime import datetime, date
import numpy as np

DB_PATH = "database/students.json"
ATTENDANCE_PATH = "database/attendance.json"
EMBEDDINGS_PATH = "database/embeddings.json"

os.makedirs("database", exist_ok=True)


def _load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ── Student CRUD ──────────────────────────────────────────────────────────────

def get_all_students():
    return _load_json(DB_PATH)


def get_student(student_id: str):
    return get_all_students().get(student_id)


def add_student(student_id: str, name: str, department: str,
                year: str, email: str, phone: str = ""):
    students = get_all_students()
    if student_id in students:
        return False, "Student ID already exists."
    students[student_id] = {
        "id": student_id,
        "name": name,
        "department": department,
        "year": year,
        "email": email,
        "phone": phone,
        "registered_at": datetime.now().isoformat(),
        "face_registered": False,
        "total_classes": 0,
        "attended_classes": 0,
    }
    _save_json(DB_PATH, students)
    return True, "Student registered successfully."


def update_face_status(student_id: str, status: bool = True):
    students = get_all_students()
    if student_id in students:
        students[student_id]["face_registered"] = status
        _save_json(DB_PATH, students)
        return True
    return False


def save_face_embedding(student_id: str, embedding: list):
    embeddings = _load_json(EMBEDDINGS_PATH)
    embeddings[student_id] = embedding
    _save_json(EMBEDDINGS_PATH, embeddings)


def get_face_embeddings():
    return _load_json(EMBEDDINGS_PATH)


def delete_student(student_id: str):
    students = get_all_students()
    if student_id in students:
        del students[student_id]
        _save_json(DB_PATH, students)
        embs = _load_json(EMBEDDINGS_PATH)
        if student_id in embs:
            del embs[student_id]
            _save_json(EMBEDDINGS_PATH, embs)
        return True
    return False


# ── Attendance ────────────────────────────────────────────────────────────────

def mark_attendance(student_id: str, subject: str = "General",
                    confidence: float = 1.0):
    attendance = _load_json(ATTENDANCE_PATH)
    today = str(date.today())
    key = f"{student_id}_{today}_{subject}"
    if key in attendance:
        return False, "Already marked for today."
    attendance[key] = {
        "student_id": student_id,
        "date": today,
        "subject": subject,
        "time": datetime.now().strftime("%H:%M:%S"),
        "confidence": round(confidence, 4),
        "method": "face_recognition",
    }
    _save_json(ATTENDANCE_PATH, attendance)

    # Update totals
    students = get_all_students()
    if student_id in students:
        students[student_id]["attended_classes"] += 1
        students[student_id]["total_classes"] = max(
            students[student_id]["total_classes"],
            students[student_id]["attended_classes"],
        )
        _save_json(DB_PATH, students)
    return True, "Attendance marked."


def get_attendance_records(date_filter: str = None, subject: str = None):
    attendance = _load_json(ATTENDANCE_PATH)
    records = list(attendance.values())
    if date_filter:
        records = [r for r in records if r["date"] == date_filter]
    if subject and subject != "All":
        records = [r for r in records if r["subject"] == subject]
    return records


def get_student_attendance(student_id: str):
    attendance = _load_json(ATTENDANCE_PATH)
    return [v for v in attendance.values() if v["student_id"] == student_id]


def get_today_stats():
    today = str(date.today())
    attendance = _load_json(ATTENDANCE_PATH)
    today_records = [v for v in attendance.values() if v["date"] == today]
    total_students = len(get_all_students())
    present = len(set(r["student_id"] for r in today_records))
    return {
        "total": total_students,
        "present": present,
        "absent": total_students - present,
        "percentage": round((present / total_students * 100) if total_students else 0, 1),
    }
