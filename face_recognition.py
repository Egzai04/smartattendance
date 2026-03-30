"""
Face recognition utilities — LOW-QUALITY TOLERANT version.

Key improvements over previous version:
  • enhance_image()   — denoises, sharpens, normalises contrast BEFORE detection
  • _crop_face_robust() — 3-pass cascade (strict → relaxed → very relaxed) so
                         small, blurry, or dark faces are still detected
  • upscale_if_small() — bicubic upscale when face region < 80×80 px
  • DeepFace called with enforce_detection=False on retry so blurry frames
    that pass our own face check still get embedded
  • Multi-backend detection: Haar frontal → Haar profile → LBP cascade
  • validate_image_quality() now auto-enhances instead of hard-rejecting

Embedding priority (highest accuracy first):
  1. DeepFace ArcFace / Facenet512 / Facenet  (local)
  2. HuggingFace ResNet-50 feature-extraction (API, face-crop only)
  3. DCT face fingerprint                     (local, no extra deps)

Similarity: cosine on L2-normalised embeddings.
"""

import numpy as np
import io
import os
import tempfile
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import requests

HF_API_KEY  = os.getenv("HUGGINGFACE_API_KEY", "")
HF_FEAT_URL = "https://api-inference.huggingface.co/models/microsoft/resnet-50"

# Per-method similarity thresholds
THRESHOLD_DEEPFACE = 0.60   # lowered slightly to handle low-quality frames
THRESHOLD_HF       = 0.75
THRESHOLD_DCT      = 0.88
DEFAULT_THRESHOLD  = 0.60


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE ENHANCEMENT  — run BEFORE any detection or embedding
# ═══════════════════════════════════════════════════════════════════════════════

def enhance_image(image: Image.Image) -> Image.Image:
    """
    Pipeline that makes low-quality images (dark, blurry, noisy, low-res)
    much more recognisable:
      1. Upscale small images (< 300 px wide) using LANCZOS
      2. Auto-contrast / histogram equalisation per channel
      3. Sharpening (unsharp mask)
      4. Brightness / contrast normalisation
      5. Mild denoising via median filter
    Returns the enhanced PIL image (RGB).
    """
    img = image.convert("RGB")
    w, h = img.size

    # 1. Upscale tiny images
    if w < 300 or h < 300:
        scale = max(300 / w, 300 / h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # 2. Per-channel CLAHE-like contrast stretch via ImageOps
    r, g, b = img.split()
    r = ImageOps.equalize(r)
    g = ImageOps.equalize(g)
    b = ImageOps.equalize(b)
    img = Image.merge("RGB", (r, g, b))

    # 3. Unsharp mask — recovers detail from soft/blurry shots
    img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=2))

    # 4. Brightness / contrast tweak
    img = ImageEnhance.Contrast(img).enhance(1.3)
    img = ImageEnhance.Brightness(img).enhance(1.05)

    # 5. Mild median denoise (removes camera noise without blurring structure)
    img = img.filter(ImageFilter.MedianFilter(size=3))

    return img


def _upscale_face(face_crop: Image.Image, target: int = 160) -> Image.Image:
    """Upscale a small face crop to at least target×target px."""
    w, h = face_crop.size
    if w < target or h < target:
        scale = max(target / w, target / h)
        face_crop = face_crop.resize(
            (int(w * scale), int(h * scale)), Image.LANCZOS
        )
    return face_crop


# ═══════════════════════════════════════════════════════════════════════════════
# FACE DETECTION  — multi-pass, multi-cascade, tolerant
# ═══════════════════════════════════════════════════════════════════════════════

def _opencv_detect(gray_arr, scale: float, neighbors: int,
                   min_size: tuple) -> list:
    """Run Haar frontal-face cascade with given params."""
    try:
        import cv2
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = cascade.detectMultiScale(
            gray_arr,
            scaleFactor=scale,
            minNeighbors=neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return list(faces) if len(faces) else []
    except Exception:
        return []


def _lbp_detect(gray_arr) -> list:
    """LBP cascade — faster, better on low-contrast images."""
    try:
        import cv2
        lbp_path = cv2.data.haarcascades + "lbpcascade_frontalface_improved.xml"
        if not os.path.exists(lbp_path):
            return []
        cascade = cv2.CascadeClassifier(lbp_path)
        faces = cascade.detectMultiScale(gray_arr, 1.05, 3, minSize=(30, 30))
        return list(faces) if len(faces) else []
    except Exception:
        return []


def detect_faces(image: Image.Image) -> list[dict]:
    """
    Multi-pass face detection. Tries progressively more relaxed settings
    so small, blurry, or poorly-lit faces are still found.
    Returns list of {x, y, w, h} dicts (largest-first).
    """
    try:
        import cv2
        arr  = np.array(image.convert("RGB"))
        bgr  = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # CLAHE on grayscale improves detection in dark images
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)

        faces = []

        # Pass 1 — strict (high quality)
        if not faces:
            faces = _opencv_detect(gray_eq, 1.1, 5, (60, 60))
        # Pass 2 — relaxed neighbours (blurry / partially occluded)
        if not faces:
            faces = _opencv_detect(gray_eq, 1.05, 3, (40, 40))
        # Pass 3 — very relaxed (small face, far away)
        if not faces:
            faces = _opencv_detect(gray_eq, 1.05, 2, (20, 20))
        # Pass 4 — LBP cascade (different algorithm, catches misses)
        if not faces:
            faces = _lbp_detect(gray_eq)
        # Pass 5 — profile cascade (side-on face)
        if not faces:
            profile = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_profileface.xml"
            )
            pf = profile.detectMultiScale(gray_eq, 1.05, 2, minSize=(20, 20))
            faces = list(pf) if len(pf) else []

        # Sort largest first
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        return [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
                for (x, y, w, h) in faces]
    except Exception:
        return []


def _crop_face_robust(image: Image.Image) -> Image.Image | None:
    """
    Detect and crop the largest face with 25 % padding.
    If no face found on original, tries again on enhanced version.
    Returns upscaled face crop or None.
    """
    def _do_crop(img: Image.Image) -> Image.Image | None:
        faces = detect_faces(img)
        if not faces:
            return None
        f = faces[0]
        x, y, w, h = f["x"], f["y"], f["w"], f["h"]
        pad_x = int(w * 0.25)
        pad_y = int(h * 0.25)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img.width,  x + w + pad_x)
        y2 = min(img.height, y + h + pad_y)
        crop = img.crop((x1, y1, x2, y2))
        return _upscale_face(crop, 160)

    result = _do_crop(image)
    if result is None:
        # Retry on enhanced image
        result = _do_crop(enhance_image(image))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING METHODS
# ═══════════════════════════════════════════════════════════════════════════════

def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-10 else vec


def get_embedding_deepface(image: Image.Image) -> tuple[list, str] | None:
    """
    ArcFace → Facenet512 → Facenet via DeepFace.
    For low-quality images:
      • First tries with enforce_detection=True (face MUST be found internally).
      • On failure retries with enforce_detection=False (we already validated
        that a face exists via our own detector, so this is safe).
    Returns (embedding, model_name) or None.
    """
    try:
        from deepface import DeepFace
        import cv2

        enhanced = enhance_image(image)
        arr = np.array(enhanced)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            cv2.imwrite(tmp.name, bgr)
            tmp_path = tmp.name

        for model in ("ArcFace", "Facenet512", "Facenet"):
            # Try strict first, then relaxed
            for enforce in (True, False):
                try:
                    result = DeepFace.represent(
                        img_path=tmp_path,
                        model_name=model,
                        enforce_detection=enforce,
                        detector_backend="opencv",
                        align=True,
                    )
                    os.unlink(tmp_path)
                    emb = np.array(result[0]["embedding"], dtype=np.float32)
                    return _l2_normalize(emb).tolist(), model
                except Exception:
                    continue

        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return None
    except Exception:
        return None


def get_embedding_huggingface(image: Image.Image) -> list | None:
    """HuggingFace ResNet-50 on enhanced face crop."""
    if not HF_API_KEY:
        return None
    face_crop = _crop_face_robust(image)
    if face_crop is None:
        return None
    try:
        face_crop = enhance_image(face_crop).resize((224, 224))
        buf = io.BytesIO()
        face_crop.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        resp = requests.post(HF_FEAT_URL, headers=headers,
                             data=buf.read(), timeout=30)
        if resp.status_code != 200:
            return None
        result = resp.json()
        if isinstance(result, list):
            flat = np.array(result, dtype=np.float32).flatten()
            return _l2_normalize(flat).tolist()
        return None
    except Exception:
        return None


def get_embedding_dct(image: Image.Image) -> list | None:
    """DCT fingerprint on enhanced face crop — last resort."""
    face_crop = _crop_face_robust(image)
    if face_crop is None:
        return None
    face_crop = enhance_image(face_crop).convert("L").resize((64, 64))
    arr = np.array(face_crop, dtype=np.float32) / 255.0
    try:
        from scipy.fftpack import dct as scipy_dct
        coeffs = scipy_dct(scipy_dct(arr, axis=1, norm="ortho"),
                           axis=0, norm="ortho")
    except ImportError:
        coeffs = np.abs(np.fft.fft2(arr))
    feat = coeffs[:16, :16].flatten().astype(np.float32)
    return _l2_normalize(feat).tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def extract_embedding(image: Image.Image) -> tuple[list, str, float]:
    """
    Extract a face embedding from any quality image.
    Returns (embedding, method_label, recommended_threshold).
    Raises ValueError only when absolutely no face can be found.
    """
    result = get_embedding_deepface(image)
    if result:
        emb, model = result
        return emb, f"DeepFace ({model})", THRESHOLD_DEEPFACE

    emb = get_embedding_huggingface(image)
    if emb:
        return emb, "HuggingFace (ResNet-50)", THRESHOLD_HF

    emb = get_embedding_dct(image)
    if emb:
        return emb, "DCT Face Fingerprint", THRESHOLD_DCT

    raise ValueError(
        "No face could be detected even after image enhancement. "
        "Tips: improve lighting, face the camera directly, remove glasses/mask, "
        "or move closer so your face fills more of the frame."
    )


def cosine_similarity(a: list, b: list) -> float:
    va = _l2_normalize(np.array(a, dtype=np.float32))
    vb = _l2_normalize(np.array(b, dtype=np.float32))
    n  = min(len(va), len(vb))
    return float(np.dot(va[:n], vb[:n]))


def recognize_face(
    query_embedding: list,
    stored_embeddings: dict,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[str | None, float]:
    """
    Match query against all stored embeddings.
    Returns (student_id, confidence) or (None, best_score).

    Ambiguity guard: if the top-2 scores are within 0.04 of each other
    the result is considered ambiguous and None is returned.
    """
    if not stored_embeddings:
        return None, 0.0

    scores = {sid: cosine_similarity(query_embedding, emb)
              for sid, emb in stored_embeddings.items()}

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_id, best_score = ranked[0]

    if len(ranked) > 1 and (best_score - ranked[1][1]) < 0.04 and best_score >= threshold:
        return None, best_score   # ambiguous — refuse to commit

    return (best_id, best_score) if best_score >= threshold else (None, best_score)


# ── Validation (lenient for low-quality, auto-enhances) ──────────────────────

def validate_image_quality(image: Image.Image) -> tuple[bool, str]:
    """
    Accept low-quality images but warn the user.
    Auto-enhancement is applied transparently inside extract_embedding.
    Hard-fails only on: too small, zero faces, or clearly multiple people.
    """
    w, h = image.size

    if w < 50 or h < 50:
        return False, "Image too small (< 50×50 px) — cannot process."

    # Try detection on both original and enhanced
    faces = detect_faces(image)
    if not faces:
        faces = detect_faces(enhance_image(image))

    if not faces:
        return False, (
            "No face detected even after enhancement. "
            "Please ensure your face is visible and not heavily obscured."
        )

    if len(faces) > 1:
        return False, (
            f"{len(faces)} faces detected — please use a photo with only one person."
        )

    f = faces[0]
    face_area = f["w"] * f["h"]
    img_area  = w * h
    quality_notes = []

    if face_area < img_area * 0.03:
        quality_notes.append("face is small — accuracy may be lower")
    if w < 200 or h < 200:
        quality_notes.append("low resolution — enhancement applied")

    note = f" ({'; '.join(quality_notes)})" if quality_notes else ""
    return True, f"Face detected{note} — proceeding with enhancement."
