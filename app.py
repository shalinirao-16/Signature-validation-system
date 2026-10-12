
import os
import hmac
import uuid
from functools import wraps

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    abort
)
from werkzeug.utils import secure_filename
from skimage.metrics import structural_similarity as ssim

# Load private configuration from the local .env file
load_dotenv()

app = Flask(__name__)

# Require configuration instead of using public default credentials
app.secret_key = os.getenv("FLASK_SECRET_KEY")
APP_USERNAME = os.getenv("APP_USERNAME")
APP_PASSWORD = os.getenv("APP_PASSWORD")

if not all([app.secret_key, APP_USERNAME, APP_PASSWORD]):
    raise RuntimeError(
        "Missing configuration. Set FLASK_SECRET_KEY, "
        "APP_USERNAME and APP_PASSWORD in your local .env file."
    )

# Upload configuration
UPLOAD_FOLDER = os.path.join(
    app.root_path, "static", "uploaded_images"
)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5 MB per image

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Require login before accessing protected pages
def login_required(view_function):
    @wraps(view_function)
    def wrapped_view(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return view_function(*args, **kwargs)

    return wrapped_view


def allowed_file(filename):
    return (
        bool(filename)
        and "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def load_uploaded_image(file_storage):
    """Validate and decode an uploaded image."""
    if not file_storage or not file_storage.filename:
        raise ValueError("Please select an image.")

    if not allowed_file(file_storage.filename):
        raise ValueError("Only PNG, JPG and JPEG images are allowed.")

    image_bytes = file_storage.read(MAX_IMAGE_SIZE + 1)
    file_storage.seek(0)

    if not image_bytes or len(image_bytes) > MAX_IMAGE_SIZE:
        raise ValueError("Each image must be smaller than 5 MB.")

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

    if image is None or image.size == 0:
        raise ValueError("An uploaded file is not a valid image.")

    return image, image_bytes


def save_image(image_bytes, original_filename):
    """Save an image using a generated filename."""
    extension = original_filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{extension}"
    safe_name = secure_filename(filename)

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)

    with open(filepath, "wb") as image_file:
        image_file.write(image_bytes)

    return filepath


@app.route("/")
def default():
    if session.get("logged_in"):
        return redirect(url_for("home"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("logged_in"):
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")

        valid_username = hmac.compare_digest(username, APP_USERNAME)
        valid_password = hmac.compare_digest(password, APP_PASSWORD)

        if valid_username and valid_password:
            session.clear()
            session["logged_in"] = True
            return redirect(url_for("home"))

        return render_template(
            "login.html",
            error="Invalid username or password"
        )

    return render_template("login.html")


@app.route("/logout", methods=["POST"])
@login_required
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/home")
@login_required
def home():
    return render_template(
        "index.html",
        result=None,
        best_match=None
    )


@app.route("/verify", methods=["POST"])
@login_required
def verify():
    reference_file = request.files.get("reference")
    signature_files = request.files.getlist("signatures")

    if not reference_file or not reference_file.filename:
        return render_template(
            "index.html",
            result="Please upload a reference signature.",
            best_match=None
        )

    candidates = [
        file for file in signature_files
        if file and file.filename
    ]

    if not candidates:
        return render_template(
            "index.html",
            result="Please upload at least one signature to compare.",
            best_match=None
        )

    saved_files = []

    try:
        reference_image, reference_bytes = load_uploaded_image(
            reference_file
        )

        # Save with a generated filename to avoid collisions
        reference_path = save_image(
            reference_bytes, reference_file.filename
        )
        saved_files.append(reference_path)

        best_match_path = None
        max_similarity = -1.0

        for candidate in candidates:
            candidate_image, candidate_bytes = load_uploaded_image(
                candidate
            )

            candidate_path = save_image(
                candidate_bytes, candidate.filename
            )
            saved_files.append(candidate_path)

            # SSIM requires both images to have the same dimensions
            resized_candidate = cv2.resize(
                candidate_image,
                (reference_image.shape[1], reference_image.shape[0]),
                interpolation=cv2.INTER_AREA
            )

            similarity = float(
                ssim(reference_image, resized_candidate)
            )

            if similarity > max_similarity:
                max_similarity = similarity
                best_match_path = candidate_path

        # Keep the original project's threshold-based result
        threshold = 0.8

        if max_similarity > threshold:
            result = "Signature is verified"
            best_match = os.path.relpath(
                best_match_path, os.path.join(app.root_path, "static")
            ).replace("\\", "/")
        else:
            result = "Signature is not verified"
            best_match = None

        # Store the result in the session rather than URL parameters
        session["verification_result"] = result
        session["best_match"] = best_match

        return redirect(url_for("result"))

    except ValueError as error:
        # Remove files saved during a failed verification
        for filepath in saved_files:
            try:
                os.remove(filepath)
            except OSError:
                pass

        return render_template(
            "index.html",
            result=str(error),
            best_match=None
        )


@app.route("/result")
@login_required
def result():
    result_text = session.get("verification_result")
    best_match = session.get("best_match")

    if not result_text:
        return redirect(url_for("home"))

    return render_template(
        "result.html",
        result=result_text,
        best_match=best_match
    )


@app.errorhandler(413)
def request_too_large(error):
    if session.get("logged_in"):
        return render_template(
            "index.html",
            result="Upload too large. Reduce the image sizes and try again.",
            best_match=None
        ), 413

    return redirect(url_for("login"))


if __name__ == "__main__":
    # Local development only; do not expose the Flask development server
    app.run(debug=False)
