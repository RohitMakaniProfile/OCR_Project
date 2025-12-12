from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
from pathlib import Path

from ocr_pipeline_easyocr import run_pipeline  # your pipeline

app = Flask(__name__)

# Where uploaded + redacted files will be stored
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
REDACTED_DIR = BASE_DIR / "redacted"

UPLOAD_DIR.mkdir(exist_ok=True)
REDACTED_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        img_path = UPLOAD_DIR / filename
        file.save(str(img_path))

        # path for redacted image
        redacted_filename = f"redacted_{filename}"
        redacted_path = REDACTED_DIR / redacted_filename

        # run OCR + PII
        result = run_pipeline(str(img_path), str(redacted_path))

        # Build URLs for frontend
        result["uploaded_image_url"] = url_for(
            "serve_file", folder="uploads", filename=filename
        )

        if result.get("redacted_path"):
            result["redacted_image_url"] = url_for(
                "serve_file", folder="redacted", filename=redacted_filename
            )
        else:
            result["redacted_image_url"] = None

        return jsonify(result)

    return jsonify({"error": "Invalid file type"}), 400


@app.route("/files/<folder>/<filename>")
def serve_file(folder, filename):
    """Serve uploaded and redacted images."""
    if folder == "uploads":
        directory = UPLOAD_DIR
    elif folder == "redacted":
        directory = REDACTED_DIR
    else:
        return "Not found", 404

    # send_from_directory is a function imported from flask
    return send_from_directory(str(directory), filename)


if __name__ == "__main__":
    app.run(debug=True)
