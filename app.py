"""
Flask web UI for the offline audio transcriber.
Run:  python app.py
Then open http://localhost:5000 in your browser.
"""

import os
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

from transcriber import transcribe

app = Flask(__name__)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# job_id -> {"status": "pending|running|done|error", "message": str, "output": Path|None}
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


def _run_transcription(job_id: str, audio_path: Path, model: str, language: str | None, task: str) -> None:
    with jobs_lock:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["message"] = "Transcribing… this may take a few minutes."

    try:
        result = transcribe(
            audio_path=str(audio_path),
            model_name=model,
            language=language or None,
            task=task,
        )
        full_text = result["text"].strip()
        detected_lang = result.get("language", "unknown")

        stem = audio_path.stem
        out_path = OUTPUT_DIR / f"{stem}_{job_id[:8]}.txt"
        out_path.write_text(full_text, encoding="utf-8")

        with jobs_lock:
            jobs[job_id]["status"] = "done"
            jobs[job_id]["message"] = f"Done! Detected language: {detected_lang}"
            jobs[job_id]["output"] = str(out_path)
    except Exception as exc:
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["message"] = f"Error: {exc}"
    finally:
        try:
            audio_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/transcribe", methods=["POST"])
def start_transcription():
    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    model = request.form.get("model", "base")
    language = request.form.get("language", "").strip() or None
    task = request.form.get("task", "transcribe")

    job_id = uuid.uuid4().hex
    filename = Path(file.filename).name
    audio_path = UPLOAD_DIR / f"{job_id}_{filename}"
    file.save(audio_path)

    with jobs_lock:
        jobs[job_id] = {"status": "pending", "message": "Queued…", "output": None}

    thread = threading.Thread(target=_run_transcription, args=(job_id, audio_path, model, language, task), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def job_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify({"status": job["status"], "message": job["message"]})


@app.route("/download/<job_id>")
def download(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None or job["status"] != "done":
        return jsonify({"error": "Not ready"}), 404
    out_path = Path(job["output"])
    return send_file(out_path, as_attachment=True, download_name=out_path.name)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
