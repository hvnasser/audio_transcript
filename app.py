"""
Flask web UI for the offline audio transcriber.
Run:  python app.py
Then open http://localhost:5000 in your browser.
"""

import os
import re
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests as http_requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, render_template, request, send_file

from transcriber import transcribe

app = Flask(__name__)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
DOWNLOADS_DIR = Path("downloads")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
DOWNLOADS_DIR.mkdir(exist_ok=True)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma", ".opus", ".mp4", ".mkv", ".mpr", ".webm"}

# job_id -> {"status": "pending|running|done|error", "message": str, "output": Path|None}
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()

# batch_id -> {"status": "running|paused|done", "total": N, "files": [...]}
batches: dict[str, dict] = {}
batches_lock = threading.Lock()

# batch_id -> threading.Event  (set = running, clear = paused)
batch_resume_events: dict[str, threading.Event] = {}


# ---------------------------------------------------------------------------
# Single-file transcription
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# URL crawl + batch transcription
# ---------------------------------------------------------------------------

def _scrape_audio_links(url: str) -> list[str]:
    """Return unique audio file URLs found on the given page.

    Deduplicates by normalized filename (lowercased, no query string) so that
    pages with both a play button and a download button for the same file only
    return one entry.  The download <a href> is preferred over an <audio src>
    when both exist.
    """
    resp = http_requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Collect candidates: (normalized_filename, full_url, priority)
    # priority 0 = <a href> (download link), 1 = <audio>/<source> (stream)
    candidates: list[tuple[str, str, int]] = []

    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        full = urljoin(url, href)
        fname = Path(urlparse(full).path).name.lower()
        if Path(urlparse(full).path).suffix.lower() in AUDIO_EXTENSIONS:
            candidates.append((fname, full, 0))

    for tag in soup.find_all(["audio", "source"], src=True):
        src = tag["src"]
        full = urljoin(url, src)
        fname = Path(urlparse(full).path).name.lower()
        if Path(urlparse(full).path).suffix.lower() in AUDIO_EXTENSIONS:
            candidates.append((fname, full, 1))

    # Keep only the highest-priority (lowest priority number) URL per filename
    best: dict[str, tuple[str, int]] = {}
    for fname, full_url, priority in candidates:
        if fname not in best or priority < best[fname][1]:
            best[fname] = (full_url, priority)

    return [url for url, _ in best.values()]


def _download_file(url: str, dest_dir: Path) -> Path:
    """Download a file from URL into dest_dir and return its local path."""
    filename = Path(urlparse(url).path).name or "audio_file"
    dest = dest_dir / filename

    # avoid overwriting existing files
    counter = 1
    stem, suffix = dest.stem, dest.suffix
    while dest.exists():
        dest = dest_dir / f"{stem}_{counter}{suffix}"
        counter += 1

    with http_requests.get(url, stream=True, timeout=120, headers={"User-Agent": "Mozilla/5.0"}) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

    return dest


def _run_batch(batch_id: str, audio_urls: list[str], download_dir: Path, model: str, language: str | None, task: str) -> None:
    resume_event = batch_resume_events[batch_id]

    for i, url in enumerate(audio_urls):
        # Wait here if the user has paused the batch
        while not resume_event.is_set():
            with batches_lock:
                batches[batch_id]["status"] = "paused"
            time.sleep(0.5)

        with batches_lock:
            batches[batch_id]["status"] = "running"
            file_entry = batches[batch_id]["files"][i]
            file_entry["status"] = "downloading"
            file_entry["message"] = "Downloading…"

        try:
            local_path = _download_file(url, download_dir)

            # Check pause again before starting transcription (which is slow)
            while not resume_event.is_set():
                with batches_lock:
                    batches[batch_id]["status"] = "paused"
                    file_entry["message"] = "Paused…"
                time.sleep(0.5)

            with batches_lock:
                batches[batch_id]["status"] = "running"
                file_entry["status"] = "transcribing"
                file_entry["message"] = "Transcribing…"

            result = transcribe(
                audio_path=str(local_path),
                model_name=model,
                language=language,
                task=task,
            )
            full_text = result["text"].strip()
            detected_lang = result.get("language", "unknown")

            out_path = OUTPUT_DIR / f"{local_path.stem}_{batch_id[:8]}_{i}.txt"
            out_path.write_text(full_text, encoding="utf-8")

            with batches_lock:
                file_entry["status"] = "done"
                file_entry["message"] = f"Done! Language: {detected_lang}"
                file_entry["output"] = str(out_path)

        except Exception as exc:
            with batches_lock:
                file_entry["status"] = "error"
                file_entry["message"] = f"Error: {exc}"

    with batches_lock:
        batches[batch_id]["status"] = "done"


@app.route("/crawl", methods=["POST"])
def start_crawl():
    data = request.get_json() or {}
    url = data.get("url", "").strip()
    folder = data.get("folder", "").strip()
    model = data.get("model", "base")
    language = data.get("language", "") or None
    task = data.get("task", "transcribe")

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        audio_urls = _scrape_audio_links(url)
    except Exception as exc:
        return jsonify({"error": f"Failed to fetch URL: {exc}"}), 400

    if not audio_urls:
        return jsonify({"error": "No audio files found on that page"}), 404

    safe_folder = re.sub(r"[^\w\-]", "_", folder) if folder else "default"
    download_dir = DOWNLOADS_DIR / safe_folder
    download_dir.mkdir(parents=True, exist_ok=True)

    batch_id = uuid.uuid4().hex
    files = [
        {
            "url": u,
            "filename": Path(urlparse(u).path).name or f"file_{idx}",
            "status": "pending",
            "message": "Queued",
            "output": None,
        }
        for idx, u in enumerate(audio_urls)
    ]

    resume_event = threading.Event()
    resume_event.set()  # start in running state
    batch_resume_events[batch_id] = resume_event

    with batches_lock:
        batches[batch_id] = {"status": "running", "total": len(files), "files": files}

    thread = threading.Thread(
        target=_run_batch,
        args=(batch_id, audio_urls, download_dir, model, language, task),
        daemon=True,
    )
    thread.start()

    return jsonify({"batch_id": batch_id, "total": len(files)})


@app.route("/batch-pause/<batch_id>", methods=["POST"])
def batch_pause(batch_id: str):
    event = batch_resume_events.get(batch_id)
    if event is None:
        return jsonify({"error": "Unknown batch"}), 404
    event.clear()  # signal the thread to pause
    with batches_lock:
        if batches[batch_id]["status"] == "running":
            batches[batch_id]["status"] = "paused"
    return jsonify({"status": "paused"})


@app.route("/batch-resume/<batch_id>", methods=["POST"])
def batch_resume(batch_id: str):
    event = batch_resume_events.get(batch_id)
    if event is None:
        return jsonify({"error": "Unknown batch"}), 404
    event.set()  # signal the thread to continue
    with batches_lock:
        if batches[batch_id]["status"] == "paused":
            batches[batch_id]["status"] = "running"
    return jsonify({"status": "running"})


@app.route("/batch-status/<batch_id>")
def batch_status(batch_id: str):
    with batches_lock:
        batch = batches.get(batch_id)
    if batch is None:
        return jsonify({"error": "Unknown batch"}), 404

    files = [
        {
            "filename": f["filename"],
            "status": f["status"],
            "message": f["message"],
            "has_output": bool(f.get("output")),
        }
        for f in batch["files"]
    ]
    return jsonify({"status": batch["status"], "total": batch["total"], "files": files, "paused": batch["status"] == "paused"})


@app.route("/batch-download/<batch_id>/<int:file_index>")
def batch_download(batch_id: str, file_index: int):
    with batches_lock:
        batch = batches.get(batch_id)
    if batch is None:
        return jsonify({"error": "Unknown batch"}), 404
    if file_index >= len(batch["files"]):
        return jsonify({"error": "Invalid file index"}), 404

    file_entry = batch["files"][file_index]
    if file_entry["status"] != "done" or not file_entry.get("output"):
        return jsonify({"error": "Not ready"}), 404

    out_path = Path(file_entry["output"])
    return send_file(out_path, as_attachment=True, download_name=out_path.name)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
