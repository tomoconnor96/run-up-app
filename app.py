import os
import uuid
import traceback
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash

from engine.tracking import read_frames
from engine.pose import extract_landmarks, interpolate_joints, model_available
from engine.phases import detect_phases, score_phases
from engine.render import render_annotated
from engine.report import build_report_html

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

ALLOWED_EXT = {"mp4", "mov", "m4v"}
MAX_CONTENT_LENGTH = 80 * 1024 * 1024
MAX_DURATION_SECONDS = 12

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.secret_key = "bowling-analysis-dev-key"


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("video")
    if not file or file.filename == "":
        flash("Choose a video first.")
        return redirect(url_for("index"))
    if not allowed_file(file.filename):
        flash("Please upload an MP4 or MOV file.")
        return redirect(url_for("index"))

    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(RESULT_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    in_path = os.path.join(job_dir, "input.mp4")
    file.save(in_path)

    try:
        frames, fps = read_frames(in_path)
        n = len(frames)
        if n < 5:
            flash("Couldn't read that video -- try a different file.")
            return redirect(url_for("index"))
        if n / fps > MAX_DURATION_SECONDS:
            flash(f"Clips over {MAX_DURATION_SECONDS}s aren't supported yet -- trim to just the "
                  f"run-up through follow-through.")
            return redirect(url_for("index"))

        if not model_available():
            flash("The pose-tracking model isn't installed on this server yet -- check the Dockerfile build step.")
            return redirect(url_for("index"))

        track_lms = extract_landmarks(frames, fps)
        joints = interpolate_joints(track_lms, n)
        phases, events = detect_phases(joints, fps)
        scores, notes = score_phases(joints, phases, events, fps)

        raw_path = os.path.join(job_dir, "annotated_raw.mp4")
        final_path = os.path.join(job_dir, "annotated.mp4")
        render_annotated(frames, fps, phases, events, joints, raw_path, final_path, in_path)

        rows_html, overall = build_report_html(scores, notes, phases, fps, file.filename)
        with open(os.path.join(job_dir, "meta.html"), "w") as f:
            f.write(rows_html)
        with open(os.path.join(job_dir, "overall.txt"), "w") as f:
            f.write(str(overall))

    except Exception:
        traceback.print_exc()
        flash("Something went wrong analysing that clip. Please try another video.")
        return redirect(url_for("index"))

    return redirect(url_for("result", job_id=job_id))


@app.route("/result/<job_id>")
def result(job_id):
    job_dir = os.path.join(RESULT_DIR, job_id)
    meta_path = os.path.join(job_dir, "meta.html")
    overall_path = os.path.join(job_dir, "overall.txt")
    if not os.path.exists(meta_path):
        flash("That analysis wasn't found -- it may have expired.")
        return redirect(url_for("index"))
    with open(meta_path) as f:
        rows_html = f.read()
    with open(overall_path) as f:
        overall = f.read()
    return render_template("result.html", job_id=job_id, rows_html=rows_html, overall=overall)


@app.route("/media/<job_id>/<filename>")
def media(job_id, filename):
    job_dir = os.path.join(RESULT_DIR, job_id)
    return send_from_directory(job_dir, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
