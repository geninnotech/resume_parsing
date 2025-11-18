from __future__ import annotations

import os
from typing import List

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_from_directory,
    jsonify
)
from markupsafe import Markup, escape

from config import (
    DATA_DIR,
    RESUMES_DIR,
    GROQ_API_KEY,
    GROQ_MODEL_NAME,
    OPENAI_API_KEY,
    OPENAI_MODEL_REASONING,
    OPENAI_MODEL_FAST,
    FLASK_SECRET_KEY,
)
from llm.llm_modules import groqLLM, openaiLLM
from services.state import reset_workspace, load_state
from services.resume_pipeline import process_resumes
from services.vectors_pipeline import build_and_save_state
from services.query_pipeline import (
    generate_vector_weights_for_query,
    compute_query_scores,
    build_answer_for_query,
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY

# ---- LLM clients ----

groq_client = groqLLM(
    api_key=GROQ_API_KEY,
    model=GROQ_MODEL_NAME,
)

openai_reasoning = openaiLLM(
    api_key=OPENAI_API_KEY,
    model=OPENAI_MODEL_REASONING,
    reasoning_model=True,
)

openai_fast = openaiLLM(
    api_key=OPENAI_API_KEY,
    model=OPENAI_MODEL_FAST,
    reasoning_model=False,
)


def linkify_candidate_tags(answer: str, candidates: List[dict]) -> Markup:
    """
    Replace <#resume-{id}#> with anchor tags to candidate detail pages.
    """
    rendered = answer
    for c in candidates:
        cid = c["id"]
        name = c["name"]
        tag = f"<#resume-{cid}#>"
        href = url_for("candidate_detail", candidate_id=cid)
        link_html = f'<a href="{href}" class="text-blue-600 underline">{escape(name)}</a>'
        rendered = rendered.replace(tag, link_html)
    # simple newline to <br> for nicer display
    rendered = rendered.replace("\n", "<br>")
    return Markup(rendered)


# ---- Routes ----


@app.route("/")
def index():
    state = load_state()
    if state is None:
        return redirect(url_for("upload"))
    return redirect(url_for("candidates"))

from werkzeug.utils import secure_filename

from werkzeug.utils import secure_filename
import json as _json

from werkzeug.utils import secure_filename
import json as _json

from werkzeug.utils import secure_filename
import json as _json

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        log_step("GET /upload → rendering upload form")
        return render_template("upload.html")

    # ===== POST =====
    log_step("=== New /upload POST request received ===")
    try:
        # Verbose request introspection
        log_step(f"RAW content_type = {request.content_type}")
        log_step(f"mimetype = {request.mimetype}")
        log_step(f"files keys = {list(request.files.keys())} (len={len(request.files)})")
        log_step(f"form keys = {list(request.form.keys())} (len={len(request.form)})")

        # Accept multiple naming conventions; prefer 'pdfs'
        files = []
        if "pdfs" in request.files:
            files = request.files.getlist("pdfs")
            log_step(f"Using field 'pdfs' with {len(files)} file(s)")
        elif "file" in request.files:
            files = request.files.getlist("file")  # supports multi even if single
            log_step(f"Using field 'file' with {len(files)} file(s)")
        elif "files[]" in request.files:
            files = request.files.getlist("files[]")
            log_step(f"Using field 'files[]' with {len(files)} file(s)")

        if not files:
            log_step("ERROR: No file fields found among ['pdfs', 'file', 'files[]']")
            return jsonify({"error": "No file part in the request"}), 400

        # Read other form fields
        job_description = (request.form.get("job_description") or "").strip()
        default_vectors_raw = (request.form.get("default_vectors") or "").strip()

        # Optional: parse just for logging / sanity
        try:
            default_vectors_parsed = _json.loads(default_vectors_raw) if default_vectors_raw else []
            log_step(f"default_vectors count = {len(default_vectors_parsed)}")
        except Exception:
            log_step("WARN: default_vectors not valid JSON, using empty list for count")
            default_vectors_parsed = []
            default_vectors_raw = "[]"  # ensure something valid to feed the model

        log_step(f"job_description present? {'yes' if job_description else 'no'}")

        # For response/debugging we can still keep original filenames,
        # but we DO NOT save them here (process_resumes will save once).
        saved_names = []
        for idx, fs in enumerate(files, start=1):
            if not fs or fs.filename == "":
                log_step(f"Skip empty file at index {idx}")
                continue
            safe_name = secure_filename(fs.filename)
            log_step(f"[Accept {idx}/{len(files)}] '{fs.filename}' (safe: '{safe_name}')")
            saved_names.append(safe_name)

        if not saved_names:
            log_step("ERROR: All provided files were empty/missing")
            return jsonify({"error": 'No valid files uploaded'}), 400

        # ===== Your pipeline hooks =====
        # 1) Reset workspace/state
        log_step("Resetting workspace/state")
        reset_workspace()

        # 2) OCR + Markdown + JSON creation for each PDF
        #    process_resumes expects: pdf_files (FileStorage list), groq_client, openai_reasoning
        log_step("Running resume processing pipeline")
        candidates = process_resumes(
            pdf_files=files,
            groq_client=groq_client,
            openai_reasoning=openai_reasoning,
        )
        log_step(f"Resume processing complete for {len(candidates)} candidate(s)")

        # 3) Build vectors/state using JD + default vectors JSON text
        #    build_and_save_state expects:
        #      - job_description
        #      - default_vectors_json_text
        #      - candidates_with_json
        #      - openai_reasoning
        log_step("Building and saving state with vectors & scores")
        build_and_save_state(
            job_description=job_description,
            default_vectors_json_text=default_vectors_raw or "[]",
            candidates_with_json=candidates,
            openai_reasoning=openai_reasoning,
        )
        log_step("State saved")

        # Craft response
        response = {
            "status": "success",
            "uploaded_files": saved_names,
            "count": len(saved_names),
            # "next_url": url_for("candidates"),  # enable if you want the UI to navigate
        }
        log_step("Returning success JSON")
        return jsonify(response), 200

    except Exception as e:
        log_step(f"EXCEPTION in /upload: {e}")
        import sys, traceback
        traceback.print_exc(file=sys.stdout)
        return jsonify({"error": "Internal server error"}), 500


@app.route("/candidates")
def candidates():
    state = load_state()
    if state is None:
        flash("No data found. Please upload resumes.", "error")
        return redirect(url_for("upload"))

    return render_template(
        "candidates.html",
        title="Candidates",
        job=state["job"],
        candidates=state["candidates"],
    )


@app.route("/query", methods=["GET", "POST"])
def query():
    state = load_state()
    if state is None:
        flash("No data found. Please upload resumes first.", "error")
        return redirect(url_for("upload"))

    answer_html = None
    ranked = None
    weights = None

    if request.method == "POST":
        user_query = request.form.get("user_query", "").strip()
        top_k = int(request.form.get("top_k", "5") or 5)

        if not user_query:
            flash("Please enter a query.", "error")
            return redirect(url_for("query"))

        weights, state = generate_vector_weights_for_query(
            user_query=user_query,
            openai_fast=openai_fast,
        )
        ranked = compute_query_scores(weights, state)
        answer = build_answer_for_query(
            user_query=user_query,
            ranked_candidates=ranked,
            top_k=top_k,
            openai_fast=openai_fast,
        )

        # Answer with links instead of raw tags
        top_for_links = ranked[:top_k]
        answer_html = linkify_candidate_tags(answer, top_for_links)

    return render_template(
        "query.html",
        title="Search / Query",
        job=state["job"],
        answer_html=answer_html,
        ranked=ranked,
        weights=weights,
    )


@app.route("/candidate/<int:candidate_id>")
def candidate_detail(candidate_id: int):
    state = load_state()
    if state is None:
        flash("No data found.", "error")
        return redirect(url_for("upload"))

    candidate = next(
        (c for c in state["candidates"] if c["id"] == candidate_id),
        None,
    )
    if candidate is None:
        flash("Candidate not found.", "error")
        return redirect(url_for("candidates"))

    # Load markdown and json for display
    md_path = DATA_DIR / candidate["markdown_path"]
    json_path = DATA_DIR / candidate["json_path"]

    with md_path.open("r", encoding="utf-8") as f:
        md_text = f.read()
    with json_path.open("r", encoding="utf-8") as f:
        json_text = f.read()

    # Just pass relative pdf path; template will link via /files/ route
    pdf_rel = candidate["pdf_path"]

    return render_template(
        "candidate_detail.html",
        title=f"Candidate {candidate_id}",
        candidate=candidate,
        markdown_text=md_text,
        json_text=json_text,
        pdf_rel_path=pdf_rel,
    )


@app.route("/files/<path:subpath>")
def files(subpath: str):
    """
    Serve PDFs/markdown/JSON from the data directory.
    """
    return send_from_directory(DATA_DIR, subpath, as_attachment=False)

def log_step(message):
    print(f"[STEP] {message}", flush=True)

@app.before_request
def _log_before_request():
    log_step(f"→ {request.method} {request.path}")
    # Keep headers/body short to avoid noise; uncomment if needed:
    # log_step(f"Headers: {dict(request.headers)}")

@app.after_request
def _log_after_request(response):
    log_step(f"← {request.method} {request.path} → {response.status}")
    return response

@app.errorhandler(405)
def _method_not_allowed(e):
    log_step(f"ERROR 405 for {request.method} {request.path}")
    return jsonify({"error": "Method Not Allowed"}), 405

@app.errorhandler(404)
def _not_found(e):
    log_step(f"ERROR 404 for {request.method} {request.path}")
    return jsonify({"error": "Not Found"}), 404


if __name__ == "__main__":
    app.run(debug=True)
