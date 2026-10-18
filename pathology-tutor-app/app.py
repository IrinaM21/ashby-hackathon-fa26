"""Pathology report-writing tutor: Gradio frontend.

Runs entirely on mock cases (data/cases.json) and the stub grader in grader.py.
Swap in the real model by implementing grader.model_checklist and setting
grader.USE_STUB = False.

    python app.py            # local, http://127.0.0.1:7860
    python app.py --share    # public link (Colab)
"""

import argparse
import json
import time
from pathlib import Path

import gradio as gr

import grader
from utils import Utils
import logging

log = logging.getLogger(__name__)

DATA = Path(__file__).parent / "data"
CASES = json.loads((DATA / "cases.json").read_text())["cases"]
CASE_BY_TITLE = {c["title"]: c for c in CASES}

DISCLAIMER = (
    "**Educational use only. Not for clinical diagnosis.** Cases and feedback are for teaching. "
    "Pathology foundation models used here (e.g. MUSK, CONCH) are licensed for non-commercial research use only."
)
# A word is marked when its raw CONCH score (cosine units, see Utils.gradcam_text) is at or below this.
# -0.10 from a 3-case check: catches most wrong words but also marks some correct ones. Tune on real reports.
MISTAKE_THRESHOLD = -0.10
# Never marked: carry no pathology meaning. Negations ("no", "not", "without") are deliberately not here.
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "with", "in", "on", "at", "to", "for", "from", "by", "as",
    "is", "are", "was", "were", "be", "been", "this", "that", "these", "those", "it", "its", "there", "has", "have",
}

CSS = """
#timer, #timer p {font-size: 1.6em; font-weight: 600; text-align: right;}
#score {font-size: 1.3em;}
.flag {padding: 8px 12px; border-radius: 8px; background: #fdecea; color: #8a1c12;}
"""


def clinical_md(case):
    c = case["clinical"]
    md = (
        f"### {case['title']} · {case['organ']}\n"
        f"**Patient:** {c['age_sex']}  \n"
        f"**History:** {c['history']}  \n"
        f"**Specimen:** {c['specimen']}  \n"
        f"**Gross:** {c['gross']}\n\n"
    )
    if case.get("review_status") == "draft":
        md += "⚠️ *Draft rubric from the source report, not yet reviewed by the team.*\n\n"
    return md + f"<sub>Image: {case['image_source']} ({case['image_license']})</sub>"


def wording_md(text, words):
    """The student's text with words that pull away from the expert answer (per CONCH) marked as likely mistakes."""
    if not text.strip():
        return "—"
    out, pos = [], 0
    for w in words:
        out.append(text[pos:w["start"]])
        word = text[w["start"]:w["end"]]
        mistake = w["raw"] <= MISTAKE_THRESHOLD and word.lower() not in STOPWORDS
        out.append(f"**⚠ {word}**" if mistake else word)
        pos = w["end"]
    out.append(text[pos:])
    return "".join(out)


def fmt_clock(seconds):
    seconds = int(abs(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

def start_case(title):
    case = CASE_BY_TITLE[title]
    session = {"case_id": case["id"], "title": title, "start": time.time(), "submitted": False}
    limit = case["time_limit_min"] * 60
    return (
        session,
        gr.Column(visible=True),                       # case panel
        str(DATA / case["image"]),
        clinical_md(case),
        f"⏱ {fmt_clock(limit)}",
        gr.Timer(active=True),
        "", "", "", "", 3,                             # clear the form
        gr.Button(interactive=True),
        gr.Column(visible=False),                      # hide old results
    )


def tick(session):
    if not session or session.get("submitted"):
        return gr.Timer(active=False), gr.update()
    case = CASE_BY_TITLE[session["title"]]
    remaining = case["time_limit_min"] * 60 - (time.time() - session["start"])
    if remaining >= 0:
        return gr.Timer(), f"⏱ {fmt_clock(remaining)}"
    return gr.Timer(), f"⏱ Time's up (+{fmt_clock(remaining)})"


def submit_report(session, findings, diagnosis, differentials, ancillary, confidence, history):
    if not session:
        raise gr.Error("Start a case first.")
    if not (findings.strip() or diagnosis.strip()):
        raise gr.Error("Write at least your findings or a diagnosis before submitting.")

    case = CASE_BY_TITLE[session["title"]]
    elapsed = round(time.time() - session["start"])
    report = {
        "findings": findings, "diagnosis": diagnosis, "differentials": differentials,
        "ancillary": ancillary, "confidence": confidence,
    }
    result = grader.grade_report(case, report, elapsed_s=elapsed)
    if result is None:
        raise gr.Error("Couldn't grade this report. Please resubmit.")

    session = {**session, "submitted": True}
    over = elapsed > case["time_limit_min"] * 60

    score = f"## Score: {result['earned']:g} / {result['total']:g} ({result['percent']}%)\n"
    score += f"Time taken: {fmt_clock(elapsed)}" + (" (over the time limit)" if over else "")

    sections = [
        [grader.SECTION_LABELS[s], f"{v['earned']:g} / {v['total']:g}"]
        for s, v in result["sections"].items()
    ]
    word_scores = result.get("word_scores", {})
    checklist = [
        [grader.SECTION_LABELS[r["section"]],
         wording_md(report.get(r["section"], ""), word_scores.get(r["section"], [])),
         f"{r['earned']:g} / {r['weight']:g}"]
        for r in result["items"]
    ]

    if result["feedback"]:
        feedback = "### Things to think about\n" + "\n".join(f"- {h}" for h in result["feedback"])
    else:
        feedback = "### Great work\nYour report covered every rubric item."

    if result["flag_reasons"]:
        flag = f"<div class='flag'>🚩 Sent for faculty review: {', '.join(result['flag_reasons'])}.</div>"
    else:
        flag = ""

    tutor = result.get("tutor_feedback")
    if tutor and tutor.get("explanation"):
        tutor_md_text = (f"### Tutor feedback\n**Match to expert (approximate): {tutor['match_percent']}%**\n\n"
                         f"{tutor['explanation']}")
    else:
        tutor_md_text = "### Tutor feedback\n*Tutor feedback isn't available for this report.*"

    ans = case["expert_answer"]
    reveal = f"**Expert diagnosis:** {ans['diagnosis']}\n\n{ans['summary']}\n\n"
    if result["missed"]:
        reveal += "**Rubric items you missed:**\n" + "\n".join(f"- {m}" for m in result["missed"]) + "\n\n"
    reveal += "**Ancillary results:**\n" + "\n".join(f"- {k}: {v}" for k, v in case["ancillary_results"].items())
    reveal += "\n\n**References:**\n" + "\n".join(f"- {r}" for r in case["references"])

    overlay_img = None
    if case.get("heatmap"):                       # render_heatmap raises KeyError without it
        try:
            overlay_img = Utils.render_heatmap(case, data_dir=DATA)
        except Exception:
            log.exception("heatmap render failed for %s", case["id"])
    if overlay_img is not None:
        reveal += ("\n\n**Model attention:** red marks regions an image-text model matched more closely "
                   "to the expert diagnosis, and blue marks weaker or no match. The greyed area was not "
                   "analysed; the white box is the region the model saw. This is an automatically "
                   "generated, approximate guide. It is not a pathologist's annotation.")
    history = history + [[case["title"], f"{result['percent']}%", fmt_clock(elapsed), "🚩" if flag else ""]]

    return (
        session,
        gr.Timer(active=False),
        gr.Column(visible=True),
        score, sections, checklist, feedback, flag, reveal,
        history, history,
        gr.Button(interactive=False),
        gr.Image(value=overlay_img, visible=overlay_img is not None),
        tutor_md_text,
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="Pathology Report Tutor") as demo:
    session = gr.State(None)
    history = gr.State([])
    timer = gr.Timer(1.0, active=False)

    gr.Markdown("# 🔬 Pathology Report Tutor\nRead the case, write a structured report, and get rubric-based feedback.")
    gr.Markdown(DISCLAIMER)

    with gr.Row(equal_height=True):
        case_dd = gr.Dropdown(list(CASE_BY_TITLE), value=CASES[0]["title"], label="Choose a case", scale=3)
        start_btn = gr.Button("Start case", variant="primary", scale=1)
        timer_md = gr.Markdown("⏱ --:--", elem_id="timer")

    with gr.Column(visible=False) as case_panel:
        with gr.Row():
            with gr.Column(scale=3):
                image = gr.Image(label="Slide", type="filepath", interactive=False, height=420)
            with gr.Column(scale=2):
                clinical = gr.Markdown()

        gr.Markdown("## Your report")
        findings = gr.Textbox(label="Microscopic findings", lines=5,
                              placeholder="Describe architecture, cytology, stroma, background...")
        diagnosis = gr.Textbox(label="Diagnosis", lines=1, placeholder="Your final diagnosis")
        with gr.Row():
            differentials = gr.Textbox(label="Differential diagnosis", lines=3,
                                       placeholder="What else did you consider, and why is it less likely?")
            ancillary = gr.Textbox(label="Ancillary studies / comments", lines=3,
                                   placeholder="IHC, molecular tests, margins, recommendations...")
        confidence = gr.Slider(1, 5, value=3, step=1, label="How confident are you? (1 = guessing, 5 = certain)")
        submit_btn = gr.Button("Submit report", variant="primary")

    with gr.Column(visible=False) as results_panel:
        gr.Markdown("## Results")
        score_md = gr.Markdown(elem_id="score")
        flag_md = gr.HTML()
        with gr.Row():
            sections_df = gr.Dataframe(headers=["Section", "Points"], interactive=False, label="By section", scale=1)
            feedback_md = gr.Markdown()
        tutor_md = gr.Markdown()
        checklist_df = gr.Dataframe(headers=["Section", "Your wording", "Points"],
                                    datatype=["str", "markdown", "str"],
                                    interactive=False, wrap=True,
                                    label="Rubric checklist (⚠ = words that pull your report away from the expert's)")
        with gr.Accordion("Reveal expert answer", open=False):
            reveal_md = gr.Markdown()
            heatmap_img = gr.Image(label="Model attention (white box = region analysed)",
                                   type="pil", interactive=False, visible=False, height=420)

    with gr.Accordion("This session's scores", open=False):
        history_df = gr.Dataframe(headers=["Case", "Score", "Time", "Flagged"], interactive=False)

    start_btn.click(
        start_case, inputs=case_dd,
        outputs=[session, case_panel, image, clinical, timer_md, timer,
                 findings, diagnosis, differentials, ancillary, confidence, submit_btn, results_panel],
    )
    timer.tick(tick, inputs=session, outputs=[timer, timer_md])
    submit_btn.click(
        submit_report,
        inputs=[session, findings, diagnosis, differentials, ancillary, confidence, history],
        outputs=[session, timer, results_panel, score_md, sections_df, checklist_df, feedback_md,
                 flag_md, reveal_md, history, history_df, submit_btn, heatmap_img, tutor_md],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    demo.launch(share=args.share, server_port=args.port, theme=gr.themes.Soft(), css=CSS)
