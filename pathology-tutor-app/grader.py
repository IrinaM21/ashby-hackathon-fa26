"""Grading logic for the pathology report tutor.

The UI only calls `grade_report(case, report)`. Everything model-related sits
behind `get_checklist()`, so the model owner can swap the stub for Qwen without
touching app.py.

Design (from the plan review): the model only marks each rubric item as
present / partial / missing with a one-line reason. Python turns that checklist
into scores using the weights in cases.json, so scores are reproducible.
"""

import json
import re
from datetime import datetime
from pathlib import Path

from utils import Utils
import logging
from confidence import score_findings, build_pool, match_percent   # import the functions, NOT `import confidence`
from llm_grader import modelchecklist

log = logging.getLogger(__name__)

# Model-confidence settings. Set COS_THRESH from your student-style calibration set.
USE_CONCH = True
COS_THRESH = 0.20          # PLACEHOLDER: tune on real student-style findings
LOW_FRACTION = 0.5         # flag if more than this share of sentences fall below COS_THRESH
CONF_SELF_RATING = 4       # student confidence at or above this counts as "confident"

DATA_DIR = Path(__file__).parent / "data"
POOL = None    

def _get_pool():
    global POOL
    if POOL is None:
        cases = json.loads((DATA_DIR / "cases.json").read_text())["cases"]
        # Placeholder source: swap in whichever field holds each case's reference findings text.
        POOL = build_pool([c["expert_answer"]["summary"] for c in cases])
    return POOL

STATUS_CREDIT = {"present": 1.0, "partial": 0.5, "missing": 0.0}
SECTIONS = ["findings", "diagnosis", "differentials", "ancillary"]
SECTION_LABELS = {
    "findings": "Microscopic findings",
    "diagnosis": "Diagnosis",
    "differentials": "Differential diagnosis",
    "ancillary": "Ancillary studies",
}
REVIEW_QUEUE = Path(__file__).parent / "data" / "review_queue.jsonl"

# Set to False once model_checklist() is implemented.
USE_STUB = True


# ---------------------------------------------------------------------------
# Checklist producers. Each returns a list of
#   {"id": rubric_item_id, "status": "present"|"partial"|"missing", "reason": str}
# ---------------------------------------------------------------------------

def _matches(text, keyword):
    # Short tokens (er, pr, scc, msi) must be whole words; longer ones match as word prefixes.
    pattern = rf"\b{re.escape(keyword)}\b" if len(keyword) <= 4 else rf"\b{re.escape(keyword)}"
    return re.search(pattern, text) is not None


def stub_checklist(case, report):
    """Keyword matcher standing in for the model. Good enough to build the UI against."""
    checklist = []
    for item in case["rubric"]:
        # Findings can show up anywhere in the report; other items are checked in their own field.
        if item["section"] == "findings":
            text = " ".join(report.get(s, "") for s in SECTIONS)
        else:
            text = report.get(item["section"], "")
        text = text.lower()

        hit = next((k for k in item["keywords"] if _matches(text, k.lower())), None)
        partial = next((k for k in item.get("partial_keywords", []) if _matches(text, k.lower())), None)
        if hit:
            checklist.append({"id": item["id"], "status": "present", "reason": f'Mentioned ("{hit}").'})
        elif partial:
            checklist.append({"id": item["id"], "status": "partial", "reason": f'Related term only ("{partial}").'})
        else:
            checklist.append({"id": item["id"], "status": "missing", "reason": "Not addressed in the report."})
    return checklist


def model_checklist(case, report):
    return modelchecklist(case, report)


def get_checklist(case, report):
    if USE_STUB:
        return stub_checklist(case, report)
    try:
        return model_checklist(case, report)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Scoring and feedback (pure Python, no model)
# ---------------------------------------------------------------------------

def score_checklist(case, checklist):
    by_id = {c["id"]: c for c in checklist}
    rows, sections = [], {s: [0.0, 0.0] for s in SECTIONS}
    for item in case["rubric"]:
        c = by_id.get(item["id"], {"status": "missing", "reason": "Not graded."})
        earned = item["weight"] * STATUS_CREDIT.get(c["status"], 0.0)
        sections[item["section"]][0] += earned
        sections[item["section"]][1] += item["weight"]
        rows.append({
            "section": item["section"],
            "label": item["label"],
            "status": c["status"],
            "earned": earned,
            "weight": item["weight"],
            "reason": c["reason"],
        })
    earned = sum(v[0] for v in sections.values())
    total = sum(v[1] for v in sections.values())
    return {
        "items": rows,
        "sections": {s: {"earned": v[0], "total": v[1]} for s, v in sections.items() if v[1]},
        "earned": earned,
        "total": total,
        "percent": round(100 * earned / total) if total else 0,
    }


SOCRATIC_PROMPTS = {
    # Findings hints point at the slide rather than naming the feature.
    "findings": "Re-examine the slide: is there a feature you haven't described that supports or refutes your diagnosis?",
    "diagnosis": "Do your findings add up to a single, specific diagnosis? How would you name it in a report?",
    "differentials": "What else could produce a similar picture, and which feature helps you rule it out?",
    "ancillary": "Is there a test or report element that would confirm or complete this diagnosis?",
}


def socratic_feedback(case, scored):
    """Hints that point at gaps without naming the answer.

    The model owner can replace this with a second Qwen prompt over the same
    checklist; keep the rule that it must not reveal the diagnosis.
    """
    hints = []
    for row in scored["items"]:
        if row["status"] == "present":
            continue
        hint = SOCRATIC_PROMPTS[row["section"]]
        if row["status"] == "partial":
            hint = "You're close. " + hint + " Try to be more specific."
        if hint not in hints:
            hints.append(hint)
    return hints


def grade_report(case, report, elapsed_s=None):
    """Main entry point used by the UI.

    report: dict with keys findings, diagnosis, differentials, ancillary, confidence.
    Returns None if grading failed, else a dict with scores, checklist and feedback.
    """
    checklist = get_checklist(case, report)
    if checklist is None:
        return None
    scored = score_checklist(case, checklist)
    scored["feedback"] = socratic_feedback(case, scored)
    scored["missed"] = [r["label"] for r in scored["items"] if r["status"] == "missing"]

    # Minimal human-in-the-loop: flag low scores, or high confidence with a missed diagnosis.
    confidence = report.get("confidence") or 0
    diag_missed = any(r["section"] == "diagnosis" and r["status"] != "present" for r in scored["items"])
    reasons = []
    if scored["percent"] < 50:
        reasons.append("score below 50%")
    if diag_missed and confidence >= 4:
        reasons.append("confident but diagnosis not matched")

    # --- CONCH image-text confidence on the findings (never blocks grading) ---
    model_conf = None
    if USE_CONCH and report.get("findings", "").strip():
        try:
            img_path = DATA_DIR / case["image"]
            model_conf = score_findings(img_path, report["findings"], _get_pool())
        except Exception:
            log.exception("CONCH confidence scoring failed for case %s", case.get("id"))
    scored["model_confidence"] = model_conf    # app.py ignores extra keys

    # --- CONCH text match: student report vs this case's expert answer (shown alongside the rubric score) ---
    scored["match_percent"] = None
    student_text = " ".join(report.get(s, "").strip() for s in ("diagnosis", "findings")).strip()
    if USE_CONCH and student_text:
        try:
            ans = case["expert_answer"]
            scored["match_percent"] = round(match_percent(student_text, f"{ans['diagnosis']}. {ans['summary']}"))
        except Exception:
            log.exception("CONCH match scoring failed for case %s", case.get("id"))

    if model_conf:
        low = [s for s in model_conf if s["cosine"] < COS_THRESH]
        if len(low) / len(model_conf) > LOW_FRACTION:
            reasons.append("findings text poorly matches the slide")
        if low and confidence >= CONF_SELF_RATING:
            reasons.append("confident, but description fits the slide weakly")
        if low:
            # Hint only: do not name the diagnosis. Quoting the student's own sentence is fine.
            scored["feedback"].append(
                "Part of your description may not match what's on this slide. "
                f'Re-read this against the image: "{low[0]["text"]}"'
            )

    scored["flag_reasons"] = reasons
    if reasons:
        _queue_for_review(case, report, scored, elapsed_s)
    return scored


def _queue_for_review(case, report, scored, elapsed_s):
    entry = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "case_id": case["id"],
        "percent": scored["percent"],
        "reasons": scored["flag_reasons"],
        "model_confidence": scored.get("model_confidence"),   # new
        "elapsed_s": elapsed_s,
        "report": report,
    }
    try:
        with REVIEW_QUEUE.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass
