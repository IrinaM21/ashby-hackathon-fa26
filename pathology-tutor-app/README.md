# Pathology Report Tutor (frontend)

Gradio app that shows a timed case (slide + clinical context), takes a structured report, and shows rubric scores with Socratic feedback. It runs on **mock data and a keyword stub grader**, so the UI works before any model is loaded.

## Files
- `app.py`: the Gradio UI. Nobody but the frontend owner needs to touch it.
- `grader.py`: grading. `grade_report(case, report)` is the only function the UI calls.
- `data/cases.json`: cases, rubric items with point weights, expert answer, ancillary results, references. **All mock text, needs a pathologist-minded check.**
- `data/images/`: placeholder slides. Regenerate with `python make_placeholders.py`; replace with licensed H&E images and fill in `image_source` / `image_license`.
- `data/review_queue.jsonl`: created when a submission is flagged for faculty review.

## Run in VS Code
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                     # open http://127.0.0.1:7860
```
Use **Python: Select Interpreter** to pick `.venv`. Run from inside this folder so `data/` resolves.

## Run in Colab
```python
!git clone <your-repo-url>
%cd <repo>/app
!pip install -q -r requirements.txt
!python app.py --share           # prints a public gradio.live link
```
Share links expire after 72 hours and change on restart.

## Plugging in the model
1. In `grader.py`, implement `model_checklist(case, report)`. Return a list of
   `{"id": "<rubric item id>", "status": "present" | "partial" | "missing", "reason": "one line"}`, one per rubric item.
2. Set `USE_STUB = False`.

Python computes the score from the weights in `cases.json`, so the model never outputs numbers. If `model_checklist` raises, the UI shows "Couldn't grade this report. Please resubmit." `socratic_feedback()` can be swapped for a second model prompt the same way; keep the rule that it must not reveal the diagnosis.

## Adding a case
Copy a case in `cases.json`, give it a unique `id`, and write rubric items with a `section` (`findings`, `diagnosis`, `differentials`, `ancillary`), a `weight`, and `keywords` / `partial_keywords` for the stub grader.
