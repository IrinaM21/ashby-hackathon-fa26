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

## Real cases from TCGA (`tools/import_tcga.py`)
Replaces the mock cases with real TCGA cases, using only the hackathon's resources:

| App field | Comes from |
|---|---|
| Slide image | A tissue-rich region (~10x) cut from the case's TCGA diagnostic slide |
| Patient, site, stage | TCGA clinical data (GDC API) |
| Expert answer, gross | The case's TCGA pathology report |
| Rubric, ancillary results | Drafted by Qwen2.5-VL from the report text only |
| Reference | Link to the case on the GDC portal |

Run it on Delta (it needs internet access to `api.gdc.cancer.gov`, and a GPU unless you use `--no-model`):
```bash
pip install -r requirements.txt -r tools/requirements.txt
python tools/import_tcga.py --project TCGA-BRCA --n 3                 # replaces cases.json
python tools/import_tcga.py --project TCGA-COAD --n 2 --append        # adds more
python tools/import_tcga.py --case-ids TCGA-A2-A0T2 --append          # a specific case
```
- Slides are large (often 0.5 to 2 GB each). If TCGA slides are preloaded on Delta, add `--slides-dir /path/to/slides` to skip downloading them.
- `--no-model` skips the model. Rubrics then contain only the diagnosis, and scanned reports are not transcribed.
- Downloads go to `data/.tcga_downloads/` (ignored by git). Report text is saved to `data/reports/<case>.txt`.

**Review step:** each imported case has `"review_status": "draft"` and shows a warning in the app. A teammate reads `data/reports/<case>.txt`, fixes the rubric in `cases.json` (labels, keywords, weights, and differentials, which reports rarely list), then sets `"review_status": "reviewed"`.

### Running the importer on Delta (step by step)
```bash
ssh <your-username>@login.delta.ncsa.illinois.edu     # password + Duo
cd /projects/<your-allocation>/$USER  2>/dev/null || cd ~  # a project folder has more space than home
git clone https://github.com/IrinaM21/ashby-hackathon-fa26.git   # first time; later: cd into it and git pull
cd ashby-hackathon-fa26/pathology-tutor-app

module load python                    # or the Python/conda module the hackathon staff recommend
python -m venv ~/tutor-env && source ~/tutor-env/bin/activate
pip install -r requirements.txt -r tools/requirements.txt
export HF_HOME=$PWD/.hf_cache         # keeps the model download out of your small home folder

accounts                              # shows your allocation name, e.g. abcd-delta-gpu
srun --account=<allocation>-delta-gpu --partition=gpuA40x4-interactive --gpus=1 --mem=32g --time=01:00:00 --pty bash
source ~/tutor-env/bin/activate && export HF_HOME=$PWD/.hf_cache
python tools/import_tcga.py --project TCGA-BRCA --n 3
```
If the GPU node can't reach the internet, run `python tools/import_tcga.py --project TCGA-BRCA --n 3 --no-model` on the login node first to download everything, then ask the staff on Slack how GPU jobs get internet access. Partition names and the allocation come from the hackathon's Delta instructions; use theirs if they differ from the ones above.

When it finishes, commit `data/cases.json`, `data/images/tcga/` and `data/reports/`, then push so teammates can pull the real cases.
