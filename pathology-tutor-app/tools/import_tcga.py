"""Build tutor cases from TCGA: real slides, real pathology reports, real clinical data.

For each TCGA case that has both a pathology report and a diagnostic (FFPE) slide:
  1. clinical context (age, sex, site, stage, diagnosis) comes from the GDC API
  2. the pathology report PDF is downloaded and its text extracted
     (text layer first; scanned pages are transcribed with the vision model)
  3. one tissue-rich image is cut from the whole-slide image
  4. a model drafts the rubric from the report text (findings, diagnosis, tests)

Every case is written with "review_status": "draft". A teammate should read the
rubric against the report text (saved alongside) before it is used for grading.

Run on Delta (needs internet to api.gdc.cancer.gov and a GPU for the model):

    python tools/import_tcga.py --project TCGA-BRCA --n 3
    python tools/import_tcga.py --project TCGA-COAD --n 2 --append
    python tools/import_tcga.py --case-ids TCGA-A2-A0T2 TCGA-BH-A0BZ --append
    python tools/import_tcga.py --project TCGA-BRCA --n 3 --slides-dir /path/to/preloaded/tcga  # skip slide downloads
    python tools/import_tcga.py --project TCGA-BRCA --n 3 --no-model   # no GPU: diagnosis-only rubric
"""

import argparse
import json
import re
import sys
from pathlib import Path

import requests

GDC = "https://api.gdc.cancer.gov"
APP_DIR = Path(__file__).resolve().parent.parent
DATA = APP_DIR / "data"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

CASE_FIELDS = [
    "case_id", "submitter_id", "primary_site", "project.project_id",
    "demographic.gender", "demographic.age_at_index",
    "diagnoses.age_at_diagnosis", "diagnoses.primary_diagnosis", "diagnoses.tissue_or_organ_of_origin",
    "diagnoses.site_of_resection_or_biopsy", "diagnoses.ajcc_pathologic_stage", "diagnoses.tumor_grade",
]
FILE_FIELDS = ["file_id", "file_name", "file_size", "data_type", "experimental_strategy", "cases.submitter_id"]


# ---------------------------------------------------------------------------
# GDC API
# ---------------------------------------------------------------------------

def _eq(field, value):
    op = "in" if isinstance(value, list) else "="
    return {"op": op, "content": {"field": field, "value": value}}


def gdc_post(endpoint, filters, fields, size):
    body = {"filters": {"op": "and", "content": filters}, "fields": ",".join(fields), "format": "JSON", "size": size}
    r = requests.post(f"{GDC}/{endpoint}", json=body, timeout=60)
    r.raise_for_status()
    return r.json()["data"]["hits"]


def find_cases(project, n, case_ids=None):
    # Only filter by project here; whether a case has a report and a slide is checked per case,
    # which is more forgiving than nested file filters on the cases endpoint.
    filters = [_eq("submitter_id", case_ids) if case_ids else _eq("project.project_id", project)]
    return gdc_post("cases", filters, CASE_FIELDS, size=len(case_ids) if case_ids else max(n * 5, 20))


def _is_diagnostic(f):
    # FFPE diagnostic slides are labelled "Diagnostic Slide" and have "-DX" in the file name.
    return f.get("experimental_strategy") == "Diagnostic Slide" or "-DX" in f.get("file_name", "").upper()


def case_files(submitter_id):
    hits = gdc_post("files", [_eq("cases.submitter_id", submitter_id), _eq("access", "open")], FILE_FIELDS, size=200)
    report = next((f for f in hits if f.get("data_type") == "Pathology Report"), None)
    slides = [f for f in hits if f.get("data_type") == "Slide Image" and _is_diagnostic(f)]
    slide = min(slides, key=lambda f: f.get("file_size", 0)) if slides else None  # smallest = quickest download
    if not (report and slide):
        types = sorted({f"{f.get('data_type')}/{f.get('experimental_strategy')}" for f in hits})
        print(f"   open files found: {len(hits)} ({', '.join(types[:8]) or 'none'})")
    return report, slide


def download(file_id, dest):
    with requests.get(f"{GDC}/data/{file_id}", stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return dest


# ---------------------------------------------------------------------------
# Clinical context
# ---------------------------------------------------------------------------

def _known(value):
    return value and str(value).lower() not in ("not reported", "unknown", "none", "--")


def clinical_from_gdc(case):
    dx = (case.get("diagnoses") or [{}])[0]
    demo = case.get("demographic") or {}
    age = demo.get("age_at_index")
    if age is None and dx.get("age_at_diagnosis") is not None:
        age = int(dx["age_at_diagnosis"] // 365.25)
    sex = {"female": "woman", "male": "man"}.get(str(demo.get("gender", "")).lower(), "patient")
    site = dx.get("site_of_resection_or_biopsy") or dx.get("tissue_or_organ_of_origin") or case.get("primary_site", "")
    history = [f"Primary site: {case.get('primary_site', 'not recorded')}."]
    if _known(dx.get("ajcc_pathologic_stage")):
        history.append(f"Pathologic stage: {dx['ajcc_pathologic_stage']}.")
    return {
        "age_sex": f"{age}-year-old {sex}" if age is not None else sex.capitalize(),
        "history": " ".join(history),
        "specimen": f"Resection / biopsy, {site}" if _known(site) else "Not recorded",
        "gross": "See report",
    }, dx


# ---------------------------------------------------------------------------
# Report text
# ---------------------------------------------------------------------------

def report_text(pdf_path, model):
    from pypdf import PdfReader

    text = "\n".join((p.extract_text() or "") for p in PdfReader(str(pdf_path)).pages).strip()
    if len(text) > 200:
        return text, "pdf text layer"
    if model is None:
        return text, "scanned PDF, not transcribed (run with the model to OCR it)"

    import pypdfium2 as pdfium
    pages = []
    for page in pdfium.PdfDocument(str(pdf_path)):
        img = page.render(scale=2).to_pil()
        pages.append(model.generate(
            "Transcribe all text on this pathology report page exactly. Output only the text.",
            image=img, max_new_tokens=1500))
    return "\n".join(pages).strip(), f"transcribed by {model.name}"


# ---------------------------------------------------------------------------
# Slide image
# ---------------------------------------------------------------------------

def find_local_slide(slides_dir, file_name):
    if not slides_dir:
        return None
    return next(Path(slides_dir).rglob(file_name), None)


def tissue_patch(svs_path, out_path, size=1024, target_mpp=1.0):
    """Cut the most tissue-rich size x size region at about 10x (1 micron per pixel)."""
    import numpy as np
    import openslide

    slide = openslide.OpenSlide(str(svs_path))
    mpp = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.25) or 0.25)
    level = slide.get_best_level_for_downsample(target_mpp / mpp)
    ds = slide.level_downsamples[level]

    # Find tissue on a small thumbnail: tissue is coloured, background is near-white.
    thumb = slide.get_thumbnail((1024, 1024)).convert("RGB")
    arr = np.asarray(thumb).astype(float)
    tissue = (arr.max(axis=2) - arr.min(axis=2)) > 25
    scale = slide.dimensions[0] / thumb.width
    win = max(1, int(size * ds / scale))  # patch size in thumbnail pixels
    best, best_xy = -1.0, (0, 0)
    for y in range(0, max(1, tissue.shape[0] - win), max(1, win // 2)):
        for x in range(0, max(1, tissue.shape[1] - win), max(1, win // 2)):
            score = tissue[y:y + win, x:x + win].mean()
            if score > best:
                best, best_xy = score, (x, y)

    x0, y0 = int(best_xy[0] * scale), int(best_xy[1] * scale)
    region = slide.read_region((x0, y0), level, (size, size)).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    region.save(out_path)
    thumb.save(out_path.with_name(out_path.stem + "_overview.png"))
    return {"level": level, "x": x0, "y": y0, "tissue_fraction": round(best, 2)}


# ---------------------------------------------------------------------------
# Model (Qwen2.5-VL, used for OCR of scanned reports and for drafting rubrics)
# ---------------------------------------------------------------------------

class Model:
    def __init__(self, name):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.name = name
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            name, torch_dtype=torch.float16, device_map="auto")

    def generate(self, prompt, image=None, max_new_tokens=800):
        content = ([{"type": "image"}] if image is not None else []) + [{"type": "text", "text": prompt}]
        text = self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                  tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image] if image is not None else None,
                                return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


RUBRIC_PROMPT = """You are helping build a pathology teaching case from a real pathology report.
Use ONLY information stated in the report. Do not add anything that is not in it.

Return JSON only, with this shape:
{{
  "final_diagnosis": "the diagnosis line as written in the report",
  "diagnosis_keywords": ["3-5 ways a student could name this diagnosis, including common synonyms"],
  "summary": "two sentences summarising the microscopic description and diagnosis",
  "gross": "one sentence from the gross description, or empty",
  "findings": [{{"label": "microscopic feature", "keywords": ["2-4 short words a student might write"]}}],
  "ancillary": [{{"test": "name of stain / test", "result": "result as written", "keywords": ["test name variants"]}}],
  "other_items": [{{"label": "e.g. margin status, lymph node status", "keywords": ["..."]}}]
}}
Give at most 5 findings, 4 ancillary tests and 3 other items.

REPORT:
{report}
"""


def parse_json(raw):
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise ValueError("no JSON object in model output")
    return json.loads(match.group(0))


def draft_with_model(model, report):
    prompt = RUBRIC_PROMPT.format(report=report[:6000])
    for attempt in range(2):
        try:
            return parse_json(model.generate(prompt, max_new_tokens=900))
        except (ValueError, json.JSONDecodeError):
            if attempt:
                return None


def _keywords(phrase):
    # Fallback keywords: the phrase itself plus its distinctive words.
    words = [w for w in re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", phrase.lower())
             if w not in {"with", "nos", "type", "carcinoma", "tumor", "tumour", "invasive", "malignant"}]
    return [phrase.lower()] + words[:3]


def build_rubric(primary_dx, draft):
    dx_keywords = _keywords(primary_dx) + [k.lower() for k in (draft or {}).get("diagnosis_keywords", [])]
    rubric = [{"id": "d1", "section": "diagnosis", "label": (draft or {}).get("final_diagnosis") or primary_dx,
               "weight": 4, "keywords": dx_keywords, "partial_keywords": ["carcinoma", "malignan"]}]
    if not draft:
        return rubric
    for i, f in enumerate(draft.get("findings", [])[:5], 1):
        rubric.append({"id": f"f{i}", "section": "findings", "label": f["label"], "weight": 2,
                       "keywords": [k.lower() for k in f.get("keywords") or _keywords(f["label"])],
                       "partial_keywords": []})
    for i, t in enumerate(draft.get("ancillary", [])[:4], 1):
        rubric.append({"id": f"a{i}", "section": "ancillary", "label": t["test"], "weight": 1,
                       "keywords": [k.lower() for k in t.get("keywords") or _keywords(t["test"])],
                       "partial_keywords": []})
    for i, o in enumerate(draft.get("other_items", [])[:3], 1):
        rubric.append({"id": f"o{i}", "section": "ancillary", "label": o["label"], "weight": 1,
                       "keywords": [k.lower() for k in o.get("keywords") or _keywords(o["label"])],
                       "partial_keywords": []})
    return rubric


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_case(case, model, slides_dir, work, min_report_chars=0):
    sid = case["submitter_id"]
    print(f"\n== {sid}")
    report_file, slide_file = case_files(sid)
    if not (report_file and slide_file):
        print("   skipped: missing open-access report or diagnostic slide")
        return None

    clinical, dx = clinical_from_gdc(case)
    primary_dx = dx.get("primary_diagnosis") or "Not recorded"

    pdf = download(report_file["file_id"], work / report_file["file_name"])
    text, text_source = report_text(pdf, model)
    print(f"   report: {len(text)} chars ({text_source})")
    if len(text) < min_report_chars:
        print(f"   skipped: report shorter than {min_report_chars} chars, rubric would be thin")
        return None
    reports_dir = DATA / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{sid}.txt").write_text(text)

    svs = find_local_slide(slides_dir, slide_file["file_name"])
    if svs is None:
        mb = slide_file.get("file_size", 0) / 1e6
        print(f"   downloading slide {slide_file['file_name']} ({mb:.0f} MB)")
        svs = download(slide_file["file_id"], work / slide_file["file_name"])
    image_rel = f"images/tcga/{sid}.png"
    patch = tissue_patch(svs, DATA / image_rel)
    print(f"   image: {image_rel} (tissue {patch['tissue_fraction']:.0%})")

    draft = draft_with_model(model, text) if (model and len(text) > 200) else None
    if model and draft is None:
        print("   rubric: model output unusable, diagnosis-only rubric")
    if draft and draft.get("gross"):
        clinical["gross"] = draft["gross"]

    portal = f"https://portal.gdc.cancer.gov/cases/{case['case_id']}"
    return {
        "id": sid.lower(),
        "title": f"{sid}: {case.get('primary_site', '')}".strip(": "),
        "organ": case.get("primary_site", ""),
        "time_limit_min": 10,
        "review_status": "draft",
        "image": image_rel,
        "image_source": f"TCGA {slide_file['file_name']} via NCI GDC",
        "image_license": "TCGA open access (NIH GDC data use policy)",
        "clinical": clinical,
        "rubric": build_rubric(primary_dx, draft),
        "expert_answer": {
            "diagnosis": (draft or {}).get("final_diagnosis") or primary_dx,
            "summary": (draft or {}).get("summary") or "See the full pathology report.",
        },
        "ancillary_results": {t["test"]: t.get("result", "") for t in (draft or {}).get("ancillary", [])},
        "references": [f"TCGA case {sid}, NCI Genomic Data Commons: {portal}"],
        "source": {
            "tcga_case": sid, "gdc_case_id": case["case_id"], "project": case["project"]["project_id"],
            "report_file_id": report_file["file_id"], "slide_file_id": slide_file["file_id"],
            "report_text": f"reports/{sid}.txt", "report_text_source": text_source,
            "patch": patch,
        },
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default="TCGA-BRCA", help="e.g. TCGA-BRCA, TCGA-COAD, TCGA-LUAD")
    p.add_argument("--n", type=int, default=3, help="number of cases")
    p.add_argument("--case-ids", nargs="*", help="specific TCGA case ids instead of the first n")
    p.add_argument("--slides-dir", help="folder with preloaded .svs files (searched recursively)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--no-model", action="store_true", help="skip OCR and rubric drafting (no GPU needed)")
    p.add_argument("--min-report-chars", type=int, default=1500,
                   help="skip cases whose report text is shorter than this (0 keeps all)")
    p.add_argument("--append", action="store_true", help="add to cases.json instead of replacing it")
    p.add_argument("--out", default=str(DATA / "cases.json"))
    p.add_argument("--work-dir", default=str(DATA / ".tcga_downloads"))
    args = p.parse_args(argv)

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    model = None if args.no_model else Model(args.model)

    cases = find_cases(args.project, args.n, args.case_ids)
    print(f"Found {len(cases)} candidate case(s) in {args.case_ids or args.project}")
    built = []
    for case in cases:
        if len(built) >= (len(args.case_ids) if args.case_ids else args.n):
            break
        c = build_case(case, model, args.slides_dir, work, args.min_report_chars)
        if c:
            built.append(c)

    out = Path(args.out)
    if not built:
        print(f"\nNo cases built, so {out} was left unchanged.")
        return 1
    existing = json.loads(out.read_text()) if (args.append and out.exists()) else {"cases": []}
    ids = {c["id"] for c in built}
    existing["cases"] = [c for c in existing["cases"] if c["id"] not in ids] + built
    existing["_note"] = ("Cases with review_status 'draft' have model-drafted rubrics: check each rubric "
                         "against data/reports/<case>.txt and set review_status to 'reviewed'.")
    out.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
    print(f"\nWrote {len(built)} case(s) to {out}")
    return 0 if built else 1


if __name__ == "__main__":
    sys.exit(main())
