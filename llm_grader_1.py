import requests

from confidence import match_percent

def grade_text_only(case, student):

    student_embed = model.embed_text(student)
    exp_embed = model.embed_text(expert)

    m_pct = match_percent(student_embed, exp_embed)

    prompt = f""" 
    You are a pathology tutor. Given the img (patch of scanned slide sample),
    case details, the student's analysis, the ground-truth analysis, and the student's
    grade (% match to expert), explain the student's grade.

    <slide sample>{img}</slide sample>
    <case details>{img}</case details>
    <student analysis>{student}</student analysis>
    <expert analysis>{case["expert_answer"]}</expert analysis>
    <grade>{m_pct}</grade>

    Respond in 4-5 sentences. Point to specific details in the student and expert 
    responses.
    """