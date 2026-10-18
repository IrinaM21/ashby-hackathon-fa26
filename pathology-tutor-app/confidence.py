# confidence.py
import re, torch
from PIL import Image
from conch.open_clip_custom import get_tokenizer

from utils import Utils

_device = Utils.get_device()
_model, _preprocess = Utils.load_conch(_device)
_tok = get_tokenizer()

def _split(text):
    return [s.strip() for s in re.split(r"(?<=[.;])\s+|\n+", text) if len(s.strip()) > 15]

@torch.no_grad()
def embed_texts(texts):
    """(len(texts) x D) normalized CONCH text embeddings."""
    return _model.encode_text(Utils.conch_tokenize(_tok, texts).to(_device), normalize=True)

def build_pool(texts):
    """Precompute caption embeddings for score_findings' softmax pool."""
    return embed_texts(texts)

@torch.no_grad()
def score_findings(image_path, findings, pool_embs):
    """pool_embs: precomputed (M x D) caption embeddings from other cases / PathGen tiles."""
    sents = _split(findings)
    if not sents:
        return None
    img = _preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(_device)   # see tiling note
    ie = _model.encode_image(img, proj_contrast=True, normalize=True)
    te = embed_texts(sents)
    cos = (ie @ te.T).squeeze(0)                                    # per-sentence cosine
    scale = _model.logit_scale.exp()
    pool = (ie @ pool_embs.T).squeeze(0)
    soft = [torch.softmax(scale * torch.cat([c[None], pool]), 0)[0].item() for c in cos]
    return [{"text": s, "cosine": c.item(), "softmax": p} for s, c, p in zip(sents, cos, soft)]

def match_percent(student_text, reference_text):
    """Percent similarity (0-100) between the student's text and the case's reference text."""
    student_emb, reference_emb = embed_texts([student_text, reference_text])
    return Utils.get_similarity_percent(student_emb, reference_emb)
