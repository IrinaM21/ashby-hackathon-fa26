# confidence.py
import re, torch
from PIL import Image
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize  # verify against repo

_model, _preprocess = create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/conch")
_model.eval().cuda()
_tok = get_tokenizer()

def _split(text):
    return [s.strip() for s in re.split(r"(?<=[.;])\s+|\n+", text) if len(s.strip()) > 15]

@torch.no_grad()
def score_findings(image_path, findings, pool_embs):
    """pool_embs: precomputed (M x D) caption embeddings from other cases / PathGen tiles."""
    sents = _split(findings)
    if not sents:
        return None
    img = _preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).cuda()   # see tiling note
    ie = _model.encode_image(img, proj_contrast=True, normalize=True)
    te = _model.encode_text(tokenize(texts=sents, tokenizer=_tok).cuda(), normalize=True)
    cos = (ie @ te.T).squeeze(0)                                    # per-sentence cosine
    scale = _model.logit_scale.exp()
    pool = (ie @ pool_embs.T).squeeze(0)
    soft = [torch.softmax(scale * torch.cat([c[None], pool]), 0)[0].item() for c in cos]
    return [{"text": s, "cosine": c.item(), "softmax": p} for s, c, p in zip(sents, cos, soft)]