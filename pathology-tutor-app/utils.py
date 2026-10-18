import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw
from timm.models import create_model
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer

from musk import utils as musk_utils
from musk import modeling  # noqa: F401  registers musk_large_patch16_384 with timm


class Utils:
    HF_REPO_ID = "xiangjx/musk"
    HF_FILENAME = "model.safetensors"
    MODEL_NAME = "musk_large_patch16_384"

    @staticmethod
    def get_hf_token():
        """Return the Hugging Face token from the HF_TOKEN environment variable, or None if unset."""
        return os.environ.get("HF_TOKEN") or None

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def load_musk(device=None, dtype=None):
        """Load the pretrained MUSK model from https://huggingface.co/xiangjx/musk.

        Defaults to float16 on CUDA and float32 elsewhere.
        """
        device = torch.device(device) if device is not None else Utils.get_device()
        if dtype is None:
            dtype = torch.float16 if device.type == "cuda" else torch.float32

        ckpt_path = hf_hub_download(Utils.HF_REPO_ID, filename=Utils.HF_FILENAME, token=Utils.get_hf_token())

        model = create_model(Utils.MODEL_NAME)
        musk_utils.load_model_and_may_interpolate(ckpt_path, model, "model|module", "")
        model.to(device=device, dtype=dtype)
        model.eval()
        return model

    CONCH_REPO_ID = "MahmoodLab/conch"
    CONCH_FILENAME = "pytorch_model.bin"
    CONCH_MODEL_CFG = "conch_ViT-B-16"
    CONCH_LOCAL_PATH = Path(__file__).parent / "weights" / "conch.pt"
    # The public release has no caption decoder weights (removed by MahmoodLab), so CONCH builds the
    # decoder with random weights. Trained decoder weights would go here; save_conch_weights doesn't write it.
    CONCH_DECODER_PATH = Path(__file__).parent / "weights" / "conch_text_decoder.pt"

    @staticmethod
    def load_conch(device=None, weights_path=CONCH_LOCAL_PATH, decoder_path=None):
        """Load the pretrained CONCH model.

        Uses weights_path (written by save_conch_weights) if it exists, so no Hugging Face call;
        otherwise downloads from https://huggingface.co/MahmoodLab/conch. Pass weights_path=None to force the download.
        The caption decoder isn't used by encode_image / encode_text, so it's only loaded when asked for:
        pass decoder_path (e.g. Utils.CONCH_DECODER_PATH) to load it into model.text_decoder; otherwise it stays
        randomly initialized.
        Returns (model, preprocess), where preprocess turns a PIL image into the model's input tensor.
        """
        device = torch.device(device) if device is not None else Utils.get_device()
        if weights_path is not None and Path(weights_path).exists():
            ckpt_path = str(weights_path)
        else:
            ckpt_path = hf_hub_download(Utils.CONCH_REPO_ID, filename=Utils.CONCH_FILENAME, token=Utils.get_hf_token())

        model, preprocess = create_model_from_pretrained(Utils.CONCH_MODEL_CFG, checkpoint_path=ckpt_path, device=device)
        if decoder_path is not None and Path(decoder_path).exists():
            model.text_decoder.load_state_dict(torch.load(decoder_path, map_location=device), strict=True)
        model.eval()
        return model, preprocess

    @staticmethod
    def conch_tokenize(tokenizer, texts):
        """Same token ids as conch.open_clip_custom.tokenize, which needs transformers<5 (batch_encode_plus).

        CONCH's context is 128 tokens, with the last reserved for <cls>: encode to 127, then pad one more.
        """
        ids = tokenizer(
            list(texts),
            max_length=127,
            add_special_tokens=True,
            return_token_type_ids=False,
            truncation=True,
            padding="max_length",
        )["input_ids"]
        return F.pad(torch.tensor(ids), (0, 1), value=tokenizer.pad_token_id)

    @staticmethod
    def save_conch_weights(path=CONCH_LOCAL_PATH, dtype=torch.float32):
        """Download CONCH from Hugging Face once and save the released weights (vision + text encoders) to path.

        Identical to the Hugging Face checkpoint at the default fp32; dtype=torch.float16 halves the file size.
        The untrained caption decoder (text_decoder.*) is left out, as in the released checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model, _ = Utils.load_conch(device="cpu", weights_path=None)
        encoders = {
            k: v.to(dtype) if v.is_floating_point() else v
            for k, v in model.state_dict().items()
            if not k.startswith("text_decoder.")
        }
        torch.save(encoders, path)
        return path

    @staticmethod
    def _generated_token_probs(model_output):
        """Return (generated_ids, token_probs), each (batch, steps), for every generated step."""
        logits = torch.stack(model_output.logits, dim=1)  # (batch, steps, vocab)
        generated_ids = model_output.sequences[:, -logits.shape[1]:]  # (batch, steps)

        probs = torch.softmax(logits.float(), dim=-1)
        token_probs = probs.gather(-1, generated_ids.unsqueeze(-1)).squeeze(-1)  # (batch, steps)
        return generated_ids, token_probs

    @staticmethod
    def get_token_probs(model_output, batch_index=0):
        """Map each generated token id to the probability the model assigned it.

        model_output: result of model.generate(..., return_dict_in_generate=True, output_logits=True),
        with .sequences (token ids) and .logits (one (batch, vocab) tensor per generated step).
        Repeated token ids share a key, so only the last occurrence's probability is kept.
        """
        generated_ids, token_probs = Utils._generated_token_probs(model_output)
        return {
            token_id.item(): prob.item()
            for token_id, prob in zip(generated_ids[batch_index], token_probs[batch_index])
        }

    @staticmethod
    def get_confidence(model_output, batch_index=0):
        """Aggregate generated token probs into one confidence percent (0-100).

        Uses the geometric mean, exp(mean log p), over every generated step, so the
        score doesn't shrink just because the output is longer. Takes the same model_output as get_token_probs.
        """
        _, token_probs = Utils._generated_token_probs(model_output)
        log_probs = token_probs[batch_index].clamp_min(1e-12).log()
        return log_probs.mean().exp().item() * 100

    @staticmethod
    def get_similarity_percent(a, b):
        """Percent similarity (0-100) between a single pair of embeddings (each (d,) or (1, d)).

        Cosine similarity with negatives clipped to 0. Unlike get_match_percents, this is an absolute score.
        """
        cos = F.cosine_similarity(a.float().reshape(1, -1), b.float().reshape(1, -1)).item()
        return max(cos, 0.0) * 100

    @staticmethod
    def get_match_percents(model, a, b):
        """Turn image-text cosine similarities into percent match across the text candidates.

        a: (n_images, d), b: (n_texts, d), image or text embeddings with out_norm=True.
        Returns (n_images, n_texts); each row sums to 100. Percents are relative to the candidates given.
        Scales by model.logit_scale as in MUSK's demo.ipynb.
        """
        cos = a.float() @ b.float().T
        return (model.logit_scale.float() * cos).softmax(dim=-1) * 100

    # ---------------------------------------------------------------------------
    # Expert-answer heatmaps (CONCH Grad-CAM), precomputed into cases.json.
    # The app should only draw them once the expert answer is revealed.
    # Re-run make_heatmaps after tools/import_tcga.py, which rewrites cases without the heatmap field.
    # ---------------------------------------------------------------------------

    CASES_PATH = Path(__file__).parent / "data" / "cases.json"
    HEATMAP_METHOD = "gradcam/conch_ViT-B-16/last-block/contrast-normal-tissue"
    # TCGA organ names -> how they read in "normal ... tissue"; other organs fall back to organ.lower().
    _ORGAN_TISSUE = {
        "Breast": "breast",
        "Bronchus and lung": "lung",
        "Colon": "colon",
        "Prostate gland": "prostate",
    }

    @staticmethod
    @torch.no_grad()
    def _conch_text_embedding(model, text):
        device = next(model.parameters()).device
        return model.encode_text(Utils.conch_tokenize(get_tokenizer(), [text]).to(device), normalize=True)

    @staticmethod
    def _heatmap_direction(model, text, contrast_text):
        """Text-side direction whose dot product with the image embedding is cos(image, text) - cos(image, contrast_text)."""
        return Utils._conch_text_embedding(model, text) - Utils._conch_text_embedding(model, contrast_text)

    @staticmethod
    def _conch_crop_box(model, image_size):
        """Where CONCH's center crop sits on the displayed image, in its pixels: [left, top, right, bottom]."""
        w, h = image_size
        side = model.visual.image_size[0]
        left, top = int(round((w - side) / 2)), int(round((h - side) / 2))   # same rounding as CenterCrop
        return [left, top, left + side, top + side]

    @staticmethod
    def _encode_image_keep_last_block(model, img):
        """Same as model.encode_image(img, proj_contrast=True, normalize=True), also returning the last ViT block's tokens.

        Mirrors timm's VisionTransformer.forward_features and CONCH's VisionTower.forward (attentional pooling),
        so the tokens are an ordinary value instead of something a forward hook has to catch.
        """
        visual, trunk = model.visual, model.visual.trunk
        x = trunk.norm_pre(trunk.patch_drop(trunk._pos_embed(trunk.patch_embed(img))))
        tokens = trunk.blocks(x)
        pooled = visual.ln_contrast(visual.attn_pool_contrast(trunk.norm(tokens))[:, 0])
        return F.normalize(pooled @ visual.proj_contrast, dim=-1), tokens

    @staticmethod
    def _encode_text_from_embeddings(model, ids, embeddings):
        """Same as model.encode_text(ids, normalize=True), starting from token embeddings.

        ids: (batch, 128) from conch_tokenize. embeddings: (batch, 127, width), the token embeddings of
        ids[:, :-1] (encode_text drops the last position to make room for <cls>).
        Mirrors CONCH's TextTransformer.forward from the embedding step on, so modified embeddings can be passed in.
        """
        text = model.text
        ids = ids[:, :-1]
        cast_dtype = text.transformer.get_cast_dtype()
        seq_len = ids.shape[1] + 1
        x = torch.cat([embeddings.to(cast_dtype), text._repeat(text.cls_emb, embeddings.shape[0])], dim=1)
        attn_mask = text.attn_mask[None, :seq_len, :seq_len] + text.build_cls_mask(ids, cast_dtype)[:, :seq_len, :seq_len]
        x = x + text.positional_embedding[:seq_len].to(cast_dtype)
        x = text.transformer(x.permute(1, 0, 2), attn_mask=attn_mask).permute(1, 0, 2)
        pooled = text.ln_final(x[:, -1]) @ text.text_projection
        return F.normalize(pooled, dim=-1)

    @staticmethod
    def gradcam_heatmap(model, preprocess, image_path, text, contrast_text):
        """Grad-CAM of cos(image, text) - cos(image, contrast_text) on CONCH's last ViT block.

        Highlights regions that make CONCH match the tile to text more than to contrast_text.
        Returns {"grid": side x side list in 0-1, "crop_box": where the grid sits on the displayed image,
        "similarity": cos(image, text), "contrast_similarity": cos(image, contrast_text)}.
        Only the center crop is scored, as in confidence.score_findings.
        """
        device = next(model.parameters()).device
        image = Image.open(image_path).convert("RGB")
        img = preprocess(image).unsqueeze(0).to(device)
        te = Utils._conch_text_embedding(model, text)
        tc = Utils._conch_text_embedding(model, contrast_text)

        with torch.enable_grad():
            ie, block_tokens = Utils._encode_image_keep_last_block(model, img)
            sim, contrast_sim = (ie @ te.T).squeeze(), (ie @ tc.T).squeeze()
            (grads,) = torch.autograd.grad(sim - contrast_sim, block_tokens)   # only the tokens' gradient, no weight grads

        n_prefix = model.visual.trunk.num_prefix_tokens
        tokens = block_tokens[0, n_prefix:].detach()   # (n_patches, channels)
        grads = grads[0, n_prefix:]
        cam = (tokens * grads.mean(0)).sum(-1).relu()
        if cam.max() > 0:
            cam = cam / cam.max()
        side = int(cam.numel() ** 0.5)
        return {
            "grid": cam.reshape(side, side).cpu().round(decimals=3).tolist(),
            "crop_box": Utils._conch_crop_box(model, image.size),
            "similarity": round(sim.item(), 4),
            "contrast_similarity": round(contrast_sim.item(), 4),
        }

    @staticmethod
    def gradcam_text(model, tokenizer, student_text, expert_text, steps=32):
        """Which words in student_text move cos(student, expert) up or down, by CONCH text embedding.

        Integrated gradients on CONCH's token embeddings, summed per word: each word's embedding moves from the
        <pad> embedding (the "no word" baseline) to its real value in `steps` steps, and the gradients along the
        way are averaged. Word scores add up to cos(student, expert) - cos(baseline, expert).
        Returns {"similarity": cos(student, expert), "words": [{"word", "start", "end", "score", "raw"}, ...]}
        in reading order. Positive pulls the student's text toward the expert's, negative pulls it away.
        score is scaled to -1..1 within this text; raw is unscaled (cosine units), so it's comparable across texts.
        Uses only the student's words, so it's safe to show before the reveal.
        """
        device = next(model.parameters()).device
        enc = tokenizer(
            [student_text],
            max_length=127,
            add_special_tokens=True,
            return_token_type_ids=False,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,
        )
        offsets = enc["offset_mapping"][0]
        ids = F.pad(torch.tensor(enc["input_ids"]), (0, 1), value=tokenizer.pad_token_id).to(device)
        te = Utils._conch_text_embedding(model, expert_text)

        with torch.no_grad():
            real = model.text.token_embedding(ids[:, :-1])                  # (1, 127, width)
            pad = model.text.token_embedding.weight[tokenizer.pad_token_id]
        # Only the student's word tokens move; <start>, <end> and padding keep their real embeddings.
        is_word = torch.tensor([end > start for start, end in offsets], device=device).view(1, -1, 1)
        baseline = torch.where(is_word, pad, real)
        alphas = (torch.arange(1, steps + 1, device=device, dtype=torch.float32) / steps).view(-1, 1, 1)
        path = (baseline + alphas * (real - baseline)).requires_grad_(True)  # (steps, 127, width)

        with torch.enable_grad():
            se = Utils._encode_text_from_embeddings(model, ids.expand(steps, -1), path)
            sims = (se @ te.T).squeeze(-1)                                   # the last step is the real text
            (grad,) = torch.autograd.grad(sims.sum(), path)
        sim = sims[-1].item()
        token_scores = ((real - baseline)[0] * grad.mean(0)).sum(-1).cpu().tolist()

        words = [
            {"word": m.group(), "start": m.start(), "end": m.end(), "score": 0.0}
            for m in re.finditer(r"[A-Za-z0-9][A-Za-z0-9'/-]*", student_text)
        ]
        for (start, end), score in zip(offsets, token_scores):
            if end <= start:                                   # special and padding tokens
                continue
            for w in words:
                if start < w["end"] and end > w["start"]:      # overlap: token offsets include the leading space
                    w["score"] += score
                    break
        peak = max((abs(w["score"]) for w in words), default=0.0)
        for w in words:
            w["raw"] = round(w["score"], 5)                    # cosine units; comparable across texts
            w["score"] = round(w["score"] / peak, 3) if peak > 0 else 0.0
        return {"similarity": round(sim, 4), "words": words}

    @staticmethod
    @torch.no_grad()
    def heatmap_masking_check(model, preprocess, image_path, text, contrast_text, grid, frac=0.1, n_random=5, seed=0):
        """Sanity check for a heatmap of cos(image, text) - cos(image, contrast_text):
        blanking its top cells should lower that difference more than blanking random cells.

        Blanked patches are set to 0 after normalization, i.e. the dataset mean color.
        """
        device = next(model.parameters()).device
        img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        direction = Utils._heatmap_direction(model, text, contrast_text)
        side = len(grid)

        flat = torch.tensor(grid).flatten()
        k = max(1, int(frac * flat.numel()))
        gen = torch.Generator().manual_seed(seed)
        random_sims = [
            Utils._score_without_cells(model, img, direction, torch.randperm(flat.numel(), generator=gen)[:k], side)
            for _ in range(n_random)
        ]
        return {
            "original": Utils._score_without_cells(model, img, direction, [], side),
            "top_blanked": Utils._score_without_cells(model, img, direction, flat.topk(k).indices, side),
            "random_blanked": sum(random_sims) / n_random,
        }

    @staticmethod
    @torch.no_grad()
    def _score_without_cells(model, img, direction, cells, side):
        """Image embedding . direction after setting the given cells of a side x side grid to 0 (the mean color)."""
        patch = model.visual.trunk.patch_embed.patch_size[0]
        x = img.clone()
        for i in cells:
            r, c = divmod(int(i), side)
            x[..., r * patch:(r + 1) * patch, c * patch:(c + 1) * patch] = 0
        return (model.encode_image(x, proj_contrast=True, normalize=True) @ direction.T).item()

    @staticmethod
    def _expert_text(case):
        ans = case["expert_answer"]
        return f'{ans["diagnosis"]}. {ans["summary"]}'

    @staticmethod
    def _normal_tissue_text(case):
        tissue = Utils._ORGAN_TISSUE.get(case["organ"], case["organ"].lower())
        return f"normal {tissue} tissue"

    @staticmethod
    def _cases_json_dumps(data):
        # Same layout as import_tcga.py, but keep each row of numbers on one line instead of one number per line.
        text = json.dumps(data, indent=2, ensure_ascii=False)
        return re.sub(r"\[\s*([-0-9.eE,\s]+?)\s*\]", Utils._one_line_numbers, text)

    @staticmethod
    def _one_line_numbers(match):
        """re.sub callback for _cases_json_dumps: "[\\n 1,\\n 2\\n]" -> "[1, 2]"."""
        return "[" + re.sub(r",\s+", ", ", match.group(1)) + "]"

    @staticmethod
    def make_heatmaps(model, preprocess, cases_path=CASES_PATH, force=False):
        """Precompute a heatmap of the expert answer vs normal tissue for each case's image and store it in cases.json.

        The map shows what in the tile makes CONCH match the expert answer more than "normal <organ> tissue".
        Skips cases whose image, texts, and method are unchanged unless force=True.
        Takes an already-loaded CONCH (from Utils.load_conch) so it isn't loaded twice. From the app folder:
            python -c "from utils import Utils; Utils.make_heatmaps(*Utils.load_conch())"
        """
        cases_path = Path(cases_path)
        data_dir = cases_path.parent
        data = json.loads(cases_path.read_text())

        for case in data["cases"]:
            image_path = data_dir / case["image"]
            text, contrast = Utils._expert_text(case), Utils._normal_tissue_text(case)
            sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
            hm = case.get("heatmap") or {}
            stored = (hm.get("method"), hm.get("text"), hm.get("contrast_text"), hm.get("image_sha256"))
            if not force and stored == (Utils.HEATMAP_METHOD, text, contrast, sha):
                print(f"{case['id']}: up to date")
                continue

            hm = Utils.gradcam_heatmap(model, preprocess, image_path, text, contrast)
            check = Utils.heatmap_masking_check(model, preprocess, image_path, text, contrast, hm["grid"])
            case["heatmap"] = {
                **hm, "text": text, "contrast_text": contrast, "method": Utils.HEATMAP_METHOD, "image_sha256": sha,
            }
            print(
                f"{case['id']}: expert {hm['similarity']:.3f} - normal {hm['contrast_similarity']:.3f}"
                f" = {check['original']:.3f} | top 10% blanked {check['top_blanked']:.3f}"
                f" vs random 10% blanked {check['random_blanked']:.3f}"
            )

        cases_path.write_text(Utils._cases_json_dumps(data))
        print("wrote", cases_path)

    # Jet-style color map anchors: blue (low) -> cyan -> green -> yellow -> red (high).
    # Jet-style color map anchors: blue (low) -> cyan -> green -> yellow -> red (high).
    _HEATMAP_STOPS = [0.0, 0.25, 0.5, 0.75, 1.0]
    _HEATMAP_COLORS = np.array([[0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]], dtype=np.float32)

    @staticmethod
    def render_heatmap(case, data_dir=CASES_PATH.parent, alpha=0.5):
        """Draw a case's stored heatmap (from cases.json) over its image. No model needed.

        Inside crop_box: blue (low) to red (high) at constant opacity alpha. Outside crop_box, which
        CONCH never saw, the image is greyed out, and the box is outlined. Returns a PIL image.
        """
        image = Image.open(Path(data_dir) / case["image"]).convert("RGB")
        hm = case["heatmap"]
        left, top, right, bottom = hm["crop_box"]

        grid = np.asarray(hm["grid"], dtype=np.float32)
        values = np.asarray(Image.fromarray(grid, mode="F").resize((right - left, bottom - top), Image.BILINEAR)).clip(0, 1)
        colors = np.stack(
            [np.interp(values, Utils._HEATMAP_STOPS, Utils._HEATMAP_COLORS[:, ch]) for ch in range(3)], axis=-1
        )
        tile = np.asarray(image.crop((left, top, right, bottom)), dtype=np.float32)
        blended = ((1 - alpha) * tile + alpha * colors).astype(np.uint8)

        out = Image.blend(image, Image.new("RGB", image.size, (128, 128, 128)), 0.6)
        out.paste(Image.fromarray(blended), (left, top))
        ImageDraw.Draw(out).rectangle([left, top, right - 1, bottom - 1], outline=(255, 255, 255), width=3)
        return out


