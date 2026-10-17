"""Writes placeholder slide images so the UI runs before real images are chosen."""
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

DATA = Path(__file__).parent / "data"

for case in json.loads((DATA / "cases.json").read_text())["cases"]:
    random.seed(case["id"])
    img = Image.new("RGB", (768, 512), (243, 214, 228))
    d = ImageDraw.Draw(img)
    for _ in range(260):  # fake H&E-ish blobs
        x, y, r = random.randint(0, 768), random.randint(0, 512), random.randint(4, 22)
        col = random.choice([(120, 60, 150), (90, 40, 130), (220, 140, 180), (200, 110, 160)])
        d.ellipse([x - r, y - r, x + r, y + r], fill=col)
    d.rectangle([0, 450, 768, 512], fill=(255, 255, 255))
    d.text((16, 470), f"PLACEHOLDER for {case['title']}: replace with a licensed H&E image", fill=(0, 0, 0))
    out = DATA / case["image"]
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print("wrote", out)
