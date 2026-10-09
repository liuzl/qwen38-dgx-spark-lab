"""Deterministic synthetic image fixtures, generated locally without external input."""

import hashlib
import json
from pathlib import Path


def create(folder):
    from PIL import Image, ImageDraw, ImageFont

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=28)
    image = Image.new("RGB", (640, 360), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 20), "CODE: K7M42", font=font, fill="black")
    for x in (40, 150, 260):
        draw.rectangle((x, 110, x + 65, 175), fill="blue")
    for x in (70, 230):
        draw.ellipse((x, 240, x + 65, 305), fill="red")
    image.save(folder / "shapes.png")
    chart = Image.new("RGB", (640, 400), "white")
    draw = ImageDraw.Draw(chart)
    draw.text((20, 15), "Quarterly sales (units)", font=font, fill="black")
    for x, label, value in ((80, "A", 12), (260, "B", 19), (440, "C", 7)):
        top = 340 - value * 12
        draw.rectangle((x, top, x + 90, 340), fill="#3974bd")
        draw.text((x + 20, top - 35), str(value), font=font, fill="black")
        draw.text((x + 25, 350), label, font=font, fill="black")
    chart.save(folder / "chart.png")
    rows = [
        {
            "case": "vision-shapes",
            "file": "shapes.png",
            "prompt": (
                "Read this image. Return only JSON with code (after CODE:), b"
                "lue_squares and red_circles."
            ),
            "expected": {"code": "K7M42", "blue_squares": 3, "red_circles": 2},
        },
        {
            "case": "vision-chart",
            "file": "chart.png",
            "prompt": (
                "Read this chart. Return only JSON with largest (category nam"
                "e), gap_b_a (B minus A), total (all three)."
            ),
            "expected": {"largest": "B", "gap_b_a": 7, "total": 38},
        },
    ]
    for row in rows:
        row["sha256"] = hashlib.sha256((folder / row["file"]).read_bytes()).hexdigest()
    (folder / "manifest.json").write_text(json.dumps(rows, indent=2))
    return rows
