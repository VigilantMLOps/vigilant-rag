"""Generate AppIcon.icns for the Atlas RAG macOS app.

Requires: pip install Pillow
"""
import shutil
import subprocess
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    raise SystemExit("Run: pip install Pillow")

OUT = Path("AppIcon.iconset")
OUT.mkdir(exist_ok=True)


def make_png(size: int) -> Image.Image:
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Gradient background (indigo → blue)
    for y in range(size):
        t = y / size
        r = int(45  + t * 15)
        g = int(55  + t * 30)
        b = int(200 + t * 35)
        draw.rectangle([0, y, size, y + 1], fill=(r, g, b, 255))

    # Mask to rounded rectangle
    radius = int(size * 0.22)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, size - 1, size - 1], radius=radius, fill=255
    )
    img.putalpha(mask)

    # Draw two overlapping rounded rects as a simple "chat bubble" mark
    draw = ImageDraw.Draw(img)
    m  = int(size * 0.18)
    rr = int(size * 0.12)
    w  = int(size * 0.07)

    # outer bubble
    draw.rounded_rectangle(
        [m, m, size - m, size - m * 1.4],
        radius=rr, outline=(255, 255, 255, 210), width=w,
    )
    # inner dot row
    dot = int(size * 0.07)
    cy  = size // 2 - int(size * 0.03)
    for cx in [int(size * 0.35), int(size * 0.5), int(size * 0.65)]:
        draw.ellipse([cx - dot, cy - dot, cx + dot, cy + dot],
                     fill=(255, 255, 255, 200))

    return img


SIZES = [16, 32, 64, 128, 256, 512, 1024]
for s in SIZES:
    make_png(s).save(OUT / f"icon_{s}x{s}.png")
    if s <= 512:
        make_png(s * 2).save(OUT / f"icon_{s}x{s}@2x.png")

subprocess.run(["iconutil", "-c", "icns", str(OUT), "-o", "AppIcon.icns"], check=True)
shutil.rmtree(OUT)
print("✓ AppIcon.icns created")
