#!/usr/bin/env python3
"""Render a receipt-backed terminal-style PNG; never manufacture live readings."""
import argparse
import hashlib
import json
from pathlib import Path

if __package__:
    from .strata_monitor_live import render
    from .render_throughput_comparison import positive
else:
    from strata_monitor_live import render
    from render_throughput_comparison import positive

MONO_FONTS = ("/System/Library/Fonts/Menlo.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
              "C:/Windows/Fonts/consola.ttf")


def draw(snapshot_path, output, font_path=None, peak=None, source=None, basis=None):
    from PIL import Image, ImageDraw, ImageFont
    raw = Path(snapshot_path).read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict) or not data.get("captured_at") or not data.get("source_endpoint"):
        raise ValueError("use an actual monitor --once --snapshot receipt with capture time/endpoint")
    if peak is not None and (not source or not basis):
        raise ValueError("peak needs --peak-source and --peak-basis; it is separate from the live snapshot")
    font_path = font_path or next((p for p in MONO_FONTS if Path(p).is_file()), None)
    if not font_path:
        raise ValueError("provide --font pointing at a monospace TrueType font")
    font = ImageFont.truetype(str(font_path), 27)
    small = ImageFont.truetype(str(font_path), 21)
    url = str(data["source_endpoint"]).removesuffix("/metrics")
    lines = render(data, url).splitlines()
    width, line_height, top = 2400, 39, 65
    height = top + line_height * len(lines) + (230 if peak is not None else 145)
    image = Image.new("RGB", (width, height), "#0d1422")
    painter = ImageDraw.Draw(image)
    for i, line in enumerate(lines):
        color = "#c9d4e6"
        if i == 0:
            color = "#6bdded"
        elif line.startswith(("SERVING", "BATCHING / PARALLEL", "HARDWARE", "REQUESTS", "ACTIVITY")):
            color = "#83aaff"
        elif line.startswith("─"):
            color = "#31425c"
        elif line.startswith(("READ-ONLY", "Read =", "C2 PP", "GPU busy/power")):
            color = "#95a4b9"
        painter.text((65, top+i*line_height), line, font=font, fill=color)
    bottom = top + line_height * len(lines) + 20
    if peak is not None:
        painter.text((65, bottom), f"SEPARATE BENCHMARK PEAK: {peak:.2f} tok/s | {basis}", font=font, fill="#ffcf74")
        painter.text((65, bottom+43), f"Source: {source}", font=small, fill="#95a4b9")
        bottom += 86
    digest = hashlib.sha256(raw).hexdigest()
    painter.text((65, bottom), "RENDERED MEASURED SNAPSHOT / fork viewer, NOT an original terminal capture", font=small, fill="#95a4b9")
    painter.text((65, bottom+34), f"Input SHA256: {digest}", font=small, fill="#95a4b9")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, dpi=(200, 200))
    output.with_suffix(".json").write_text(json.dumps({"snapshot_sha256": digest, "captured_at": data["captured_at"],
        "source_endpoint": data["source_endpoint"], "peak_tok_s": peak, "peak_source": source,
        "peak_basis": basis, "rendered_not_terminal_capture": True}, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return width, height


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/media/5090-tui-measured.png"))
    parser.add_argument("--font", type=Path)
    parser.add_argument("--peak-tok-s", type=positive)
    parser.add_argument("--peak-source")
    parser.add_argument("--peak-basis")
    args = parser.parse_args(argv)
    try:
        size = draw(args.snapshot, args.output, args.font, args.peak_tok_s, args.peak_source, args.peak_basis)
    except (ValueError, OSError, ImportError) as error:
        parser.error(str(error))
    print(f"{args.output} ({size[0]}×{size[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
