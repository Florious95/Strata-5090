#!/usr/bin/env python3
"""PNG/SVG evidence comparison; unknown measurements stay pending, never zero."""
import argparse
from html import escape
import json
import math
from pathlib import Path


FONTS = ("/System/Library/Fonts/STHeiti Medium.ttc",
         "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
         "C:/Windows/Fonts/msyh.ttc")


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("throughput must be finite and positive")
    return value


def rows(args):
    for name in ("short", "community"):
        if getattr(args, name + "_tok_s") is not None:
            if not getattr(args, name + "_source") or not getattr(args, name + "_basis"):
                raise ValueError(f"--{name}-source and --{name}-basis are required with a measured/claimed value")
    if not 250 <= args.short_prompt_tokens <= 400:
        raise ValueError("this toy comparison requires an actual prompt length in 250..400 tokens")
    return [
        {"label": "258K 极限长上下文", "detail": "1×5090 · C2双路 · 三轮HTTP池化（历史验收）",
         "value": 209.9903, "color": "#2463cb", "basis": "HTTP pooled: 24,576 tokens / 117.033979 s",
         "source": "benchmarks/results/PR-EVIDENCE-LEDGER-20261012.json → c15c25f-fusion43-queued4-summary.json"},
        {"label": "75K 实际工程写代码", "detail": "1×5090 · 单路active · 最新实际生产请求",
         "value": 173.4, "color": "#15836e", "basis": "75,680 prompt / 1,005 output; native decode 5.7953 s",
         "source": "docs/evidence/production-coding-20261012.json;55-round pooled native=172.977 across9,989→75,680 contexts"},
        {"label": f"{args.short_prompt_tokens} Token 玩具短prompt", "detail": "1×5090 · 选取最高值，不代表稳定业务吞吐",
         "value": args.short_tok_s, "color": "#d18a18", "basis": args.short_basis or "待bench实测，非零吞吐",
         "source": args.short_source or "pending"},
        {"label": "社区 4 卡宣传数字", "detail": "4×GPU · 外部宣称；硬件/上下文/口径按来源",
         "value": args.community_tok_s, "color": "#9158b5", "basis": args.community_basis or "待原始数字与来源，尚不可比",
         "source": args.community_source or "pending"},
    ]


def draw(items, output, font_path=None, svg_only=False):
    width, height = 2400, 1620
    image = painter = None
    if not svg_only:
        from PIL import Image, ImageDraw, ImageFont
        font_path = font_path or next((p for p in FONTS if Path(p).is_file()), None)
        if not font_path:
            raise ValueError("PNG requires a Chinese font: pass --font /path/to/font or use --svg-only")
        image = Image.new("RGB", (width, height), "#f6f8fc")
        painter = ImageDraw.Draw(image)
    fonts = {}
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="#f6f8fc"/>',
           '<g font-family="Heiti SC,Noto Sans CJK SC,Microsoft YaHei,sans-serif">']

    def rect(x, y, w, h, fill, stroke="none"):
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}"/>')
        if painter:
            painter.rectangle((x, y, x+w, y+h), fill=fill, outline=None if stroke == "none" else stroke)

    def text(x, y, value, size=30, fill="#263147"):
        value = str(value)
        svg.append(f'<text x="{x}" y="{y}" dominant-baseline="text-before-edge" font-size="{size}" fill="{fill}">{escape(value)}</text>')
        if painter:
            if size not in fonts:
                fonts[size] = ImageFont.truetype(str(font_path), size)
            painter.text((x, y), value, font=fonts[size], fill=fill)

    text(95, 55, "真实业务吞吐，与跑分数字不是同一个问题", 64)
    text(98, 142, "真实长上下文 / 真实工程  vs  玩具短文本最高值  vs  社区四卡宣称", 35, "#55637a")
    rect(95, 215, 2210, 85, "#fff0dd")
    text(125, 237, "口径、上下文、并发与卡数不同：展示数字，不直接排名，也不计算虚假提升。", 34, "#8c5210")
    left, plot_width = 670, 1320
    maximum = max(row["value"] for row in items if row["value"] is not None)
    # Linear, zero-origin scale shared by every known bar. No broken/log axis.
    scale_max = math.ceil(maximum / 50) * 50
    for i in range(6):
        value = i * scale_max / 5
        x = left + plot_width * i / 5
        rect(x, 367, 1, 902, "#dde3ed")
        text(x-22, 324, f"{value:g}", 25, "#778398")
    text(left+plot_width+40, 324, "tok/s", 27, "#778398")
    for i, row in enumerate(items):
        y = 395 + i * 221
        text(105, y, row["label"], 41, row["color"])
        text(105, y+67, row["detail"], 23, "#55637a")
        if row["value"] is None:
            text(left+35, y+13, "待实测" if i == 2 else "待来源", 46, "#8793a6")
        else:
            bar_width = plot_width * row["value"] / scale_max
            rect(left, y+9, bar_width, 64, row["color"])
            text(left+bar_width+23, y+8, f'{row["value"]:.2f}', 47, row["color"])
        text(left, y+96, row["basis"], 25, "#55637a")
    text(100, 1304, "210 ≠ 双路各210；173 ≠ 客户端端到端速度；短prompt ≠ context容量只有250。", 31)
    text(100, 1360, "最高值不是中位数/池化值；短请求decode增长后KV也增长；社区4卡数据不是本机复测。", 28)
    text(100, 1435, "75K最新173.4；9.99K→75.68K工程55轮池化172.98。完整来源见同名JSON。", 28, "#55637a")
    text(100, 1490, "Strata-5090 / evidence comparison / lossless golden baseline; no runtime or GPU mutation", 24, "#778398")
    svg.append("</g></svg>")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".svg").write_text("\n".join(svg), encoding="utf-8")
    output.with_suffix(".json").write_text(json.dumps({"rows": items, "axis": "linear-zero-origin", "directly_comparable": False}, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    if image:
        image.save(output.with_suffix(".png"), dpi=(200, 200))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short-tok-s", type=positive)
    parser.add_argument("--short-prompt-tokens", type=int, default=250)
    parser.add_argument("--short-source", help="raw receipt path/URL; no private prompts or credentials")
    parser.add_argument("--short-basis", help="timer/active slots/statistic/output length; e.g. peak native decode, C1")
    parser.add_argument("--community-tok-s", type=positive)
    parser.add_argument("--community-source", help="original public claim URL, not our own measurement")
    parser.add_argument("--community-basis", help="GPU models/count/context/timer/quality, unknowns explicitly labeled")
    parser.add_argument("--output", type=Path, default=Path("docs/media/throughput-comparison-pending"))
    parser.add_argument("--font", type=Path)
    parser.add_argument("--svg-only", action="store_true", help="standard-library only, no Pillow/font needed")
    args = parser.parse_args(argv)
    try:
        draw(rows(args), args.output, args.font, args.svg_only)
    except (ValueError, OSError, ImportError) as error:
        parser.error(str(error))
    print(args.output.with_suffix(".svg"))
    if not args.svg_only:
        print(args.output.with_suffix(".png"))
    print(args.output.with_suffix(".json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
