from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path.cwd()
OUT = ROOT / "near_miss_samples" / "outputs" / "page05_mimo_results_table_replacement.png"

METHODS = ["RoboCLIP", "GVL-style VLM", "TOPReward"]
COLORS = {
    "RoboCLIP": "#2F67E8",
    "GVL-style VLM": "#0F8A83",
    "TOPReward": "#C74D63",
}

SUCCESS = {
    "RoboCLIP": {"spearman": 1.000, "mae": 0.110},
    "GVL-style VLM": {"spearman": 0.998, "mae": 0.175},
    "TOPReward": {"spearman": 0.312, "mae": 0.400},
}
FAILURE = {
    "RoboCLIP": {"rate": 0.778, "fp": "7/9", "sc": "2/3", "sp": "3/3", "peg": "2/3"},
    "GVL-style VLM": {"rate": 0.889, "fp": "8/9", "sc": "3/3", "sp": "2/3", "peg": "3/3"},
    "TOPReward": {"rate": 0.556, "fp": "5/9", "sc": "1/3", "sp": "1/3", "peg": "3/3"},
}


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    font_dir = Path(r"C:\Windows\Fonts")
    candidates = ["msyhbd.ttc", "msyh.ttc", "simhei.ttf"] if bold else ["msyh.ttc", "simhei.ttf"]
    for name in candidates:
        path = font_dir / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


W, H = 1600, 886
img = Image.new("RGB", (W, H), "#FFFFFF")
d = ImageDraw.Draw(img)

F = {
    "panel": get_font(44, True),
    "sub": get_font(27),
    "label": get_font(33, True),
    "small": get_font(25),
    "value": get_font(34, True),
    "heat": get_font(35, True),
    "tiny": get_font(23),
}
C = {
    "ink": "#1F2A37",
    "navy": "#0E376D",
    "muted": "#566579",
    "grid": "#D7E2EF",
    "track": "#E8EEF6",
    "head": "#F0F5FB",
}


def rect(x0, y0, x1, y1, fill=None, outline=None, width=1):
    d.rectangle([x0, y0, x1, y1], fill=fill, outline=outline, width=width)


def text(x, y, s, font, fill=C["ink"], anchor=None):
    d.text((x, y), s, font=font, fill=fill, anchor=anchor)


def fp_num(v: str) -> int:
    return int(v.split("/")[0])


# Outer frame mirrors the table style from the PPT.
rect(0, 0, W - 1, H - 1, fill="#FFFFFF", outline=C["grid"], width=3)

# Top two metric panels.
pad = 44
gap = 44
panel_w = (W - 2 * pad - gap) // 2
panel_h = 370
left = (pad, 38, pad + panel_w, 38 + panel_h)
right = (pad + panel_w + gap, 38, W - pad, 38 + panel_h)

for box in [left, right]:
    rect(*box, fill="#FFFFFF", outline=C["grid"], width=2)

text(left[0] + 32, left[1] + 28, "成功轨迹排序能力", F["panel"], C["navy"])
text(left[0] + 32, left[1] + 83, "Spearman vs time ↑；灰字为 MAE ↓", F["sub"], C["muted"])
text(right[0] + 32, right[1] + 28, "失败样本误判风险", F["panel"], "#8B2638")
text(right[0] + 32, right[1] + 83, "最终帧 false positive rate ↓；灰字为 FP 数", F["sub"], C["muted"])


def draw_metric_rows(box, values, suffix_func):
    x0, y0, x1, y1 = box
    label_x = x0 + 32
    bar_x = x0 + 315
    bar_w = x1 - bar_x - 145
    y = y0 + 142
    for method in METHODS:
        color = COLORS[method]
        text(label_x, y - 8, method, F["label"], color)
        rect(bar_x, y, bar_x + bar_w, y + 36, fill=C["track"])
        rect(bar_x, y, bar_x + int(bar_w * values[method]), y + 36, fill=color)
        text(bar_x + bar_w + 22, y - 5, f"{values[method]:.3f}", F["value"], C["ink"])
        text(bar_x + bar_w + 22, y + 34, suffix_func(method), F["small"], C["muted"])
        y += 82


draw_metric_rows(
    left,
    {m: SUCCESS[m]["spearman"] for m in METHODS},
    lambda m: f"MAE {SUCCESS[m]['mae']:.3f}",
)
draw_metric_rows(
    right,
    {m: FAILURE[m]["rate"] for m in METHODS},
    lambda m: f"FP {FAILURE[m]['fp']}",
)

# Heatmap.
heat_y = 454
text(pad, heat_y, "误判分布：每个任务 3 个失败 / near-miss 样本", F["panel"], C["navy"])
text(pad, heat_y + 55, "单元格为最终帧 FP 数 / 3；颜色越深，越容易把失败当作高进度", F["sub"], C["muted"])

table_x = pad
table_y = heat_y + 105
row_label_w = 345
col_w = (W - 2 * pad - row_label_w) // 3
cell_h = 72

rect(table_x, table_y, table_x + row_label_w, table_y + cell_h, fill=C["head"], outline=C["grid"], width=2)
text(table_x + 28, table_y + 18, "方法", F["heat"], C["navy"])
headers = [("StackCube", "sc"), ("StackPyramid", "sp"), ("PegInsertionSide", "peg")]
for j, (header, _) in enumerate(headers):
    cx = table_x + row_label_w + j * col_w
    rect(cx, table_y, cx + col_w, table_y + cell_h, fill=C["head"], outline=C["grid"], width=2)
    text(cx + col_w / 2, table_y + 18, header, F["heat"], C["navy"], anchor="ma")

red_steps = {0: "#FFFFFF", 1: "#FBE6EA", 2: "#F4B8C4", 3: "#D95B73"}
for i, method in enumerate(METHODS):
    ry = table_y + cell_h * (i + 1)
    rect(table_x, ry, table_x + row_label_w, ry + cell_h, fill="#FFFFFF", outline=C["grid"], width=2)
    rect(table_x + 16, ry + 13, table_x + 24, ry + cell_h - 13, fill=COLORS[method])
    text(table_x + 42, ry + 20, method, F["heat"], COLORS[method])
    for j, (_, key) in enumerate(headers):
        cx = table_x + row_label_w + j * col_w
        value = FAILURE[method][key]
        fp = fp_num(value)
        rect(cx, ry, cx + col_w, ry + cell_h, fill=red_steps[fp], outline=C["grid"], width=2)
        fill = "#FFFFFF" if fp == 3 else C["ink"]
        text(cx + col_w / 2, ry + 20, value, F["heat"], fill, anchor="ma")

text(
    pad,
    H - 42,
    "读法：左侧越长越好，右侧越短越好；三种方法在成功轨迹上可排序，但在失败样本上均有高估。",
    F["tiny"],
    C["muted"],
)

OUT.parent.mkdir(parents=True, exist_ok=True)
img.save(OUT, quality=95)
print(OUT)
