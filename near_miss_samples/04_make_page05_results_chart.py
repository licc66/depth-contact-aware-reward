from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path.cwd()
OUT = ROOT / "near_miss_samples" / "outputs" / "page05_mimo_results_right_panel.png"

METHODS = ["RoboCLIP-style", "GVL-style VLM", "TOPReward-style"]
METHOD_SHORT = {
    "RoboCLIP-style": "RoboCLIP",
    "GVL-style VLM": "GVL-style VLM",
    "TOPReward-style": "TOPReward",
}
COLORS = {
    "RoboCLIP-style": "#2F67E8",
    "GVL-style VLM": "#0F8A83",
    "TOPReward-style": "#C74D63",
}

# Values verified from the current experiment CSVs:
# - MiMo统一复现对比结果.csv
# - near_miss_samples/outputs/mimo_failure_eval/mimo_failure_eval_method_summary.csv
SUCCESS_SPEARMAN = {
    "RoboCLIP-style": 1.000,
    "GVL-style VLM": (0.999 + 0.996 + 0.999) / 3,
    "TOPReward-style": (0.000 + 0.000 + 0.935) / 3,
}
SUCCESS_MAE = {
    "RoboCLIP-style": (0.091 + 0.091 + 0.148) / 3,
    "GVL-style VLM": (0.200 + 0.174 + 0.150) / 3,
    "TOPReward-style": (0.500 + 0.500 + 0.201) / 3,
}
FAILURE = {
    "RoboCLIP-style": {
        "rate": 0.778,
        "fp": "7/9",
        "stackcube": "2/3",
        "stackpyramid": "3/3",
        "peginsertion": "2/3",
    },
    "GVL-style VLM": {
        "rate": 0.889,
        "fp": "8/9",
        "stackcube": "3/3",
        "stackpyramid": "2/3",
        "peginsertion": "3/3",
    },
    "TOPReward-style": {
        "rate": 0.556,
        "fp": "5/9",
        "stackcube": "1/3",
        "stackpyramid": "1/3",
        "peginsertion": "3/3",
    },
}


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    font_dir = Path(r"C:\Windows\Fonts")
    candidates = ["msyhbd.ttc", "msyh.ttc", "simhei.ttf"] if bold else ["msyh.ttc", "simhei.ttf"]
    for name in candidates:
        path = font_dir / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


W, H = 1600, 900
img = Image.new("RGB", (W, H), "#FFFFFF")
d = ImageDraw.Draw(img)

F = {
    "tag": get_font(26, True),
    "title": get_font(36, True),
    "sub": get_font(21),
    "panel": get_font(27, True),
    "label": get_font(24, True),
    "small": get_font(18),
    "value": get_font(24, True),
    "heat": get_font(23, True),
    "note": get_font(23, True),
}
C = {
    "ink": "#1F2A37",
    "navy": "#0E376D",
    "muted": "#566579",
    "grid": "#D7E2EF",
    "track": "#E8EEF6",
    "yellow_bg": "#FFF6E8",
    "yellow_bd": "#E9B75E",
}


def rect(x0, y0, x1, y1, fill=None, outline=None, width=1):
    d.rectangle([x0, y0, x1, y1], fill=fill, outline=outline, width=width)


def text(x, y, s, f, fill=C["ink"], anchor=None):
    d.text((x, y), s, font=f, fill=fill, anchor=anchor)


def line(x0, y0, x1, y1, fill=C["grid"], width=2):
    d.line([x0, y0, x1, y1], fill=fill, width=width)


def fp_num(value: str) -> int:
    return int(value.split("/")[0])


# Header
rect(54, 38, 68, 52, fill="#0F8A83")
text(82, 31, "RESULTS | 成功轨迹 vs 失败样本", F["tag"], "#0F5F58")
text(54, 72, "量化结果：成功可排序，失败易高估", F["title"], C["ink"])
text(
    54,
    119,
    "MiMo-v2.5；3 个官方成功轨迹 + 9 个失败/near-miss 视频；score_end ≥ 0.8 记为最终帧误判",
    F["sub"],
    C["muted"],
)
line(54, 156, 1546, 156)

# Top panels
left = (54, 184, 760, 442)
right = (830, 184, 1546, 442)
heat = (54, 484, 1546, 844)
for box in [left, right, heat]:
    rect(*box, fill="#FFFFFF", outline=C["grid"], width=2)


def draw_bar_panel(box, title, subtitle, values, secondary, secondary_prefix, lower_is_better=False):
    x0, y0, x1, y1 = box
    title_color = "#8B2638" if lower_is_better else C["navy"]
    text(x0 + 28, y0 + 22, title, F["panel"], title_color)
    text(x0 + 28, y0 + 59, subtitle, F["small"], C["muted"])

    bar_x, bar_w, bar_h = x0 + 232, 408, 28
    start_y = y0 + 100
    for i, method in enumerate(METHODS):
        y = start_y + i * 62
        col = COLORS[method]
        text(x0 + 28, y - 4, METHOD_SHORT[method], F["label"], col)
        rect(bar_x, y, bar_x + bar_w, y + bar_h, fill=C["track"])
        val = values[method]
        rect(bar_x, y, bar_x + int(bar_w * val), y + bar_h, fill=col)
        text(bar_x + bar_w + 16, y - 2, f"{val:.3f}", F["value"], C["ink"])
        text(bar_x + bar_w + 16, y + 28, f"{secondary_prefix} {secondary[method]}", F["small"], C["muted"])

    # Numeric labels at the bar end carry the scale; omitting axis ticks keeps the
    # slide replacement readable after it is reduced into the right-side panel.


draw_bar_panel(
    left,
    "成功轨迹排序能力",
    "Spearman vs time ↑；括号为 progress MAE ↓",
    SUCCESS_SPEARMAN,
    {m: f"{SUCCESS_MAE[m]:.3f}" for m in METHODS},
    "MAE",
)
draw_bar_panel(
    right,
    "失败样本误判风险",
    "最终帧 false positive rate ↓；括号为 FP 数",
    {m: FAILURE[m]["rate"] for m in METHODS},
    {m: FAILURE[m]["fp"] for m in METHODS},
    "FP",
    lower_is_better=True,
)

# Heatmap
x0, y0, x1, y1 = heat
text(x0 + 28, y0 + 20, "误判分布：每个任务 3 个失败/near-miss 样本", F["panel"], C["navy"])
text(x0 + 28, y0 + 57, "单元格为最终帧 FP 数 / 3；颜色越深，越容易把失败当作高进度", F["small"], C["muted"])

row_label_w, col_w, cell_h = 310, 360, 58
table_x, table_y = x0 + 28, y0 + 96
headers = [
    ("StackCube", "stackcube"),
    ("StackPyramid", "stackpyramid"),
    ("PegInsertionSide", "peginsertion"),
]
rect(table_x, table_y, table_x + row_label_w, table_y + cell_h, fill="#F0F5FB", outline=C["grid"])
text(table_x + 18, table_y + 14, "方法", F["heat"], C["navy"])
for j, (header, _) in enumerate(headers):
    cx = table_x + row_label_w + j * col_w
    rect(cx, table_y, cx + col_w, table_y + cell_h, fill="#F0F5FB", outline=C["grid"])
    text(cx + col_w / 2, table_y + 14, header, F["heat"], C["navy"], anchor="ma")

red_steps = {0: "#FFFFFF", 1: "#FBE6EA", 2: "#F4B8C4", 3: "#D95B73"}
for i, method in enumerate(METHODS):
    ry = table_y + cell_h * (i + 1)
    rect(table_x, ry, table_x + row_label_w, ry + cell_h, fill="#FFFFFF", outline=C["grid"])
    rect(table_x + 8, ry + 8, table_x + 14, ry + cell_h - 8, fill=COLORS[method])
    text(table_x + 26, ry + 13, method, F["heat"], COLORS[method])
    for j, (_, key) in enumerate(headers):
        cx = table_x + row_label_w + j * col_w
        value = FAILURE[method][key]
        fp = fp_num(value)
        rect(cx, ry, cx + col_w, ry + cell_h, fill=red_steps[fp], outline=C["grid"])
        fill = "#FFFFFF" if fp == 3 else C["ink"]
        text(cx + col_w / 2, ry + 13, value, F["heat"], fill, anchor="ma")

OUT.parent.mkdir(parents=True, exist_ok=True)
img.save(OUT, quality=95)
print(OUT)
