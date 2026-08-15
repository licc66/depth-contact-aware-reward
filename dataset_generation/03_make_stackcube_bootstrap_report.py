from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


DATASET = Path(r"E:\reward_model_dataset\raw_rollouts\stackcube_bootstrap_v1")
INDICES = Path(r"E:\reward_model_dataset\pair_indices\stackcube_bootstrap_v1")
REPORTS = Path(r"E:\reward_model_dataset\reports\stackcube_bootstrap_v1")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_frame(path: Path, frame_idx: int | None = None) -> Image.Image:
    cap = cv2.VideoCapture(str(path))
    if frame_idx is None:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        frame_idx = max(0, n - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read {path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def get_font(size: int, bold: bool = False):
    font_dir = Path(r"C:\Windows\Fonts")
    names = ["msyhbd.ttc", "msyh.ttc", "simhei.ttf"] if bold else ["msyh.ttc", "simhei.ttf"]
    for name in names:
        path = font_dir / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def make_preview(rows: list[dict[str, str]]) -> Path:
    sample_ids = [
        "SC-SUCC-0000",
        "SC-SUCC-0000-TRUNC-20",
        "SC-SUCC-0000-TRUNC-60",
        "SC-SUCC-0000-TRUNC-92",
        "SC-SUCC-0000-OFFSET-posx-035mm",
        "SC-SUCC-0000-OFFSET-posx-055mm",
        "SC-SUCC-0000-OFFSET-posx-075mm",
        "SC-SUCC-0006",
    ]
    by_id = {row["sample_id"]: row for row in rows}
    thumbs = []
    for sample_id in sample_ids:
        row = by_id.get(sample_id)
        if not row:
            continue
        image = read_frame(Path(row["video_path_windows"]))
        image.thumbnail((260, 160), Image.Resampling.LANCZOS)
        thumbs.append((sample_id, row.get("source_type", ""), row.get("observed_success", ""), image.copy()))

    width = 1120
    cell_w = 280
    cell_h = 210
    rows_n = (len(thumbs) + 3) // 4
    canvas = Image.new("RGB", (width, rows_n * cell_h + 60), "#FFFFFF")
    draw = ImageDraw.Draw(canvas)
    title_font = get_font(24, True)
    small_font = get_font(15)
    draw.text((20, 18), "StackCube bootstrap v1 preview: success / truncation / offset near-miss", font=title_font, fill="#1F2A37")
    for i, (sample_id, source_type, success, image) in enumerate(thumbs):
        x = (i % 4) * cell_w + 20
        y = (i // 4) * cell_h + 60
        draw.rectangle([x, y, x + cell_w - 16, y + cell_h - 10], outline="#D7E2EF", width=2)
        canvas.paste(image, (x + 10, y + 10))
        draw.text((x + 10, y + 172), sample_id, font=small_font, fill="#0E376D")
        draw.text((x + 10, y + 190), f"{source_type} | success={success}", font=small_font, fill="#566579")
    out = REPORTS / "stackcube_bootstrap_v1_preview.png"
    REPORTS.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    traj = load_csv(DATASET / "trajectory_manifest.csv")
    clips = load_csv(INDICES / "clip_manifest.csv")
    pairs = load_csv(INDICES / "pair_manifest.csv")

    preview_path = make_preview(traj)
    source_counts = Counter(row["source_type"] for row in traj)
    split_counts = Counter(row["split"] for row in traj)
    pair_counts = Counter(row["pair_type"] for row in pairs)
    split_clip_counts = Counter(row["split"] for row in clips)
    split_pair_counts = Counter(row["split"] for row in pairs)

    report = {
        "dataset": str(DATASET),
        "indices": str(INDICES),
        "num_trajectories": len(traj),
        "num_clips": len(clips),
        "num_pairs": len(pairs),
        "source_counts": dict(source_counts),
        "split_counts": dict(split_counts),
        "split_clip_counts": dict(split_clip_counts),
        "split_pair_counts": dict(split_pair_counts),
        "pair_counts": dict(pair_counts),
        "preview": str(preview_path),
    }
    (REPORTS / "stackcube_bootstrap_v1_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "StackCube bootstrap v1 数据生成报告",
        "",
        f"实体数据目录：{DATASET}",
        f"索引目录：{INDICES}",
        f"预览图：{preview_path}",
        "",
        f"轨迹数：{len(traj)}",
        f"clip 数：{len(clips)}",
        f"pair 数：{len(pairs)}",
        "",
        "轨迹来源：",
    ]
    lines += [f"- {k}: {v}" for k, v in source_counts.items()]
    lines += ["", "split："]
    lines += [f"- {k}: trajectories={split_counts[k]}, clips={split_clip_counts[k]}, pairs={split_pair_counts[k]}" for k in sorted(split_counts)]
    lines += ["", "pair 类型："]
    lines += [f"- {k}: {v}" for k, v in pair_counts.items()]
    lines += [
        "",
        "说明：",
        "- 当前只做 StackCube，作为 reward model 数据链路 MVP。",
        "- 保存实体轨迹视频/h5/json，但 pair 不重复保存 A/B 帧，只通过 clip_id 和帧区间引用。",
        "- 后续可在这个索引上继续做 MiMo 标注、物理规则特征、fusion label 和 reward model 训练。",
    ]
    (REPORTS / "stackcube_bootstrap_v1_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
