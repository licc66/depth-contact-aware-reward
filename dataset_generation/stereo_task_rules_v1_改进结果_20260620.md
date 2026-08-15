# stereo/depth task rules v1 改进结果

更新时间：2026-06-20

## 改动动机

阶段 2 的 `fusion_v0` 里，MiMo 与物理启发式冲突共 174 条，其中：

- contact/stage vs MiMo：110 条
- stereo/depth vs MiMo：57 条
- physical proxy 内部冲突：7 条

诊断发现 contact/stage 在冲突样本中与 candidate label 一致率很高，但 stereo/depth v0 在冲突样本中只有 31.6% 与 candidate label 一致。主要问题是 PegInsertion 仍用通用 object-goal distance，容易把 near-miss 的“靠近孔但未插入”误判为高进度。

## 新增脚本

`dataset_generation/16_build_stereo_task_rules_v1.py`

核心变化：

- StackCube：从通用 3D distance 改为 top cube 与 base cube 的 XY footprint overlap、高度差、移动量。
- StackPyramid：从单一 top-goal distance 改为 base readiness、top cube 到 base midpoint 的 XY/height 支撑关系。
- PegInsertion：从通用 peg-head/hole distance 改为 ManiSkill evaluate 记录的 `peg_head_at_hole_x` 与 `peg_head_at_hole_yz_norm`，对应插入轴深度与横向对齐误差。

注意：PegInsertion 当前使用的是仿真 evaluate 中的几何量，作为 stereo/depth 可估计几何 proxy 的上限版本；如果后续要更严格模拟真实双目，需要用 stereo pose/keypoint 估计替换这个字段来源。

## 输出

stereo/depth v1 输出：

`D:\Users\User\Desktop\reward_model_dataset\stereo_task_rules_v1`

关键文件：

- `pair_stereo_task_rules_v1.csv`
- `clip_stereo_task_rules_v1.csv`
- `frame_stereo_task_rules_v1.csv`
- `training_pairs_joined_stereo_v1.csv`
- `stereo_task_rules_v1_summary.json`

重新 fusion 后输出：

`D:\Users\User\Desktop\reward_model_dataset\fusion_labels\bootstrap_v1_fusion_stereo_v1`

关键文件：

- `final_pair_labels_v0.csv`
- `preference_loss_pairs_v0.csv`
- `order_loss_pairs_v0.csv`
- `manual_review_pairs_v0.csv`
- `diagnostics/conflict_diagnostics_report_v0.md`

## 前后对比

| 指标 | fusion_v0 | fusion_stereo_v1 |
| --- | ---: | ---: |
| total pairs | 1782 | 1782 |
| manual review rows | 174 | 127 |
| stereo-MiMo conflict rows | 57 | 17 |
| 当前版本 stereo-MiMo conflict 中 stereo 符合 candidate | 18 / 57 = 31.6% | 17 / 17 = 100% |
| success_vs_peg_near_miss 中 stereo-MiMo conflict | 21 | 0 |
| preference-loss rows | 1026 | 1056 |
| order-loss rows | 469 | 456 |

同一批旧版 57 条 stereo-MiMo conflict 的固定样本口径：

| 指标 | v0 | stereo_v1 |
| --- | ---: | ---: |
| 固定样本数 | 57 | 57 |
| clear stereo label 数 | 57 | 24 |
| 符合 candidate 的 clear label 数 | 18 | 24 |
| 若把 unsure 也算未命中 | 18 / 57 = 31.6% | 24 / 57 = 42.1% |
| 只看 clear label precision | 18 / 57 = 31.6% | 24 / 24 = 100% |
| 明确错误 label 数 | 39 | 0 |

所以提升不是简单从 31.6% 到 100% 的同分母提升；更准确地说，v1 把旧版 57 条冲突里的 39 条明确错误强判断降成了 `unsure` 或改正，保留下来的 24 条明确 stereo 判断全部与 candidate 一致。

全量 stereo/depth proxy 口径：

| 指标 | v0 | stereo_v1 |
| --- | ---: | ---: |
| all-pair agreement，unsure 计为未命中 | 1314 / 1782 = 73.7% | 1327 / 1782 = 74.5% |
| clear label precision | 1314 / 1393 = 94.3% | 1327 / 1327 = 100% |
| clear label 数 | 1393 | 1327 |

## stereo/depth v1 pair 标签分布

| label | count |
| --- | ---: |
| A>B | 1066 |
| B>A | 261 |
| unsure | 455 |

按 pair type：

| pair type | A>B | B>A | unsure |
| --- | ---: | ---: | ---: |
| intra_success_temporal_gap | 0 | 261 | 354 |
| success_vs_truncated_terminal | 111 | 0 | 7 |
| success_vs_offset_hard_negative | 96 | 0 | 0 |
| near_miss_vs_early_truncated | 719 | 0 | 27 |
| success_vs_pyramid_near_miss | 96 | 0 | 0 |
| success_vs_peg_near_miss | 44 | 0 | 67 |

## 结论

v1 已经解决最明显的 PegInsertion near-miss 误判：`success_vs_peg_near_miss` 不再出现 stereo 把 near-miss 判高于 success 的强冲突。

当前剩余 127 条 manual review 全部是 MiMo 与物理 proxy 冲突，且在 candidate weak reference 下物理侧都与 candidate 一致。后续建议：

- 保留这些 review 样本，不直接作为强监督。
- 抽样人工看 20-30 条，确认 candidate 是否真的可靠。
- 下一步可以基于 `fusion_stereo_v1` 做 train/val/test split 和 reward model v0。
