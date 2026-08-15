# 阶段 2：fusion label v0 状态

更新时间：2026-06-20

## 输入

数据已从服务器迁移到：

`D:\Users\User\Desktop\reward_model_dataset`

阶段 2 输入表：

`D:\Users\User\Desktop\reward_model_dataset\training_tables\bootstrap_v1_mimo_stereo_contact_pairs\training_pairs_joined.csv`

该表包含 1782 个 pair，已经对齐：

- MiMo 视觉语义 preference
- stereo/depth 几何 proxy
- contact/stage 物理 proxy
- A/B clip 视频路径与帧区间

## 方法

本版只做保守规则融合，不训练模型，也不继续调用 MiMo。

规则要点：

- MiMo 作为离线视觉语义 preference 来源，不作为最终 reward 真值。
- contact/stage 作为硬阶段门控：若它与 MiMo 清晰冲突，进入 review，不直接当强标签。
- stereo/depth 作为阶段内几何进度信号：强冲突且没有 contact 支持时进入 review；弱冲突只降权。
- optional temporal pair 只作为低权重 order/progress 监督。
- unsure 样本保留，但第一版 preference loss 权重为 0。

## 输出

输出目录：

`D:\Users\User\Desktop\reward_model_dataset\fusion_labels\bootstrap_v1_fusion_v0`

关键文件：

- `final_pair_labels_v0.csv`
- `trainable_pairs_v0.csv`
- `preference_loss_pairs_v0.csv`
- `order_loss_pairs_v0.csv`
- `manual_review_pairs_v0.csv`
- `unsure_zero_weight_pairs_v0.csv`
- `fusion_summary_v0.json`
- `fusion_report_v0.md`

## 结果摘要

| item | count |
| --- | ---: |
| total pairs | 1782 |
| preference-loss rows | 1026 |
| order-loss rows | 469 |
| manual-review rows | 174 |
| local clip path missing | 0 |

最终偏好标签分布：

| final_preference_label_v0 | count |
| --- | ---: |
| A>B | 1044 |
| B>A | 451 |
| unsure | 287 |

fusion bucket：

| bucket | count |
| --- | ---: |
| main_semantic_preference | 752 |
| optional_order_semantic | 296 |
| physical_consensus_without_mimo | 255 |
| no_preference_unsure | 113 |
| contact_mimo_conflict_review | 110 |
| stereo_geometry_only_without_mimo | 107 |
| contact_stage_only_without_mimo | 70 |
| stereo_mimo_conflict_review | 57 |
| optional_order_low_confidence | 12 |
| physical_proxy_conflict_review | 7 |
| main_semantic_candidate_conflict | 3 |

## 复现命令

```powershell
python dataset_generation\14_build_fusion_labels_v0.py `
  --input D:\Users\User\Desktop\reward_model_dataset\training_tables\bootstrap_v1_mimo_stereo_contact_pairs\training_pairs_joined.csv `
  --out D:\Users\User\Desktop\reward_model_dataset\fusion_labels\bootstrap_v1_fusion_v0 `
  --old-root E:\reward_model_dataset `
  --new-root D:\Users\User\Desktop\reward_model_dataset
```

## 下一步

建议下一步先做训练/验证/测试划分，按 trajectory_id 或 clip_id 前缀避免同一轨迹泄漏；然后训练一个轻量 pairwise reward model v0，对比：

- MiMo-only
- physical-only
- fusion-v0

`manual_review_pairs_v0.csv` 暂时不要作为强监督直接训练。
