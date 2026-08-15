# bootstrap_v1 MiMo 与训练前 pair table 状态

更新时间：2026-06-11

## 已完成内容

本轮已经完成需要调用 MiMo 的离线视觉语义 preference 标注，并把 MiMo 标签、双目几何 proxy、contact/stage proxy 合并成训练前 pair-level 总表。

## MiMo 输出

合并目录：

`E:\reward_model_dataset\vlm_labels\mimo_bootstrap_pairwise_all_v1`

关键文件：

- `mimo_pairwise_labels_all.csv`
- `mimo_pairwise_labels_all.json`
- `mimo_pairwise_labels_all_summary.json`
- `MiMo_pairwise_labels_all_report.md`

数量：

| label source | pair 数 |
| --- | ---: |
| core / true pair | 1167 |
| optional intra-success temporal pair | 615 |
| total | 1782 |

MiMo preference 分布：

| preference | count |
| --- | ---: |
| A>B | 814 |
| B>A | 416 |
| unsure | 552 |

清晰偏好中，MiMo 与候选规则标签一致率：0.8837。

## 训练前 joined table

输出目录：

`E:\reward_model_dataset\training_tables\bootstrap_v1_mimo_stereo_contact_pairs`

关键文件：

- `training_pairs_joined.csv`
- `training_pairs_joined.json`
- `training_pairs_joined_summary.json`
- `training_pairs_joined_report.md`

joined table 状态：

| item | value |
| --- | ---: |
| rows | 1782 |
| columns | 63 |
| missing MiMo | 0 |
| missing stereo | 0 |
| missing contact/stage | 0 |
| nonzero preference weight rows | 1050 |
| sum preference weight hint | 861.3 |

保守监督 bucket：

| bucket | count | 含义 |
| --- | ---: | --- |
| main_preference | 744 | core/true pair 中 MiMo 清晰，且未被物理 proxy 判为冲突 |
| optional_order | 294 | optional 时间顺序 pair 中 MiMo 清晰且与候选时间顺序一致 |
| hard_conflict_review | 180 | MiMo 与 stereo/contact 任一清晰物理 proxy 冲突，暂不作为强标签 |
| no_preference_unsure | 552 | MiMo unsure，暂不给 preference loss |
| low_weight_optional_or_candidate_conflict | 12 | optional 中 MiMo 清晰但与候选顺序不一致，可低权重或人工复查 |

## 使用注意

- 这些 MiMo 标签是离线视觉语义 preference，不是最终 reward 真值。
- `preference_label_hint_v0` 和 `preference_loss_weight_hint_v0` 只是第一版训练用的保守提示字段。
- `hard_conflict_review` 建议先保留，不要直接当强监督；后续应由 depth/contact/stage-aware fusion 再处理。
- `unsure` 样本可用于不确定性分析或人工复查，第一版 preference loss 中建议权重为 0。
- 没有把 API key 写进脚本或输出文件。

## 复现命令

合并 MiMo core + optional：

```powershell
python dataset_generation\12_merge_mimo_pairwise_labels.py `
  --core E:\reward_model_dataset\vlm_labels\mimo_bootstrap_pairwise_v1\mimo_pairwise_labels.csv `
  --optional E:\reward_model_dataset\vlm_labels\mimo_bootstrap_pairwise_v1_optional\mimo_pairwise_labels.csv `
  --out E:\reward_model_dataset\vlm_labels\mimo_bootstrap_pairwise_all_v1
```

构建训练前 joined table：

```powershell
python dataset_generation\13_build_training_pair_table.py `
  --queues E:\reward_model_dataset\pair_indices\stackcube_bootstrap_v1\pair_annotation_queue.csv `
           E:\reward_model_dataset\pair_indices\stackpyramid_bootstrap_v1\pair_annotation_queue.csv `
           E:\reward_model_dataset\pair_indices\peginsertion_bootstrap_v1\pair_annotation_queue.csv `
  --mimo-labels E:\reward_model_dataset\vlm_labels\mimo_bootstrap_pairwise_all_v1\mimo_pairwise_labels_all.csv `
  --stereo-pair-labels E:\reward_model_dataset\stereo_features\stackcube_bootstrap_v1\pair_stereo_geometry_labels.csv `
                       E:\reward_model_dataset\stereo_features\stackpyramid_bootstrap_v1\pair_stereo_geometry_labels.csv `
                       E:\reward_model_dataset\stereo_features\peginsertion_bootstrap_v1\pair_stereo_geometry_labels.csv `
  --contact-pair-labels E:\reward_model_dataset\contact_stage_features\stackcube_bootstrap_v1\pair_contact_stage_labels.csv `
                        E:\reward_model_dataset\contact_stage_features\stackpyramid_bootstrap_v1\pair_contact_stage_labels.csv `
                        E:\reward_model_dataset\contact_stage_features\peginsertion_bootstrap_v1\pair_contact_stage_labels.csv `
  --out E:\reward_model_dataset\training_tables\bootstrap_v1_mimo_stereo_contact_pairs
```
