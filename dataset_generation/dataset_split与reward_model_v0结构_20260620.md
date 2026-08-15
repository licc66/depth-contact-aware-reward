# 数据集划分与 Reward Model v0 结构建议

更新时间：2026-06-20

## 1. 当前可训练数据入口

本次划分使用的最终标签表：

```text
D:\Users\User\Desktop\reward_model_dataset\fusion_labels\bootstrap_v1_fusion_stereo_v1\final_pair_labels_v0.csv
```

输出目录：

```text
D:\Users\User\Desktop\reward_model_dataset\dataset_splits\bootstrap_v1_fusion_stereo_v1_clean
```

推荐后续训练直接读取：

```text
train_pairs.csv
val_pairs.csv
test_pairs.csv
train_preference_pairs.csv
val_preference_pairs.csv
test_preference_pairs.csv
train_order_pairs.csv
val_order_pairs.csv
test_order_pairs.csv
```

## 2. 正确样本过滤规则

保留样本同时满足：

```text
final_preference_label_v0 in {A>B, B>A}
needs_manual_review_v0 = false
final_preference_label_v0 == candidate_label
preference_loss_weight_v0 > 0
clip_a_video_path_local 和 clip_b_video_path_local 均存在
```

被剔除样本：

| 剔除原因 | 数量 |
| --- | ---: |
| final_label_not_clear | 270 |
| final_label_candidate_mismatch | 16 |
| 合计 | 286 |

说明：这里的 `candidate_label` 仍是构造出来的弱参考，不是人工真值。当前 split 的目标是先保证训练输入干净、可复现、无明显路径错误和标签冲突。

## 3. 划分结果

| split | 样本数 | source groups | preference rows | order rows |
| --- | ---: | ---: | ---: | ---: |
| train | 1109 | 17 | 778 | 331 |
| val | 184 | 3 | 128 | 56 |
| test | 203 | 4 | 146 | 57 |

泄漏检查：

| 检查项 | 数量 |
| --- | ---: |
| source group leakage | 0 |
| base success id leakage | 0 |

任务分布：

| split | PegInsertion | StackCube | StackPyramid |
| --- | ---: | ---: | ---: |
| train | 400 | 304 | 405 |
| val | 64 | 59 | 61 |
| test | 61 | 80 | 62 |

pair 类型分布：

| split | intra_success_temporal_gap | near_miss_vs_early_truncated | success_vs_offset_hard_negative | success_vs_peg_near_miss | success_vs_pyramid_near_miss | success_vs_truncated_terminal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 331 | 491 | 60 | 74 | 72 | 81 |
| val | 56 | 74 | 12 | 15 | 12 | 15 |
| test | 57 | 91 | 16 | 13 | 12 | 14 |

## 4. 输入视频审计

额外运行：

```text
dataset_generation\18_audit_reward_split_inputs.py
```

审计目录：

```text
D:\Users\User\Desktop\reward_model_dataset\dataset_splits\bootstrap_v1_fusion_stereo_v1_clean\input_audit_v1
```

审计结果：

| 项目 | 数量 |
| --- | ---: |
| pair rows | 1496 |
| unique videos | 438 |
| bad videos | 0 |
| bad pairs | 0 |
| frame count min | 20 |
| frame count max | 221 |

说明：当前 train / val / test 引用到的视频均可被 OpenCV 打开，首帧可读，采样帧索引没有越界。

## 5. Reward Model v0 设计原则

参考偏好学习 reward model 的常见做法，第一版建议采用：

```text
单个 clip -> scalar progress / reward score
两个 clip -> Bradley-Terry pairwise preference loss
```

也就是模型分别给 A/B 两个片段打分：

```text
s_A = R(clip_A, task)
s_B = R(clip_B, task)
P(A > B) = sigmoid(s_A - s_B)
```

如果标签是 `A>B`，训练时最大化 `sigmoid(s_A - s_B)`；如果标签是 `B>A`，反过来。

这一结构和 PEBBLE、T-REX、RLHF reward model、RL-VLM-F 这类 preference reward learning 思路一致；区别是本项目的输入不是纯 RGB，而是额外加入双目几何、接触阶段和稳定性特征。

## 6. 建议模型结构

### 6.1 输入

每个 clip 输入：

```text
RGB/video 语义特征
task text 特征
stereo/depth 几何特征
contact/stage 物理特征
可选：gripper/action/history 特征
```

注意：

```text
MiMo / VLM 判断结果只作为离线监督标签来源，不建议作为 reward model 的在线输入特征。
```

否则模型会依赖 teacher 输出，后面 RL 在线 rollout 时无法低成本调用。

### 6.2 分支结构

推荐 v0：

```text
RGB branch:
  sampled frames
  -> frozen CLIP/OpenCLIP image encoder
  -> temporal mean pooling 或 tiny temporal transformer
  -> rgb_embed

Task branch:
  task text
  -> frozen CLIP/OpenCLIP text encoder
  -> task_embed

Physical branch:
  stereo distance / depth error / geometry score
  contact stage / grasp ratio / support contact ratio / stage score
  -> MLP
  -> physical_embed

Fusion:
  concat(rgb_embed, task_embed, physical_embed)
  -> MLP / small transformer
  -> progress_score
  -> stage_logits
```

第一版可以先不端到端训练视觉 encoder，冻结 CLIP/OpenCLIP，只训练 temporal pooling、physical MLP 和 fusion head。这样数据量 1496 pairs 更稳。

### 6.3 输出

```text
progress_score: scalar，用于 pairwise preference loss
stage_logits: 当前任务阶段分类，可作为辅助 loss
optional_uncertainty: 后续如果需要再加，不建议第一版加入
```

## 7. 训练损失

主损失：

```text
Preference loss:
  -log sigmoid(score_winner - score_loser)
```

权重：

```text
使用 preference_loss_weight_v0
```

辅助损失：

```text
Order loss:
  对 intra_success_temporal_gap 样本使用较小权重，约束成功轨迹内后段分数更高。

Stage loss:
  用 contact_stage_label_proxy 或 stage id 训练 stage_logits。
```

建议第一版 loss：

```text
L = L_preference + 0.3 * L_order + 0.2 * L_stage
```

其中 0.3 和 0.2 是初始值，后续用 val/test 调。

## 8. 评估指标

至少保留四类指标：

| 指标 | 用途 |
| --- | --- |
| pair accuracy | 看 A/B 偏好判断是否正确 |
| near-miss false positive rate | 看失败/接近成功样本是否被误判为高进度 |
| success temporal monotonicity | 看成功轨迹内部是否越往后分越高 |
| task-wise accuracy | 分别看 StackCube / StackPyramid / PegInsertion，避免总分掩盖任务差异 |

建议报告时单独列：

```text
RGB-only
Physical-only
RGB + Physical fusion
```

这样能直接说明双目深度和接触信息是否真正提升了 near-miss / failure 判别。

## 9. 第一版实现顺序

建议下一步不要直接训练大模型，先做四个小脚本。由于 OpenCLIP 权重需要固定到本地 cache，先准备权重再预计算特征：

```text
19_prepare_openclip_weights.py
20_precompute_clip_frame_embeddings.py
21_train_reward_model_v0.py
22_eval_reward_model_v0.py
```

其中：

```text
19: 下载一次 OpenCLIP 预训练权重到本地固定 cache，并生成 manifest。
20: 从 train/val/test pairs 中涉及的视频抽帧，缓存 CLIP/OpenCLIP frame embeddings。
21: 读取 split csv + embedding cache + stereo/contact 数值列，训练 pairwise reward model。
22: 输出整体、分任务、分 pair_type 的 pair accuracy 和 near-miss false positive rate。
```

推荐权重 cache：

```text
D:\Users\User\Desktop\reward_model_dataset\model_cache\openclip
```

如果使用 `19_prepare_openclip_weights.py --source direct`，后续脚本应优先读取 manifest 里的本地 checkpoint 路径，而不是再次使用 `pretrained=openai` 触发联网下载。

## 10. 参考方法

当前结构主要参考这些公开 reward learning / VLM reward 方法：

```text
Deep RL from Human Preferences: pairwise clip preference -> reward model
T-REX: trajectory ranking -> scalar reward model
PEBBLE: preference over behavior clips -> reward model -> RL relabeling
RL-VLM-F: VLM 给 observation pair 偏好，再学习 reward function
RoboCLIP / VLM-RM / GVL / TOPReward: 用 VLM/CLIP 作为 zero-shot 或 teacher reward 信号
RoboReward / Robometer: 强调 negative / near-miss / trajectory comparison 对机器人 reward model 的重要性
```

可追溯链接：

| 方法 | 与本项目相关的点 | 链接 |
| --- | --- | --- |
| Deep RL from Human Preferences | 轨迹片段 pairwise preference -> reward predictor | https://proceedings.neurips.cc/paper/2017/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html |
| T-REX | ranked demonstrations -> scalar reward model | https://arxiv.org/abs/1904.06387 |
| PEBBLE | two behavior clips preference -> reward model -> relabel replay data | https://proceedings.mlr.press/v139/lee21i.html |
| RoboCLIP | 单个视频/文本 demonstration 通过 VLM 表征生成 robot reward | https://proceedings.neurips.cc/paper_files/paper/2023/hash/ae54ce310476218f26dd48c1626d5187-Abstract-Conference.html |
| RL-VLM-F | 用 VLM 给 observation pair 偏好，再学习 reward function | https://arxiv.org/abs/2402.03681 |
| GVL | 把 value estimation 改写成 shuffled frames temporal ordering | https://openreview.net/forum?id=friHAl5ofG |
| TOPReward | 从 VLM 内部 token logits/probabilities 提取 progress reward | https://arxiv.org/abs/2602.19313 |
| RoboReward | 机器人 VLM reward 数据集，强调 negatives 和 near-misses | https://arxiv.org/abs/2601.00675 |
| Robometer | trajectory comparisons 训练通用机器人 reward model | https://robometer.github.io/ |

对本项目最关键的落点是：

```text
VLM/MiMo 适合作为离线偏好标注来源；
双目深度和 contact/stage 适合作为可在线输入的物理约束特征；
训练出来的轻量 reward model 才是后续 RL 阶段真正调用的 dense reward。
```
