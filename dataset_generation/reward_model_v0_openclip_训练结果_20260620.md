# Reward Model v0 OpenCLIP 训练结果

更新时间：2026-06-20

## 1. 本次训练输入

数据划分：

```text
D:\Users\User\Desktop\reward_model_dataset\dataset_splits\bootstrap_v1_fusion_stereo_v1_clean
```

OpenCLIP 特征：

```text
D:\Users\User\Desktop\reward_model_dataset\reward_model_features\openclip_vit_b32_v1
```

模型输出：

```text
D:\Users\User\Desktop\reward_model_dataset\reward_model_runs\reward_model_v0_openclip
```

本次训练使用：

```text
OpenCLIP ViT-B-32 image/text embedding
stereo/depth 几何特征
contact/stage 物理特征
task one-hot
```

没有把 MiMo/VLM 判断结果作为模型输入；MiMo 只作为离线标签来源的一部分。

## 2. 训练设置

| 项目 | 数值 |
| --- | --- |
| train pairs | 1109 |
| val pairs | 184 |
| test pairs | 203 |
| epochs | 80 |
| batch size | 128 |
| GPU | RTX 4070 Ti |
| loss | pairwise Bradley-Terry preference loss |

训练脚本：

```text
dataset_generation\21_train_reward_model_v0.py
```

训练时已加入 `tqdm` 进度条。

## 3. 整体结果

| variant | train acc | val acc | test acc | test preference acc | test order acc | hard-neg test acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rgb_only | 0.993 | 0.924 | 0.916 | 0.952 | 0.825 | 0.952 |
| physical_only | 0.964 | 0.973 | 1.000 | 1.000 | 1.000 | 1.000 |
| fusion | 0.989 | 0.995 | 1.000 | 1.000 | 1.000 | 1.000 |

best validation checkpoint：

| variant | best epoch | best val acc | train acc at best |
| --- | ---: | ---: | ---: |
| rgb_only | 36 | 0.924 | 0.976 |
| physical_only | 71 | 0.973 | 0.957 |
| fusion | 17 | 0.995 | 0.988 |

test 错误数：

| variant | wrong / total | PegInsertion wrong | StackCube wrong | StackPyramid wrong |
| --- | ---: | ---: | ---: | ---: |
| rgb_only | 17 / 203 | 1 | 7 | 9 |
| physical_only | 0 / 203 | 0 | 0 | 0 |
| fusion | 0 / 203 | 0 | 0 | 0 |

## 4. 分任务结果

| variant | split | PegInsertion | StackCube | StackPyramid |
| --- | --- | ---: | ---: | ---: |
| rgb_only | val | 0.984 | 0.983 | 0.803 |
| rgb_only | test | 0.984 | 0.913 | 0.855 |
| physical_only | val | 0.969 | 1.000 | 0.951 |
| physical_only | test | 1.000 | 1.000 | 1.000 |
| fusion | val | 1.000 | 1.000 | 0.984 |
| fusion | test | 1.000 | 1.000 | 1.000 |

## 5. 结果解释

当前结果说明：

```text
RGB-only 已经能较好拟合偏好标签，但在 StackPyramid 和成功轨迹内部 order pair 上明显更弱。
physical-only 明显更贴合当前 clean fusion labels。
fusion 进一步把 RGB 语义和物理特征合起来，验证集最高，整体最稳。
```

需要谨慎：

```text
当前标签是 MiMo + stereo/contact 规则融合得到的弱监督标签，不是人工真值。
physical-only / fusion 的 test=1.000 说明模型很好复现了这套标签规则；
它还不能直接证明模型已经具备真实机器人泛化能力。
```

后续要证明 reward model 真的有用，需要再做：

```text
人工抽检一部分 val/test 预测错误和高置信正确样本；
构造新的 out-of-distribution near-miss / failure 视频；
检查 reward 曲线是否随成功过程单调上升；
最终再接 RL rollout，观察 dense reward 是否能区分假成功。
```
