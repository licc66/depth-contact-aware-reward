# 深度/接触条件式 Push-Back 过程 Reward 想法

## 1. 背景

PROGRESSOR 的 push-back 思想是：

```text
如果机器人在线探索轨迹不像专家轨迹，但 reward model 误给了高 progress，
就把这个 progress 往低处压。
```

这个思想很有用，但也有一个问题：

```text
它主要根据“是否偏离专家分布”来压低 progress。
如果在线轨迹其实已经做得不错，但分布和专家略有不同，也可能被误伤。
```

因此，我这里可以借鉴 push-back，但不完全照搬。核心改动是：

```text
不是所有 online trajectory 都 push-back。
只有当语义进度高、但物理几何/接触证据不支持时，才 push-back。
```

## 2. 核心创新想法

先让 VLM 或视觉进度模型给出一个语义进度：

```text
p_sem = semantic_progress(image/video, task_prompt)
```

它回答的是：

```text
当前画面看起来像任务完成到了哪一步？
```

然后用深度和接触信息给出一个物理一致性判断：

```text
p_phys = physical_consistency(depth, contact, env_truth)
```

它回答的是：

```text
当前状态在几何和接触上是否真的支持这个进度？
```

最终 reward 不直接等于 `p_sem`，而是：

```text
inconsistency = max(0, p_sem - p_phys)
p_final = clamp(p_sem - lambda * inconsistency, 0, 1)
```

直觉：

```text
VLM 说进度高，但深度/接触不支持
=> 压低 progress

VLM 说进度高，深度/接触也支持
=> 保留 progress

VLM 说进度低
=> 不强行抬高，避免 reward hacking
```

## 3. 和 PROGRESSOR 的区别

PROGRESSOR 的 push-back 更像：

```text
online rollout 不像专家
=> push back
```

我这里的版本更像：

```text
semantic progress 和 physical evidence 不一致
=> push back
```

这样可以避免一个问题：

```text
机器人找到了不同于专家、但仍然有效的解法。
```

如果这个解法在深度和接触上是正确的，就不应该被压低。

## 4. 为什么能恢复 progress

PROGRESSOR 没有显式恢复机制，主要靠策略重新回到专家分布。

我这里可以有更自然的恢复方式：

```text
只要后续状态的深度/接触证据重新变好，push-back 惩罚就自动减小。
```

例如 StackCube：

```text
VLM 认为 cube 已经接近目标上方，p_sem = 0.8
但环境真值/双目深度显示 cube 高度不对，p_phys = 0.4
=> p_final 被压到 0.5 左右

后续机器人把 cube 提高并移动到 cubeB 正上方，p_phys = 0.8
=> push-back 消失，p_final 恢复到 0.8
```

也就是说，恢复不依赖“这条轨迹是否像专家”，而依赖：

```text
当前物理状态是否重新合理。
```

## 5. 第一版先用环境真值

当前阶段建议先用 ManiSkill 环境真值做 oracle 版本：

```text
object position
tcp position
goal position
contact state
success condition
```

原因：

```text
环境真值误差最小，适合先验证算法思想。
```

这一版可以写成：

```text
VLM/视觉模型给 p_sem
环境真值给 p_phys
最终得到 p_final
```

后续再替换成：

```text
双目视差估计的 depth
可观测接触/力觉/contact signal
```

这样实验故事更清楚：

```text
oracle version 证明算法上限
stereo/contact version 证明实际可行性
```

## 6. 可以使用的物理一致性信号

### StackCube

```text
tcp-object distance
object-goal 3D distance
object 是否在 support cube 上方
object 高度是否接近目标高度
gripper 是否稳定抓取
放置后 object 是否静止
```

### StackPyramid

```text
top cube 到 base center 的距离
top cube 高度是否合理
base cubes 是否已经形成稳定底座
gripper 是否在正确阶段接触 top cube
```

### PegInsertion

```text
peg head 到 hole 的距离
peg 轴向插入深度
yz 对齐误差
peg 与 hole/box 的接触是否合理
是否出现错误碰撞
```

## 7. 可能的 reward 形式

### 形式 A：直接压低语义进度

```text
p_final = p_sem - lambda * max(0, p_sem - p_phys)
reward = p_final
```

优点：

```text
简单，容易解释，适合第一版。
```

### 形式 B：potential-based shaping

```text
Phi(s) = p_final(s)
r_process = gamma * Phi(s_next) - Phi(s)
```

优点：

```text
更符合 reward shaping 理论，写论文时更好解释。
```

### 形式 C：阶段内 push-back

先判断阶段：

```text
approach
grasp
move
align
place / insert
```

再在每个阶段内部做：

```text
p_stage_final = p_stage_sem - lambda * inconsistency_stage
```

优点：

```text
适合长任务，避免整段任务只有一个 0 到 1 progress。
```

## 8. 推荐实验对比

可以设计这些 baseline：

```text
1. VLM-only progress
2. depth/contact-only process reward
3. PROGRESSOR-style unconditional push-back
4. condition-based push-back, env truth version
5. condition-based push-back, stereo/contact estimated version
```

评价指标：

```text
Progress MAE
Stage Accuracy
Spearman / VOC
Near-miss false positive rate
Success rate under RL
Reward hacking cases
```

尤其要重点展示：

```text
VLM 认为任务快完成了，但深度/接触不成立的 near-miss。
```

这类样例最能体现这个方法的价值。

## 9. 当前可以写成的创新点

暂定表述：

```text
提出一种物理一致性条件下的 progress push-back 过程 reward。
该方法将语义进度估计与深度几何、接触状态结合，
只在高语义进度与低物理一致性冲突时压低 reward，
从而降低 VLM-only reward 对 near-miss 和视觉相似状态的误判。
```

更短一点：

```text
Semantic progress proposes, physical consistency verifies.
```

中文理解：

```text
语义负责提出“看起来到哪一步了”，
深度和接触负责验证“物理上是否真的到了这一步”。
```

## 10. 后续待补

```text
1. 选定最终 p_phys 公式
2. 把已有三任务的环境真值 depth/contact 信号整理成统一接口
3. 设计 near-miss 失败轨迹生成方式
4. 先在 StackCube 上验证 p_final 曲线是否合理
5. 再扩展到 StackPyramid 和 PegInsertion
6. 最后把 env truth depth 替换为 stereo SGBM / 深度模型结果
```
