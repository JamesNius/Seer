# config.toml 参数详解

> 路径均为从 `mortal/` 目录出发的相对路径

---

## [control] 训练控制

| 参数 | 默认 | 说明 |
|------|------|------|
| `version` | 4 | 模型版本 (1/2/3/4)，v4 最强 |
| `online` | false | 在线训练模式，单机训设为 false |
| `device` | cuda | `cuda` 或 `cpu`，有 GPU 用 cuda |
| `enable_amp` | true | 混合精度，1080Ti 开启加速 ~40% |
| `enable_compile` | false | torch.compile，Windows 不稳定 |
| `batch_size` | 512 | 批次大小，1080Ti 11GB 可跑 768 |
| `opt_step_every` | 1 | 梯度累积步数，1=每步更新 |
| `save_every` | 400 | 每 N 步保存 checkpoint (~15分钟) |
| `test_every` | 20000 | 每 N 步对战评估 (~12小时) |
| `submit_every` | 400 | 在线模式参数提交间隔，离线无用 |

### 保存的文件
- `checkpoints/mortal.pth` — 每 save_every 步更新
- `checkpoints/best.pth` — test_play 刷新记录时复制

---

## [test_play] 对战评估

| 参数 | 默认 | 说明 |
|------|------|------|
| `games` | 400 | 每次评估打多少局 |
| `log_dir` | test_play_logs | 对局日志目录 |

评估结果写入 TensorBoard:
- `test_play/avg_ranking` 平均顺位
- `test_play/ranking/1st` ~ `4th` 顺位分布
- `test_play/behavior/agari` 和率

---

## [env] 环境

| 参数 | 默认 | 说明 |
|------|------|------|
| `gamma` | 1.0 | 折扣因子，麻将无折扣用 1.0 |
| `pts` | [6,4,2,0] | 顺位奖励 [1位,2位,3位,4位] |

---

## [resnet] 模型架构

| 参数 | 默认 | 说明 |
|------|------|------|
| `conv_channels` | 192 | 卷积通道数 |
| `num_blocks` | 40 | ResBlock 层数 |

**不要改**，改了和训练数据不兼容。

---

## [cql] / [aux] 损失权重

| 参数 | 默认 | 说明 |
|------|------|------|
| `min_q_weight` | 5 | CQL 保守约束强度，越大越保守 |
| `next_rank_weight` | 0.2 | 排名预测辅助任务权重 |

---

## [freeze_bn]

| 参数 | 默认 | 说明 |
|------|------|------|
| `mortal` | false | 是否冻结 BatchNorm，在线训练才开 |

---

## [optim] 优化器

| 参数 | 默认 | 说明 |
|------|------|------|
| `eps` | 1e-8 | AdamW epsilon |
| `betas` | [0.9, 0.999] | AdamW beta |
| `weight_decay` | 0.1 | 权重衰减 |
| `max_grad_norm` | 5.0 | 梯度裁剪上限，0=不裁剪 |

---

## [optim.scheduler] 学习率

| 参数 | 默认 | 说明 |
|------|------|------|
| `peak` | 2e-4 | 峰值学习率（warmup 后到达） |
| `final` | 2e-5 | 最终学习率（max_steps>0 时衰减到） |
| `warm_up_steps` | 2000 | 前 N 步线性加热 |
| `max_steps` | 0 | 衰减总步数，0=不衰减永远保持 peak |

---

## [dataset] 数据集

| 参数 | 默认 | 说明 |
|------|------|------|
| `globs` | [...] | 数据文件路径 glob 列表 |
| `file_batch_size` | 15 | 每批加载多少个文件 |
| `num_workers` | 1 | 数据加载线程数，Windows 建议 1 |
| `num_epochs` | 1 | 数据遍历轮数 |
| `file_index` | file_index.pth | 文件索引缓存，改 globs 后需删除 |

---

## [baseline] 对战对手

| 参数 | 默认 | 说明 |
|------|------|------|
| `baseline.train.state_file` | checkpoints/baseline.pth | 自对弈对手 |
| `baseline.test.state_file` | checkpoints/baseline.pth | test_play 对手 |

---

## [grp] 排名预测模型

| 参数 | 默认 | 说明 |
|------|------|------|
| `state_file` | grp.pth | GRP 模型文件 |
| `network.hidden_size` | 64 | GRP 隐层大小 |
| `network.num_layers` | 2 | GRU 层数 |

---

## 常见调参场景

| 场景 | 改动 |
|------|------|
| 验证流程跑通 | `save_every=100`, `test_every=1000000` |
| 正式开始训练 | `save_every=400`, `test_every=20000` |
| GPU 显存不足 | `batch_size=256` |
| loss 震荡 | `peak=1e-4`, `max_grad_norm=3.0` |
| 切换年份 | 改 `globs` + 删 `file_index.pth` |
| 开启评估 | `test_every=20000` + `copy mortal.pth baseline.pth` |
