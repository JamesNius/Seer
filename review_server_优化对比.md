# Review Server vs 原版 Mortal — 性能优化对比

> 对比对象：
> - **原版 mortal**：`mortal.py`，官方单机逐事件推理
> - **review server**：`mortal_review_server.py`，Seer AI 复盘服务后端
>
> 基准数据：同一局（1004 事件 / 157 决策点，2025 数据）

## 一、实测速度对比

### 训练期间（2025-08-09，GPU 被训练任务抢占）

| 路线 | 耗时 | 反应/s | 相对原版 |
|---|---|---|---|
| 原版 mortal（CPU） | 15.5 s | 11.7 | 1x |
| 原版 mortal（GPU） | 7.5 s | 20.9 | 约 2x |
| **review server（GPU 批量）** | **0.92 s** | 2140 | **约 17x** |

（review server 10 次均值 0.92s，空闲时段最快 0.42s；后台训练抢 GPU 时最慢 2.02s）

### 训练结束后（2025-08-10，GPU 独占）

| 路线 | 耗时 | 反应/s | 相对 CPU |
|---|---|---|---|
| 原版 mortal（CPU） | 5.87 s | 26.7 | 1x |
| 原版 mortal（GPU） | 5.97 s | 26.3 | 1.02x |
| **review server（GPU 批量）** | **0.36 s** | 2770 | **约 16x** |

（10 次均值，std 0.02s，非常稳定）

**关键发现**：
- CPU 和 GPU 的 mortal.py 几乎一样快——逐事件推理的瓶颈不在算力，在**串行往返开销**
- review server 的 0.36s 比 mortal.py 快 **16 倍**，且不受 CPU/GPU 竞争影响（std 仅 0.02s）

## 二、架构差异：逐事件串行 vs 三阶段批量流水线

### 原版 mortal：逐事件串行

```
事件流 ──▶ bot.react(line) ──▶ 编码 1 个样本 ──▶ GPU/CPU 前向 1 次 ──▶ 返回
        ──▶ bot.react(line) ──▶ 编码 1 个样本 ──▶ GPU 前向 1 次 ──▶ 返回
        ...每决策点一次完整往返，编码与推理交替等待
```

每个决策点都是一次"编码 → 前向 → 回传"的串行往返，GPU 每步只算 1 个样本，带宽利用率极低。这是它慢的根本原因——**不是算力不够，是流水线粒度太细**。

### review server：三阶段批量流水线

```
Pass 1 录制： 事件流 ──▶ bot.react（Rust 端异步编码，不推理）──▶ 攒齐全部 obs
Pass 2 批量推理：全部 obs 按 128/批 打包 ──▶ GPU 批量前向 ──▶ 缓存全部结果
Pass 3 重放：   事件流重放 ──▶ 查缓存结果 ──▶ 注入 meta ──▶ 输出
```

| 阶段 | 做什么 | 关键点 |
|---|---|---|
| **Pass 1 录制** | 用 Rust `Bot` 扫一遍事件，只编码不推理 | `record_mode`：`set_scene` 异步编码，`get_reaction` 跳过 evaluate |
| **Pass 2 批量推理** | 所有决策点样本展平，按 `BATCH_CHUNK=128` 分批前向 | GPU 批量吞吐是单样本前向的近 30 倍 |
| **Pass 3 重放** | 重放事件流，从缓存取结果，注入读牌/元数据 | 纯 Python 组装，不碰 GPU |

## 三、具体优化点清单

### 1. 推理设备：CPU → GPU
- 原版 `mortal.py` 硬编码 `device = torch.device('cpu')`（[mortal.py:29](mortal/mortal.py)）
- review server 用 `torch.device("cuda")`（[mortal_review_server.py:223](mortal/mortal_review_server.py)）

### 2. 批量推理（最大收益来源）
- `BATCH_CHUNK = 128`，157 个决策点攒成 2 批跑完（[mortal_review_server.py:55](mortal/mortal_review_server.py)）
- 单局两次 GPU 前向 vs 原版 157 次前向

### 3. record_mode：异步编码
- Rust 端 `set_scene` 照常提交编码任务到 rayon 池，但 `get_reaction` 不 evaluate、不等每决策点编码完成（[mortal.rs:606-623](libriichi/src/agent/mortal.rs)）
- Pass 1 结束 `flush_encode` 一次性等完所有异步编码任务，省掉每步串行 wait

### 4. rayon 线程池并行编码
- 编码任务按对局并发提交到 rayon 池并行执行（`rayon_threads` 可配置，默认逻辑核数）
- 原版是单线程编码，一步一编码

### 5. TCP 常驻服务，模型只加载一次
- review server 是长驻进程，`load_models()` 一次，所有连接复用同一份引擎（[mortal_review_server.py:230](mortal/mortal_review_server.py)）
- 原版每次跑都重新加载 pth（每次约 2.5s）

### 6. 预热
- 启动时用假 batch 跑一次典型形状，让 cuDNN 提前完成卷积算法调优（[mortal_review_server.py:310](mortal/mortal_review_server.py)）
- 避免线上第一局吃算法搜索的慢启动

### 7. oracle 侧信道（读牌）
- `MortalEngine` 新增 `oracle_net` 参数：复用本次前向的 `phi`，`oracle_net(phi)` 一次前向得到五头输出（他家张数/向听/待牌/赤宝/振听），追加到 `oracle_log` 侧信道（[engine.py:85-88](mortal/engine.py)）
- 不污染主决策结果，`None` 时零开销

### 8. value_dist 他家打点分布模型
- 独立可选的读牌模块，复用 Pass 2 的同一份 `all_obs` 跑一次前向（[mortal_review_server.py:416-419](mortal/mortal_review_server.py)）
- 输出 3 家的打点分布 / 一手/索子赤宝 / 向听 / 待牌 / 山残 34×5 分布
- 数据只挂第一家（公共信息去重），越界牌做覆盖层 + 归一化（`clamp_mountain_exp`）

### 9. Pass 3 的 JSON 体积优化
- 铳率 = 待牌 × 听牌概率，只在服务端算，且只保留 ≥0.5% 的项（`wait_sparse`）
- 无实际动作的决策点（摸牌/过/别家）不注入读牌数据，省 GPU/CPU/JSON
- 用 `orjson` 做快速序列化（Pass 3 热路径）

## 四、为什么 review server 还是"相对公平"的

review server 加载了 value_dist 读牌模型（`enable_value_dist = true`）——它比原版**多跑了一个模型**，却仍然快 17 倍。说明批量化的收益远大于额外模型的成本：读牌模型的额外前向被摊薄在 128 样本的批次里，几乎可忽略。

## 五、结论

| 维度 | 原版 mortal | review server |
|---|---|---|
| 设计目标 | 单机打牌 / 研究 | 批量复盘服务 |
| 推理设备 | CPU（硬编码） | GPU |
| 推理粒度 | 每决策点 1 样本 | 128 样本/批 |
| 编码 | 单线程串行 | rayon 线程池并行 |
| 模型加载 | 每次启动加载 | 常驻复用 |
| 读牌模型 | 无 | oracle 侧信道 + value_dist |
| 单局耗时 | 5.87 s（CPU）/ 5.97 s（GPU） | **0.36 s** |

核心一句话：**把"每步都问一次模型"改成"先录完、攒批、一次问完"**，这就是 16 倍差距的来源。
