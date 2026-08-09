# 项目结构

```
Mortal/
├── README.md
├── LICENSE                    (AGPL-3.0)
├── Cargo.toml                 (Rust workspace)
├── Dockerfile                 (Docker 推理镜像)
├── .gitignore
│
├── libriichi/                 ← Rust 核心引擎
│   ├── Cargo.toml             (PyO3 模块)
│   └── src/
│       ├── lib.rs             (Python 模块注册)
│       ├── consts.rs          (obs_shape, ACTION_SPACE)
│       ├── tile.rs            (牌数据结构)
│       ├── state/
│       │   ├── player_state.rs (PlayerState 状态机)
│       │   └── obs_repr.rs    (1012 通道编码)
│       ├── agent/
│       │   ├── mortal.rs      (MortalBatchAgent 推理调度)
│       │   └── mjai_log.rs    (mjai 日志解析)
│       ├── dataset/
│       │   ├── gameplay.rs    (GameplayLoader 训练数据)
│       │   └── grp.rs         (GRP 排名预测数据)
│       ├── arena/             (自对弈引擎)
│       ├── mjai/              (mjai 协议)
│       ├── stat/              (统计评估)
│       └── algo/              (和了判定/向听/SP搜索)
│
├── mortal/                    ← Python 训练 & 推理
│   │
│   ├── 核心训练 ────────────
│   │   ├── train.py           ★ 主训练脚本
│   │   ├── train_grp.py       GRP 排名模型训练
│   │   ├── model.py           Brain/DQN/AuxNet/GRP 模型定义
│   │   ├── engine.py          MortalEngine 推理引擎
│   │   ├── dataloader.py      数据加载器 (FileDatasetsIter)
│   │   ├── reward_calculator.py  奖励计算
│   │   └── lr_scheduler.py    学习率调度器
│   │
│   ├── 配置 ────────────────
│   │   ├── config.toml        ★ 主配置 (我们重写)
│   │   ├── config.example.toml 作者示例
│   │   ├── config.py          配置加载 (import toml)
│   │   ├── baseline_config.toml
│   │   └── server_config.toml
│   │
│   ├── 推理/对战 ───────────
│   │   ├── mortal.py          ★ mjai Bot 入口
│   │   ├── player.py          TestPlayer/TrainPlayer
│   │   ├── one_vs_three.py    1v3 对战
│   │   ├── client.py          在线客户端
│   │   └── server.py          在线服务端
│   │
│   ├── 工具 (我们新增) ─────
│   │   tools/
│   │   └── split_data.py      数据拆分
│   │   scripts/
│   │   ├── switch_year.bat    切换训练年份
│   │   ├── run_train_loop.ps1 训练守护脚本
│   │   ├── mortal.bat         mortal.py 封装
│   │   └── run_review.bat     一键复盘
│   │   review/
│   │   ├── run_test.py        对战评估
│   │   └── reverse_mjai_review.py 格式转换
│   │
│   ├── 实验 ────────────────
│   │   ├── light_search.py    前6巡搜索 (关闭)
│   │   └── generate_training_data.py  自对弈数据生成
│   │
│   ├── 文档 (我们新增) ─────
│   │   docs/
│   │   ├── README.md
│   │   ├── config_guide.md    参数详解
│   │   ├── obs_channels.md    通道说明
│   │   ├── roadmap.md         优化路线图
│   │   ├── our_changes.md     修改清单
│   │   └── project_structure.md 本文档
│   │   commands.txt           命令速查
│   │   PLAN.txt               训练计划
│   │
│   ├── 运行时 ──────────────
│   │   checkpoints/           模型保存
│   │   runs/                  TensorBoard 日志
│   │   json/                  测试/复盘数据
│   │   test_data/             临时测试数据
│   │
│   └── 数据文件 ────────────
│       file_index.pth         文件索引缓存
│       grp_file_index.pth     GRP 索引缓存
│       grp.pth                训练好的 GRP
│       dummy_grp.pth          随机 GRP (备用)
│       dummy.pth              随机 baseline (备用)
│
├── checkpoints/               (原始 pre-trained 模型，已弃用)
├── docs/                      (项目官方 mdbook 文档)
├── log-viewer/                (牌谱查看器)
├── exe-wrapper/               (Rust 可执行文件包装)
└── target/                    (Rust 编译产物)
```

## 关键文件速查

| 做什么 | 文件 |
|--------|------|
| 训练 | `python train.py` |
| 配参数 | 编辑 `config.toml` |
| 看训练曲线 | `tensorboard --logdir runs` |
| 测模型 | `python review/run_test.py 40` |
| 切换年份 | `scripts/switch_year.bat 2021` |
| 拆分数据 | `python tools/split_data.py E:/天凤训练数据/2021` |
