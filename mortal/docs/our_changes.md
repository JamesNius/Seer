# 我们新增/修改的文件

## 修改的原文件

| 文件 | 改动内容 |
|------|---------|
| `libriichi/Cargo.toml` | `[lib] name` 从 `riichi` 改为 `libriichi`，修复 PyO3 模块名不匹配 |
| `mortal/model.py` | `AuxNet` 添加可配置 `input_size` 参数（默认 1024，v1 用 512） |
| `mortal/train.py` | checkpoint 加载容错（optimizer/scheduler/scaler 缺失不崩溃） |
| `mortal/lr_scheduler.py` | `max_steps=0` 时保持 peak LR 不降（原版会降为 final） |
| `mortal/config.toml` | 完整重写：v4 + GPU + GRP + test_play 参数 |
| `.gitignore` | 添加训练产物目录、`.idea/` |
| `libriichi/__init__.py` (site-packages) | 添加 `sys.modules` 注册，修复 `from libriichi.xxx import yyy` |

## 新增文件

### 训练类

| 文件 | 用途 |
|------|------|
| `mortal/generate_training_data.py` | 用 OneVsThree arena 生成自对弈训练数据 |

### 工具类

| 文件 | 位置 | 用途 |
|------|------|------|
| `split_data.py` | `tools/` | 年份数据拆分为 1 万文件子目录 |
| `switch_year.bat` | `scripts/` | 切换训练年份（改 config + 清缓存） |
| `run_train_loop.ps1` | `scripts/` | 训练崩溃自动重启守护脚本 |
| `mortal.bat` | `scripts/` | mortal.py 批处理封装 |

### 评估类

| 文件 | 位置 | 用途 |
|------|------|------|
| `run_test.py` | `review/` | 快速对战评估（vs baseline） |
| `reverse_mjai_review.py` | `review/` | mjai-reviewer JSON → mjai/tenhou 格式 |
| `run_review.bat` | `scripts/` | 一键复盘：牌谱 → Mortal 分析 → HTML |

### 文档类

| 文件 | 位置 | 内容 |
|------|------|------|
| `README.md` | `docs/` | 文档索引 + 待写推荐 |
| `config_guide.md` | `docs/` | config.toml 参数详解 |
| `obs_channels.md` | `docs/` | v4 模型 1012 通道完整说明 |
| `roadmap.md` | `docs/` | 优化路线图 + 行动清单 |
| `PLAN.txt` | `mortal/` | 分阶段训练计划 |
| `commands.txt` | `mortal/` | 常用命令速查 |

### 实验类

| 文件 | 位置 | 说明 |
|------|------|------|
| `light_search.py` | `mortal/` | 前 6 巡搜索（实验阶段，当前关闭） |

## 编译产物（不提交 git）

| 文件 | 说明 |
|------|------|
| `mortal/checkpoints/` | 模型权重文件 |
| `mortal/dummy_grp.pth`, `dummy.pth` | 初始模型/GRP，已废弃 |
| `mortal/grp.pth` | 训练好的 GRP 模型 |
| `libriichi.cp312-win_amd64.pyd` | 编译好的 Rust 模块 (site-packages) |
