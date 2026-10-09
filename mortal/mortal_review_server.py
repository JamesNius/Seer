"""
Mortal Review TCP Server — 常驻 GPU 推理服务（批量推理优化版）
=============================================================
用法:
  python mortal_review_server.py [端口号]

批量推理架构:
  Pass 1: RecordEngine 收集所有 (obs, masks, invisible_obs)，Rust 正常推进状态
  Pass 2: 真引擎批量推理 (chunk=128)
  Pass 3: ReplayEngine 按序返回预计算结果，Rust 重新生成 reaction JSON
  Pass 4: GRP phi_matrix 计算

Oracle (可选): config [review] enable_oracle = true 开启，
  每步读牌数据注入 reaction 的 meta.oracle（list，通常长度1，kan决策为2）
"""

import json
import logging
import os
import socket
import threading
import sys
import time
import traceback
import inspect
import warnings
from contextlib import nullcontext
from datetime import datetime, timezone

import numpy as np
import torch
import orjson  # 快速 JSON(Pass 3 热路径)
from model import Brain, DQN, GRP, CategoricalPolicy
try:
    from model import OracleNet          # 老分支有; 合并后的 policy 分支已移除这个头
except ImportError:
    OracleNet = None
from engine import MortalEngine
from value_dist.infer import load_predictor
from libriichi.mjai import Bot
from libriichi.dataset import Grp

# ---- 初始化 ----
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)8s %(filename)12s:%(lineno)-4s %(message)s",
)
# 预触发一次 NumPy 不可写数组警告（libriichi 返回的数组不可写），避免运行期刷屏
warnings.simplefilter("ignore")
_dummy = np.array([])
_dummy.setflags(write=False)
torch.as_tensor(_dummy)
del _dummy
warnings.simplefilter("default")

from config import config  # noqa: E402

DEFAULT_PORT = 9876
BATCH_CHUNK = 128
# GPU 全局锁：多连接并发时推理串行执行（等价于排队，结果一致）。
# 实验用：NO_GPU_LOCK=1 时关闭（测 CPU 并发能力，GPU 下会串谱勿开）。
_NO_GPU_LOCK = os.environ.get("NO_GPU_LOCK", "0") == "1"
_GPU_LOCK = threading.Lock()
# 并发连接信号量：限制同时处理的连接数，防止过多线程挤爆 GPU 显存。
# 超过上限的连接在 accept 后等待信号量(排队)，不拒绝。
_CONN_SEM = threading.BoundedSemaphore(4)


# ── 可见牌追踪器（mountain 越界覆盖层用）────────────────────────────────────
# 口径与提取管线 value_dist/extract.py 完全一致（复用 HandTracker + tile_id/_deaka），
# 禁止另起口径。visible[t] = 河(net=去被鸣) + 副露 + 指示牌 + 四家暗手。
# 模型 mountain 输出自由，可能给已见牌分配剩余概率 → 越界(> 4-visible)。
# 覆盖层: 对 g > (4-visible[t]) 清零 → 归一化 → 算 E[mountain]，数学上不越界。
from collections import Counter
from value_dist.hand_tracker import HandTracker, tile_id, _deaka


class VisibleTracker:
    """同步喂 mjai 事件，维护每牌已见张数(34维) + 每家立直/副露状态。
    与提取管线同口径。副露数用于铳率计算(四副露=必然听牌)。"""

    def __init__(self, viewer):
        self.viewer = viewer
        self.tracker = HandTracker()
        self.kawa_pool = Counter()      # 打出的牌
        self.called_pai = Counter()     # 被鸣走的牌(从河里剔除)
        self.reach_accepted = [False] * 4   # 每家是否已立直(供 pTenpai 判定)
        self.meld_count = [0] * 4           # 每家副露数(供 pTenpai 判定)

    def update(self, e):
        if not isinstance(e, dict):
            return
        t = e.get("type")
        if t == "start_kyoku":
            self.kawa_pool = Counter()
            self.called_pai = Counter()
            self.reach_accepted = [False] * 4
            self.meld_count = [0] * 4
        elif t == "dahai":
            self.kawa_pool[tile_id(_deaka(e.get("pai", "")))] += 1
        elif t in ("chi", "pon", "daiminkan"):
            self.called_pai[tile_id(_deaka(e.get("pai", "")))] += 1
        elif t == "reach_accepted" and "actor" in e:
            self.reach_accepted[e["actor"]] = True
        elif t in ("chi", "pon", "daiminkan", "ankan") and "actor" in e:
            self.meld_count[e["actor"]] += 1
        elif t == "kakan" and "actor" in e:
            self.meld_count[e["actor"]] += 1
        try:
            self.tracker.update(e)
        except Exception:
            pass

    def p_tenpai(self, pidx):
        """该家听牌概率: 已立直/四副露=必然听牌→1, 否则用模型 vd_shan[0]。"""
        if self.reach_accepted[pidx] or self.meld_count[pidx] >= 4:
            return 1.0
        return None   # 未定, 调用方用 vd_shan[0]

    def visible34(self):
        """每牌已见张数(34维) = net_kawa + meld + ind + 自家暗手。

        只算"模型能观测到的"部分可观测信息: 河 + 副露 + 指示 + 自家手牌。
        不包含别家暗手——模型看不到别家手牌, clamp 若用上帝视角(四家手牌)
        会把模型合理的高山存预测错误清零(实测 5s raw=2.28 被砍到 0,
        Σ 从 77 压到 62)。"""
        net_kawa = self.kawa_pool - self.called_pai
        meld = Counter()
        ks = self.tracker.ks
        if ks is None:
            return [0] * 34
        for p in range(4):
            h = ks.hands[p]
            for m in h.chis:
                for x in m: meld[tile_id(x)] += 1
            for m in h.pons:
                for x in m: meld[tile_id(x)] += 1
            for m in h.minkans: meld[tile_id(m)] += 4
            for m in h.ankans: meld[tile_id(m)] += 4
        ind = Counter(tile_id(d) for d in ks.dora_markers)
        # 只统计自家暗手(viewer), 不统计别家——clamp 用部分可观测口径
        hands = [0] * 34
        for x in ks.hands[self.viewer].menzen:
            hands[tile_id(x)] += 1
        return [net_kawa[i] + meld[i] + ind[i] + hands[i] for i in range(34)]


def clamp_mountain_exp(mountain_dist, visible34):
    """mountain_dist: (...,34,5) softmax 分布 → 越界清零归一 → 每牌期望山残。

    对每张牌 t, 物理上限 = 4 - visible[t]。把 g > 上限 的概率清零后重新归一，
    再算 E[mountain]。返回同前几维 + (...,34) 的期望山残(不越界)。

    全程 float32 矢量化, 零循环零对象创建: (...,34,5) 一口气算。
    """
    md = np.asarray(mountain_dist, dtype=np.float32)   # (...,34,5)
    vis = np.asarray(visible34, dtype=np.int32)        # (34,)
    maxg = np.clip(4 - vis, 0, 4)                       # (34,) 每牌上限
    # 构造掩码: g > maxg 的位置清零。probs[..., t, g], g 是最后一维索引。
    g_idx = np.arange(5, dtype=np.int32)                # (5,)
    mask = g_idx[None, :] > maxg[:, None]               # (34,5) True=越界
    probs = md * (~mask)                                 # (...,34,5) 清零
    s = probs.sum(axis=-1, keepdims=True)               # (...,34,1)
    s = np.maximum(s, 1e-30)                            # 防除零(float32 最小正常数≈1e-38)
    probs = probs / s                                   # 归一
    # E = Σ g·P(g); 若某牌全部越界(s≈0), 归一后 E 会偏; 退化情况由 s 兜底
    weights = np.arange(5, dtype=np.float32)            # (5,)
    exp = (probs * weights).sum(axis=-1)                # (...,34)
    return exp


def obj_rtype(s):
    """快速取 reaction JSON 的 type 字段(用 orjson, Pass 3 热路径)。"""
    try:
        return orjson.loads(s).get("type", "")
    except Exception:
        return ""

class RecordEngine:
    """Pass 1：收集所有 (obs, masks, invisible_obs)，不跑 GPU。
    占位 action 取 mask 里第一个合法动作（Rust 会校验动作合法性）。"""

    def __init__(self, real_engine: MortalEngine):
        self._real = real_engine
        self.records: list[tuple] = []  # (obs, masks, invisible_obs)

    def __getattr__(self, name):
        return getattr(self._real, name)

    def react_batch(self, obs, masks, invisible_obs=None):
        self.records.append((obs, masks, invisible_obs))
        masks_arr = np.asarray(masks)
        # mask 是布尔数组，argmax 返回第一个 True 的下标 = 第一个合法动作
        legal_actions = masks_arr.argmax(axis=-1).tolist()
        dummy_q = np.zeros(masks_arr.shape, dtype=np.float32)
        return legal_actions, dummy_q.tolist(), masks_arr.tolist(), [True] * len(obs)


class ReplayEngine:
    """Pass 3：按序返回 Pass 2 算好的真实结果（每次 react_batch 取一条完整 record）。

    一条 record 可能含 followup 假想样本（排在主样本后），Rust 端 evaluate()
    需要拿到全部样本（含 followup），才能在 get_reaction 里按 action_idx /
    followup_idx 正确索引 q_values。followup 的具体使用在 Rust 内完成，
    Python 只需原样返回整条 record，不做剥离。

    skip_encode=True 时 Rust 端 set_scene 跳过 obs 编码（仅构造占位批对齐样本数），
    大幅节省 Pass3 的 CPU 时间。
    """

    def __init__(self, real_engine: MortalEngine, results: list):
        self._real = real_engine
        self._results = results
        self._cursor = 0
        self.skip_encode = True   # Rust 端读取此属性

    def __getattr__(self, name):
        return getattr(self._real, name)

    def react_batch(self, obs, masks, invisible_obs=None):
        assert self._cursor < len(self._results), (
            f"ReplayEngine ran out of results: cursor={self._cursor}/{len(self._results)}"
        )
        result = self._results[self._cursor]
        self._cursor += 1
        return result


class MortalReviewServer:
    def __init__(self, port: int = DEFAULT_PORT):
        self.port = port
        self.device = torch.device("cuda")
        self.engines: dict[str, MortalEngine] = {}
        self.grp: GRP | None = None
        self.value_dist = None
        self.tags: dict[str, str] = {}
        self.version: int = 0

    def load_models(self) -> None:
        logging.info("Loading Mortal models...")

        # ── oracle（可选）：加载冻结快照，config [review] 控制 ──
        # 自 step 1,065,600 起 oracle 与主网解耦：训练中 oracle 头随共享 backbone 漂移，
        # 当前 checkpoint 里的 oracle_net 已不再对齐，只能读快照。
        # oracle 是共享组件：多模型共用同一份，不随主模型变。
        oracle_net = None
        self.version = config["control"].get("version", 1)
        _oracle_on = config.get("review", {}).get("enable_oracle", False)
        if _oracle_on and OracleNet is None:
            logging.warning(
                "enable_oracle=true, 但 model.py 内已无 OracleNet(合并到 policy 分支时被移除) → 跳过 oracle"
            )
        if _oracle_on and OracleNet is not None:
            oracle_state_file = config["review"].get(
                "oracle_state_file", config["control"]["state_file"]
            )
            oracle_state = torch.load(
                oracle_state_file,
                weights_only=True,
                map_location=self.device,
            )
            oracle_net = OracleNet(
                input_size=512 if self.version == 1 else 1024
            ).eval()
            oracle_net.load_state_dict(oracle_state["oracle_net"])
            logging.info(f"Oracle net loaded from {oracle_state_file}")

        # GRP 模型（用于 phi_matrix，共享一份）
        grp = GRP(**config["grp"]["network"])
        grp_state = torch.load(
            config["grp"]["state_file"],
            weights_only=True,
            map_location=self.device,
        )
        grp.load_state_dict(grp_state["model"])
        grp.to(self.device)
        self.grp = grp

        # ── 他家打点分布模型（可选，完全独立，共享一份）──
        self.value_dist = load_predictor(config, self.device)
        if self.value_dist is not None:
            logging.info("Value-dist net loaded")
        else:
            logging.info("Value-dist net disabled or checkpoint missing")

        # ── 主模型：遍历 [review.models] 加载多个 Brain+DQN ──
        models_cfg = config.get("review", {}).get("models", {})
        if not models_cfg:
            # 兼容旧配置：无 [review.models] 时回退到单模型
            models_cfg = {"Seer": config["control"]["state_file"]}
        for name, state_file in models_cfg.items():
            state = torch.load(state_file, weights_only=True, map_location=self.device)
            cfg = state["config"]
            version = cfg["control"].get("version", 1)
            num_blocks = cfg["resnet"]["num_blocks"]
            conv_channels = cfg["resnet"]["conv_channels"]

            if "tag" in state:
                tag = state["tag"]
            else:
                # 直接用模型名做 tag(Seer->Seer, SeerX->SeerX), 不再自动生成 mortal4-...架构标签
                tag = name

            # ── 兼容两种模型: 老(DQN + BatchNorm) / 新(policy + GroupNorm) ──
            # 用存档 key 区分: 老 = current_dqn, 新 = policy_net
            is_policy = "policy_net" in state
            norm = "GN" if is_policy else "BN"
            brain = Brain(
                version=version, num_blocks=num_blocks, conv_channels=conv_channels, Norm=norm
            ).eval()
            brain.load_state_dict(state["mortal"])
            if is_policy:
                # policy 模型: head 直接输出 π → 交给引擎转成 τ·log(π),
                # 复盘侧 softmax(q/τ) 就能精确还原 π, rating / 恶手率自动兼容。
                head = CategoricalPolicy().eval()
                head.load_state_dict(state["policy_net"])
            else:
                head = DQN(version=version).eval()
                head.load_state_dict(state["current_dqn"])

            engine_kwargs = dict(
                version=version,
                is_oracle=False,
                device=self.device,
                enable_amp=False,
                enable_quick_eval=False,
                enable_rule_based_agari_guard=True,
                name=name,
                rayon_threads=config.get("review", {}).get("rayon_threads"),
            )
            if is_policy:
                # 与 seer-mortal 主程序里的 temperature = 0.1 保持一致
                engine_kwargs["review_policy_tau"] = 0.1
            # 合并后的引擎不再接受 oracle_net → 只在引擎支持时才传(老分支兼容)
            if "oracle_net" in inspect.signature(MortalEngine.__init__).parameters:
                engine_kwargs["oracle_net"] = oracle_net

            engine = MortalEngine(brain, head, **engine_kwargs)
            self.engines[name] = engine
            self.tags[name] = tag
            logging.info(f"Model [{name}] loaded from {state_file} (tag={tag})")

        logging.info(f"Mortal models loaded. engines={list(self.engines.keys())}")

        # 预热：对每个 engine 用假数据跑一次典型 batch 形状，让 cuDNN 提前完成卷积算法调优
        try:
            from libriichi.consts import obs_shape as OBS_SHAPE

            logging.info("Warming up inference...")
            shape = OBS_SHAPE(self.version)            # 不拆包，整体当形状用
            fake_obs = [np.zeros(shape, dtype=np.float32)] * BATCH_CHUNK
            fake_masks = [np.ones((46,), dtype=bool)] * BATCH_CHUNK
            for engine in self.engines.values():
                engine.react_batch(fake_obs, fake_masks, None)
                if getattr(engine, 'oracle_log', None) is not None:
                    engine.oracle_log.clear()   # 清掉预热产生的 oracle 垃圾数据
            logging.info(f"Warm-up done ({len(self.engines)} engines, obs shape: {shape})")
        except Exception:
            logging.warning(f"Warm-up failed (non-fatal):\n{traceback.format_exc()}")

    def handle(self, conn) -> None:
        conn.settimeout(120)
        """处理一次牌谱审查会话。"""
        t_total = time.time()

        with (
            conn.makefile("r", buffering=65536, encoding="utf-8") as rf,
            conn.makefile("w", buffering=65536, encoding="utf-8") as wf,
        ):
            line = rf.readline()
            if not line:
                return
            # 协议第一行: "player_id [model]"。空格分隔，缺省 model 用 Seer（向后兼容旧客户端）。
            parts = line.strip().split()
            player_id = int(parts[0])
            if player_id not in range(4):
                raise ValueError(f"Invalid player_id: {player_id}")
            model = parts[1] if len(parts) > 1 else "Seer"
            if model not in self.engines:
                raise ValueError(f"Unknown model: {model} (available: {list(self.engines.keys())})")
            engine = self.engines[model]

            all_lines = [ln.strip() for ln in rf if ln.strip()]

            # ============ Pass 1: 录制 ============
            # 插桩: init = Bot/引擎创建, loop = 逐事件 react + obs 编码
            t_init0 = time.time()
            record_engine = RecordEngine(engine)
            # record_mode：Rust 端 set_scene 照常异步编码，但 get_reaction 不 evaluate、
            # 不等每决策点的编码完成（省掉串行 wait）；Pass 1 结束 flush_encode 一次性等完。
            record_engine.record_mode = os.environ.get("RECORD_MODE", "1") == "1"
            bot = Bot(record_engine, player_id)
            t_init = time.time() - t_init0
            t_loop0 = time.time()
            for line in all_lines:
                bot.react(line)
            t_loop = time.time() - t_loop0
            if record_engine.record_mode:
                # flush_encode：等所有编码任务并行完成，返回按 record 分组的 (obs, masks, inv)
                recs = bot.flush_encode()
            else:
                recs = record_engine.records
            t_record = t_init + t_loop   # Pass 1 录制耗时(注意: 与 Pass 3 的 t_post 无关)

            # ============ Pass 2: 批量推理 ============
            t2 = time.time()   # t_gpu 起点
            # GPU 全局锁：Mortal 推理 + value_dist + oracle_log 共享引擎状态,
            # 多连接并发时必须串行(clear 也必须在锁内, 防串谱)。
            with (_GPU_LOCK if not _NO_GPU_LOCK else nullcontext()):
                engine.oracle_log.clear()   # 每连接清零，防止串谱

                # 关键：暗杠/加杠决策点一次 react_batch 带 2 个样本（kan 候选 + 主决策），
                # 不能假设 batch=1。先记录每条 record 的样本数，展平后推理，再按原样分组。
                counts = [len(r[0]) for r in recs]
                all_obs = [s for r in recs for s in r[0]]
                all_masks = [s for r in recs for s in r[1]]
                # flush_encode 无 invisible 时返回空列表；空转 None 避免 np.stack([])
                all_inv = (
                    [s for r in recs for s in r[2]]
                    if recs and recs[0][2] else None
                )

                # 逐样本结果
                sample_actions, sample_q, sample_masks, sample_greedy = [], [], [], []
                for i in range(0, len(all_obs), BATCH_CHUNK):
                    actions, q_out, masks_out, is_greedy = engine.react_batch(
                        all_obs[i : i + BATCH_CHUNK],
                        all_masks[i : i + BATCH_CHUNK],
                        all_inv[i : i + BATCH_CHUNK] if all_inv is not None else None,
                    )
                    sample_actions.extend(actions)
                    sample_q.extend(q_out)
                    sample_masks.extend(masks_out)
                    sample_greedy.extend(is_greedy)

                # 按每条 record 的原始样本数重新分组（kan 决策还原成 2 个一组）
                flat_results = []
                pos = 0
                for c in counts:
                    flat_results.append((
                        sample_actions[pos : pos + c],
                        sample_q[pos : pos + c],
                        sample_masks[pos : pos + c],
                        sample_greedy[pos : pos + c],
                    ))
                    pos += c
                assert pos == len(sample_actions), "推理结果与录制样本数不一致"

                # oracle 侧信道：同样按 record 分组（oracle_flat 与 sample_* 逐样本对齐）
                oracle_groups = []
                oracle_flat = engine.oracle_log
                if oracle_flat:
                    pos = 0
                    for c in counts:
                        oracle_groups.append(oracle_flat[pos : pos + c])
                        pos += c
                    assert pos == len(oracle_flat), "oracle 数据与录制样本数不一致"

                # ── 他家读牌多任务模型：复用同一份 all_obs 跑一次前向（每决策点一次），按 counts 分组 ──
                value_groups = []
                if self.value_dist is not None and all_obs:
                    pred = self.value_dist.predict(all_obs, chunk=BATCH_CHUNK)  # dict of (N,...)
                    vg, st, dr = pred["value"], pred["somete"], pred["dora"]
                    wg = pred["wait"]; sh = pred["shan"]; mg = pred["mountain"]
                    pos = 0
                    for c in counts:
                        group = []
                        for j in range(c):
                            idx = pos + j
                            per_player = []
                            for k in range(3):
                                # 待牌概率全量存(Pass 3 注入时乘听牌概率算铳率并过滤,
                                # 因 pTenpai 需决策点时的立直/副露状态, Pass 2 未知)
                                wait_full = [round(float(p), 3) for p in wg[idx, k]]
                                entry = {
                                    "value_ge": [round(float(x), 3) for x in vg[idx, k]],
                                    "somete": [round(float(st[idx, k, 0]), 3),
                                               round(float(st[idx, k, 1]), 3),
                                               round(float(st[idx, k, 2]), 3),
                                               round(float(st[idx, k, 3]), 3)],
                                    "vd_dora": round(float(dr[idx, k]), 3),
                                    "vd_shan": [round(float(x), 3) for x in sh[idx, k]],
                                    "wait_full": wait_full,   # 34 待牌概率(临时, Pass3 算铳率)
                                }
                                # 山残: 只第一家 k==0 存原始 34×5 分布(公共信息三家相同,
                                # 去重只挂第一家; 前端只读 opponents[0].mountain)。
                                # Pass 3 注入时按可见牌做越界覆盖(存原始而非期望,
                                # 因为覆盖需在知道该决策点 visible 后做)。
                                if k == 0:
                                    entry["mountain_raw"] = [
                                        [round(float(mg[idx, t, g]), 6) for g in range(5)]
                                        for t in range(34)
                                    ]
                                per_player.append(entry)
                            group.append(per_player)
                        value_groups.append(group)
                        pos += c
                    assert pos == len(vg), "value_dist 数据与录制样本数不一致"

            n_chunks = (len(all_obs) + BATCH_CHUNK - 1) // BATCH_CHUNK if all_obs else 0
            t_pass2 = time.time() - t2
            rate = len(all_obs) / t_pass2 if t_pass2 > 0 else 0
            print(
                f"[mortal-batch] Pass 2: {len(recs)} calls ({len(all_obs)} samples) "
                f"→ {n_chunks} chunks in {t_pass2:.2f}s ({rate:.0f} samples/s)",
                flush=True,
            )

            # ============ Pass 3: 重放 ============
            t_post0 = time.time()
            replay_engine = ReplayEngine(engine, flat_results)
            bot = Bot(replay_engine, player_id)
            vt = VisibleTracker(player_id)   # 同步维护可见牌(与提取管线同口径)
            t_py = 0.0
            for line in all_lines:
                before = replay_engine._cursor
                # 喂 VisibleTracker 维护当前可见状态(在 bot.react 前, 反映该决策点)
                vt.update(orjson.loads(line))
                reaction = bot.react(line)
                if reaction is None:
                    wf.write('{"type":"none","meta":{"mask_bits":0}}\n')
                    continue
                # 消耗了 record 的反应 → 把该 record 的 oracle 组注入 meta
                t_py0 = time.time()
                # 跳过 none 决策点(主视角无实际动作: 摸牌/过/别家和): value_dist 三家
                # 读牌在这些时点前端不需要, 注入是纯浪费(GPU/CPU/JSON体积)。
                if replay_engine._cursor > before:
                    _rtype = obj_rtype(reaction)
                    _is_none = (_rtype == "none")
                else:
                    _rtype = None
                    _is_none = False
                # value_dist 独立可替代 oracle：无 oracle_net 时直接把 value_dist 组
                # 作为 meta.oracle 下发(结构同为 [样本][3家]dict)，前端不感知数据源。
                if value_groups and replay_engine._cursor > before and not _is_none:
                    obj = orjson.loads(reaction)
                    meta = obj.setdefault("meta", {})
                    vgroup = value_groups[before]          # [样本][3家]dict
                    # 越界覆盖层: 只第一家 k==0 有 mountain_raw(Pass 2 去重), 这里只算/写第一家。
                    # mountain 是公共牌山信息(三家相同), 只挂第一家省 JSON 体积;
                    # 前端只读 opponents[0].mountain。
                    vis = vt.visible34()
                    raws = []          # (M,34,5)
                    raws_flat = []     # 每家的位置映射
                    for sample_idx, per_player in enumerate(vgroup):
                        for per in per_player:
                            if "mountain_raw" in per:
                                raws.append(np.asarray(per["mountain_raw"], dtype=np.float32))
                                raws_flat.append((sample_idx, per))
                    if raws:
                        batch = np.stack(raws)                     # (M,34,5)
                        exps = clamp_mountain_exp(batch, vis)      # (M,34)
                        for (sample_idx, per), exp_row in zip(raws_flat, exps):
                            # 一维全量 34 数组(1位小数), 0 也发(前端缺失显示 0 兜底)
                            per["mountain"] = [round(float(x), 1) for x in exp_row]
                            del per["mountain_raw"]
                    # 铳率 = 待牌 × 听牌(服务端算, 前端直接消费; 只保留 ≥0.5% 省传输)
                    # 听牌概率: 该家立直/四副露→1, 否则 vd_shan[0]
                    for sample_idx, per_player in enumerate(vgroup):
                        for k, per in enumerate(per_player):
                            if "wait_full" not in per:
                                continue
                            abs_p = (player_id + 1 + k) % 4
                            pten = vt.p_tenpai(abs_p)
                            if pten is None:
                                vds = per.get("vd_shan") or [0] * 5
                                pten = vds[0] if vds[0] is not None else 1.0
                            wait_sparse = []
                            for t, p in enumerate(per["wait_full"]):
                                ron = p * pten
                                if ron >= 0.005:   # ≥0.5% 才发(低于不计算不传输)
                                    wait_sparse.append([t, round(ron, 3)])
                            per["vd_wait"] = wait_sparse
                            del per["wait_full"]
                    oracle = meta.get("oracle")
                    if oracle and len(oracle) == len(vgroup):
                        for sample_idx, per_player in enumerate(vgroup):
                            for k in range(3):
                                oracle[sample_idx][k].update(per_player[k])
                    else:
                        meta["oracle"] = vgroup   # value_dist 独用时直接作为 oracle
                    reaction = orjson.dumps(obj).decode()   # 快速序列化
                elif oracle_groups and replay_engine._cursor > before:
                    obj = orjson.loads(reaction)
                    obj.setdefault("meta", {})["oracle"] = oracle_groups[before]
                    reaction = orjson.dumps(obj).decode()
                wf.write(reaction + "\n")
                t_py += time.time() - t_py0
            t_post = time.time() - t_post0
            t_rust = t_post - t_py
            wf.flush()

            # ============ 插桩报告 ============
            print(
                f"[perf] init={t_init*1000:.0f}ms loop={t_loop*1000:.0f}ms "
                f"t_gpu={t_pass2*1000:.0f}ms t_post={t_post*1000:.0f}ms "
                f"(t_py={t_py*1000:.0f}ms t_rust={t_rust*1000:.0f}ms) "
                f"total={time.time()-t_total:.2f}s (n_recs={len(recs)} n_obs={len(all_obs)})",
                flush=True,
            )

            # ============ Pass 4: GRP ============
            ins = Grp.load_log("\n".join(all_lines))
            feature = ins.take_feature()
            seq = [
                torch.as_tensor(feature[: idx + 1], device=self.device)
                for idx in range(len(feature))
            ]
            with (_GPU_LOCK if not _NO_GPU_LOCK else nullcontext()):
                with torch.inference_mode():
                    logits = self.grp(seq)
                matrix = self.grp.calc_matrix(logits)

            wf.write(json.dumps({
                "model_tag": self.tags[model],
                "phi_matrix": matrix.tolist(),
                "has_oracle": bool(oracle_groups),
                "has_value_ge": bool(value_groups),
            }, ensure_ascii=False) + "\n")
            wf.flush()

        print(
            f"[mortal-server] {len(recs)} calls, {len(feature)} kyokus, "
            f"{time.time() - t_total:.2f}s",
            flush=True,
        )

    def _handle_safe(self, conn, addr) -> None:
        """每连接一线程的处理入口：信号量限并发 + 异常隔离。"""
        try:
            with _CONN_SEM:
                self.handle(conn)
        except Exception:
            logging.error(f"Error handling {addr}:\n{traceback.format_exc()}")
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def serve_forever(self) -> None:
        """启动 TCP 监听，每连接一线程(CPU 段并行)，GPU 推理由 _GPU_LOCK 串行。"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", self.port))
            sock.listen(16)   # 提高 backlog, 允许更多并发连接排队
            logging.info(f"Mortal review server listening on 127.0.0.1:{self.port}")

            while True:
                conn, addr = sock.accept()
                threading.Thread(
                    target=self._handle_safe, args=(conn, addr), daemon=True
                ).start()


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    server = MortalReviewServer(port=port)
    server.load_models()
    server.serve_forever()


if __name__ == "__main__":
    main()