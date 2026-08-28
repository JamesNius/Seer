"""tenhou_react_server.py —— SeerX 实时何切服务（Akagi 云端推理 API 后端）

实现 Akagi 的 /v3/react 协议：
  请求  POST /v3/react  {"model": str, "player_id": int, "events": [mjai事件...]}
  响应  {"reaction": {mjai事件}, "candidates": [...], "model": str}
  健康  GET /healthz    {"status": "ok", "models": [...], "queue_depth": {}}

工作方式（当前版本）：
  每次请求新建一个 libriichi.mjai.Bot，从头重放 events（当前小局的事件流，
  Akagi 保证 < 512 事件），最后一步返回 reaction。Bot 是状态机，重放过程
  中间事件只做 state.update（极快），只有决策点才编码 obs + 推理。

  后续优化方向（见文件尾 TODO）：长驻 Bot 按「牌局」缓存，增量喂，避免
  每步重放。当前重放成本实测 ~0.6~0.8s/小局，实时可接受。

用法：
  python tenhou_react_server.py          # 默认端口 8787
  python tenhou_react_server.py --port 8787 --model SeerX
"""

import argparse
import logging
import os
import sys
import time
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# 项目根目录（mortal/ 子目录），确保能 import model/engine/config
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from model import Brain, DQN          # noqa: E402
from engine import MortalEngine       # noqa: E402
from config import config             # noqa: E402
from libriichi.mjai import Bot        # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tenhou_react")


class RecordEngine:
    """Pass1 录制引擎：collect obs/masks，不跑 GPU。占位 action 取 mask 第一个合法动作。"""

    def __init__(self, real_engine):
        self._real = real_engine
        self.record_mode = True

    def __getattr__(self, name):
        return getattr(self._real, name)

    def react_batch(self, obs, masks, invisible_obs=None):
        # record_mode 下 Rust 端不真正调这个；这里是兜底（非 record_mode 时才走）
        import numpy as np
        masks_arr = np.asarray(masks)
        legal = masks_arr.argmax(axis=-1).tolist()
        dummy_q = np.zeros(masks_arr.shape, dtype=np.float32)
        return legal, dummy_q.tolist(), masks_arr.tolist(), [True] * len(obs)


class ReplayEngine:
    """Pass3 重放引擎：只在「最后一次」react_batch 调用返回真实结果，其余占位。

    因为 Bot.react 对每个决策点都调一次 react_batch，而实时何切只需要最后一个
    决策点的真实 reaction。用 total_calls 标记总决策点数，前 total_calls-1 次
    返回占位合法动作，最后一次返回 Pass2 的真实推理结果。
    """

    def __init__(self, real_engine, results, total_calls):
        self._real = real_engine
        self._results = results       # 只有 1 条真实 record
        self._total_calls = total_calls
        self._call = 0
        self.skip_encode = False

    def __getattr__(self, name):
        return getattr(self._real, name)

    def react_batch(self, obs, masks, invisible_obs=None):
        import numpy as np
        self._call += 1
        if self._call < self._total_calls:
            # 前面的决策点：占位（取 mask 第一个合法动作）
            masks_arr = np.asarray(masks)
            legal = masks_arr.argmax(axis=-1).tolist()
            dummy_q = np.zeros(masks_arr.shape, dtype=np.float32)
            return legal, dummy_q.tolist(), masks_arr.tolist(), [True] * len(obs)
        # 最后一个决策点：返回真实结果
        return self._results[0]

# 动作索引 → 牌名。索引 0-33 是普通 34 张，34-36 是赤 5m/5p/5s。
# 顺序来自 libriichi/src/tile.rs 的 MJAI_PAI_STRINGS。
_TILES = (["1m","2m","3m","4m","5m","6m","7m","8m","9m",
           "1p","2p","3p","4p","5p","6p","7p","8p","9p",
           "1s","2s","3s","4s","5s","6s","7s","8s","9s",
           "E","S","W","N","P","F","C"])
_AKA = ["5mr", "5pr", "5sr"]   # 动作索引 34,35,36

def _label_to_action(label: int) -> str:
    """把 46 维动作索引转成 Akagi 的粗粒度 action 标签。

    0-33 切普通牌, 34-36 切赤牌, 37 立直, 38-40 吃, 41 碰,
    42 杠, 43 和, 44 流局, 45 pass。
    """
    if 0 <= label <= 33:
        return f"dahai:{_TILES[label]}"
    if 34 <= label <= 36:
        return f"dahai:{_AKA[label - 34]}"
    if label == 37:
        return "reach"
    if 38 <= label <= 40:
        return f"chi{label - 37}"   # chi1/chi2/chi3 (低/中/高)
    if label == 41:
        return "pon"
    if label == 42:
        return "kan"
    if label == 43:
        return "hora"
    if label == 44:
        return "ryukyoku"
    if label == 45:
        return "none"
    return f"unknown:{label}"


class ReactRequest(BaseModel):
    model: Optional[str] = None
    player_id: int = 0
    events: List[dict]


class ReactResponse(BaseModel):
    reaction: Optional[dict] = None
    candidates: List[dict] = []
    model: Optional[str] = None


class TenhouReactServer:
    def __init__(self, model_name: str = "SeerX"):
        self.device = torch.device(config["control"]["device"])
        self.model_name = model_name
        self.engine = self._load_engine(model_name)
        # 长驻 Bot（增量）：跨请求保持状态，每手牌只喂新增事件
        self._bot = None        # 当前小局的 Bot
        self._bot_key = None    # 当前小局指纹（用于检测换小局）
        self._bot_n = 0         # Bot 已处理的事件数
        log.info(f"SeerX 引擎加载完成: {model_name} (device={self.device})")

    def _load_engine(self, model_name: str) -> MortalEngine:
        """加载指定模型的 Brain+DQN，返回 MortalEngine。"""
        models_cfg = config.get("review", {}).get("models", {})
        state_file = models_cfg.get(model_name)
        if not state_file:
            # 兼容：没有 [review.models] 时回退到 control.state_file
            state_file = config["control"]["state_file"]
            log.warning(f"模型 {model_name} 不在 [review.models]，回退 {state_file}")

        t0 = time.time()
        state = torch.load(state_file, weights_only=True, map_location=self.device)
        cfg = state["config"]
        version = cfg["control"].get("version", 1)
        num_blocks = cfg["resnet"]["num_blocks"]
        conv_channels = cfg["resnet"]["conv_channels"]

        brain = Brain(version=version, num_blocks=num_blocks,
                      conv_channels=conv_channels).eval()
        dqn = DQN(version=version).eval()
        brain.load_state_dict(state["mortal"])
        dqn.load_state_dict(state["current_dqn"])

        engine = MortalEngine(
            brain, dqn, oracle_net=None, version=version, is_oracle=False,
            device=self.device, enable_amp=False, enable_quick_eval=False,
            enable_rule_based_agari_guard=True, name=model_name,
        )
        log.info(f"模型 {model_name} 加载耗时 {time.time()-t0:.2f}s "
                 f"(version={version}, blocks={num_blocks}, ch={conv_channels})")
        return engine

    def react(self, player_id: int, events: List[dict]) -> Optional[dict]:
        """增量推理：长驻 Bot 跨请求保持状态，每手牌只喂新增事件。

        与完整重放结果一致（已验证 51/51），但每手牌只喂 1~3 个新事件，快。
        换小局（events 变短/指纹变）时重建 Bot。
        """
        import json as _json

        key = self._kyoku_key(player_id, events)

        # 判断能否增量：同一小局，且 events 是「已处理前缀 + 新增尾部」
        if self._bot is not None and key == self._bot_key and len(events) >= self._bot_n:
            bot = self._bot
            new_events = events[self._bot_n:]
            reaction_str = None
            try:
                for ev in new_events:
                    line = _json.dumps(ev, ensure_ascii=False)
                    r = bot.react(line)
                    if r is not None:
                        reaction_str = r
                self._bot_n = len(events)
            except Exception as e:
                log.warning(f"增量喂异常(重建Bot兜底): {e}")
                self._bot = None
                self._bot_key = None
                self._bot_n = 0
                return None
        else:
            # 重放：新建 Bot，喂完整 events
            bot = Bot(self.engine, player_id)
            reaction_str = None
            try:
                for ev in events:
                    line = _json.dumps(ev, ensure_ascii=False)
                    r = bot.react(line)
                    if r is not None:
                        reaction_str = r
                self._bot = bot
                self._bot_key = key
                self._bot_n = len(events)
            except Exception as e:
                log.warning(f"完整重放异常(返回None兜底): {e}")
                self._bot = None
                self._bot_key = None
                self._bot_n = 0
                return None

        if reaction_str is None:
            return None
        reaction = _json.loads(reaction_str)

        # 对齐复盘：只有 1 个合法动作(如立直后摸切)的决策点，复盘端用
        # `mask_bits.count_ones() <= 1` 跳过，这里同样跳过返回 None。
        meta = reaction.get("meta") or {}
        mask_bits = meta.get("mask_bits")
        if mask_bits is not None and mask_bits.bit_count() <= 1:
            return None

        # 修正 tsumogiri: libriichi Bot 只按"切牌==刚摸牌"判摸切, 没考虑
        # 手牌里(不含副露)是否也有同张。若手里还有同张, 切这张可能是手切(打手里的),
        # 应标 false。赤5 天然保护: 赤5 只有一张, 摸赤5打赤5 手牌无同张 → 仍摸切。
        self._fix_tsumogiri(player_id, events, reaction)
        return reaction

    def _fix_tsumogiri(self, player_id: int, events: List[dict], reaction: dict) -> None:
        """重建自家手牌(不含副露), 修正 dahai 的 tsumogiri。原地修改 reaction。"""
        if reaction.get("type") != "dahai":
            return
        pai = reaction.get("pai")
        if not pai:
            return
        # 从 events 重建自家手牌(不含副露) + 最近一次自家摸的牌
        hand: list = []          # 当前手牌(不含副露, 不含刚摸的)
        last_tsumo = None        # 最近一次自家摸的牌
        for ev in events:
            t = ev.get("type")
            actor = ev.get("actor")
            if t == "start_kyoku":
                tehais = ev.get("tehais") or []
                if 0 <= player_id < len(tehais):
                    hand = list(tehais[player_id])
                last_tsumo = None
            elif actor == player_id:
                if t == "tsumo":
                    last_tsumo = ev.get("pai")
                    # 刚摸的牌暂不进 hand, 单独用 last_tsumo 判断来源
                elif t == "dahai":
                    d = ev.get("pai")
                    if d in hand:
                        hand.remove(d)
                    last_tsumo = None
                elif t in ("pon", "chi", "daiminkan", "ankan", "kakan"):
                    # 副露: consumed + pai 离开手牌
                    consumed = ev.get("consumed") or []
                    for c in consumed:
                        if c in hand:
                            hand.remove(c)
                    p = ev.get("pai")
                    if p in hand:
                        hand.remove(p)
                    last_tsumo = None
        # 判定: 切的牌 == 刚摸的牌, 且手牌里(不含刚摸)还有同张 → 可能是手切(打手里的)
        if last_tsumo is not None and pai == last_tsumo:
            if pai in hand:
                reaction["tsumogiri"] = False
            else:
                reaction["tsumogiri"] = True

    @staticmethod
    def _kyoku_key(player_id: int, events: List[dict]) -> tuple:
        """用 start_kyoku 的内容做小局指纹（同一小局固定）。"""
        sk = None
        for e in events:
            if e.get("type") == "start_kyoku":
                sk = e
                break
        if sk is None:
            return (player_id, None, None, None)
        return (player_id, sk.get("kyoku"), sk.get("honba"), sk.get("oya"))

    @staticmethod
    def _build_candidates(reaction: dict, topk: int = 10, temperature: float = 1.0) -> List[dict]:
        """从 reaction.meta 的 q_values + mask_bits 算 top-k 候选（softmax 概率）。

        q_values 按 mask_bits 升序对应合法动作；对 q/T 做 softmax 得概率，
        按概率降序取 top-k，映射成粗粒度 action 标签。

        temperature: 保持 1.0，如实反映模型 Q 值分布的确定性。SeerX 的 Q 值
        差距极大(最优≈0, 次优≈-12)，T=1 时第一选概率接近 100%，这正符合
        "模型很确定"的直觉。之前误用 T=10 摊平，导致第一选只有 23%(失真)。
        """
        meta = reaction.get("meta") or {}
        q_values = meta.get("q_values")
        mask_bits = meta.get("mask_bits")
        if not q_values or mask_bits is None:
            # 无 meta 时退化为单候选
            return [{"action": reaction.get("type", "none"), "prob": 1.0}]

        # 还原合法动作索引（mask_bits 的 bit 位 = 动作 label）
        labels = [i for i in range(64) if (mask_bits >> i) & 1]
        if len(labels) != len(q_values):
            # 数量对不上（罕见），退化为单候选
            return [{"action": reaction.get("type", "none"), "prob": 1.0}]

        # softmax with temperature
        import math
        qs = [float(q) / temperature for q in q_values]
        mx = max(qs)
        exps = [math.exp(q - mx) for q in qs]
        s = sum(exps)
        probs = [e / s for e in exps]

        # 按概率降序排序
        pairs = sorted(zip(labels, probs), key=lambda x: -x[1])
        out = []
        for label, prob in pairs[:topk]:
            action = _label_to_action(label)
            out.append({"action": action, "prob": round(prob, 6)})
        return out


app = FastAPI(title="tenhou-react-server")
_server: TenhouReactServer = None


@app.on_event("startup")
def _startup():
    global _server
    model = os.environ.get("TENHOU_MODEL", "SeerX")
    _server = TenhouReactServer(model_name=model)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "models": [_server.model_name], "queue_depth": {}}


@app.get("/v3/key")
def key_status():
    """Akagi 会调此接口验证 key。返回一个「无限额」的固定状态。"""
    return {
        "plan": "local",
        "expires_at": "2999-12-31",
        "usage_today": 0,
        "rpd": 0,
        "rpm": 9999.0,
        "topk": 10,
    }


@app.get("/v3/models")
def models():
    """Akagi 查询可用模型列表。"""
    return {
        "models": [
            {"id": _server.model_name, "game": "tenhou", "desc": "SeerX local"},
        ],
    }


@app.post("/v3/react", response_model=ReactResponse)
def react(req: ReactRequest):
    if _server is None:
        raise HTTPException(status_code=503, detail="server not ready")
    t0 = time.time()
    # (调试已完成: 完整 dump events 到文件的功能已注释, 需要时恢复)
    reaction = _server.react(req.player_id, req.events)
    elapsed = (time.time() - t0) * 1000
    if reaction is None:
        log.info(f"react player_id={req.player_id} events={len(req.events)} "
                 f"-> None ({elapsed:.1f}ms)")
        return ReactResponse(reaction=None, model=_server.model_name)
    candidates = _server._build_candidates(reaction)
    # 打日志：第一选动作 + 前 3 名候选 + 概率，方便验证 Akagi 是否用了本 bot 输出
    first = candidates[0] if candidates else {}
    top3 = ", ".join(f"{c['action']}={c['prob']*100:.1f}%" for c in candidates[:3])
    log.info(f"react player_id={req.player_id} events={len(req.events)} "
             f"-> 第一选 {first.get('action','?')} | top3: {top3} ({elapsed:.1f}ms)")
    return ReactResponse(reaction=reaction, candidates=candidates,
                         model=_server.model_name)


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    if args.model:
        os.environ["TENHOU_MODEL"] = args.model
    uvicorn.run(app, host="0.0.0.0", port=args.port)
