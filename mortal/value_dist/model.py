"""独立的他家读牌多任务模型 v2。完全独立于 Mortal policy net。

共享 ResNet backbone(复用 model.py 结构, 小规模化), 多头:
  每个头由 HEADS 注册表描述(名字/输出形状/激活), ValueDistNet 按注册表
  动态构建。加头只需在 HEADS 加一项, 训练/推理端零改动自动识别。

当前六头:
  - shan:     3家×5档     P(向听=0/1/2/3/4+)     [softmax, 全程]
  - wait:     3家×34      P(该家在听牌且荣和此牌)  [mask: shan>0]
  - dora:     3家×7档     P(该家持 dora 张数=0..6+) [softmax, 全程]
  - somete:   3家×4       P(染 万/筒/索/字)        [sigmoid, 全程]
  - mountain: 34×5        P(牌山剩余张数=0..4)     [softmax per tile, 全程]
  - value:    3家×6阈值   P(和了打点>=t_k)         [mask: shan>1]

输入 = encode_obs(4) 的 (1012,34) obs, 一次前向同时出全部头。
"""
import torch
from torch import nn

from model import ResNet  # 复用 Mortal 的 backbone 结构(不修改它)
from libriichi.consts import obs_shape

THRESHOLDS = [2000, 3900, 7700, 11600, 18000, 36000]
N_HEAD = 3          # 对手数(下家/对家/上家)
N_THRESH = len(THRESHOLDS)
N_TILE = 34
N_CALL = 2          # 碰/杠, 吃
N_DORA_BUCKET = 7   # dora 张数分桶 0..5,6+
N_SHAN = 5          # 向听 0/1/2/3/4+
N_SOMETE = 4        # 万/筒/索/字
N_MOUNTAIN = 5      # 山残 0..4

# ── head 注册表 ─────────────────────────────────────────────────────────────
# 每个头: name(变量名) / shape(输出形状, 排除 batch) / kind(激活)。
# ValueDistNet.forward 按此顺序返回 tuple; 加头只改这里。
# temperature: 校准温度, None=未校准(1.0)。按头独立, 存 checkpoint。
HEADS = [
    {"name": "shan",     "shape": (N_HEAD, N_SHAN),       "kind": "softmax"},
    {"name": "wait",     "shape": (N_HEAD, N_TILE),       "kind": "sigmoid"},
    {"name": "dora",     "shape": (N_HEAD, N_DORA_BUCKET), "kind": "softmax"},
    {"name": "somete",   "shape": (N_HEAD, N_SOMETE),     "kind": "sigmoid"},
    {"name": "mountain", "shape": (N_TILE, N_MOUNTAIN),   "kind": "softmax"},
    {"name": "value",    "shape": (N_HEAD, N_THRESH),     "kind": "sigmoid_clamp"},
]


def head_specs():
    """返回注册表副本(供 checkpoint 存取)。"""
    return [dict(h) for h in HEADS]


class ValueDistNet(nn.Module):
    """obs (B,1012,34) -> 多头(由 HEADS 注册表驱动)。"""

    def __init__(self, conv_channels=96, num_blocks=20, version=4, heads=None):
        super().__init__()
        self.heads = heads if heads is not None else HEADS
        in_channels = obs_shape(version)[0]
        self.backbone = ResNet(in_channels, conv_channels, num_blocks)
        d = 1024
        self.head_modules = nn.ModuleDict()
        self.head_shapes = {}
        for h in self.heads:
            name, shape = h["name"], tuple(h["shape"])
            self.head_shapes[name] = shape
            out = 1
            for s in shape:
                out *= s
            self.head_modules[name] = nn.Linear(d, out)

    def forward(self, obs):
        phi = self.backbone(obs)                                    # (B,1024)
        out = []
        for name, shape in self.head_shapes.items():
            mod = self.head_modules[name]
            out.append(mod(phi).unflatten(-1, shape))
        return tuple(out)

    @torch.no_grad()
    def predict_probs(self, obs, temps=None):
        """前向 + 概率化。返回 dict(名字 -> (B,*shape) 概率)。
        temps: 可选 dict(name -> 温度), 各头独立校准。
        """
        outs = self.forward(obs)
        temps = temps or {}
        result = {}
        for h, logits in zip(self.heads, outs):
            name, kind = h["name"], h["kind"]
            t = temps.get(name, 1.0)
            if t != 1.0:
                logits = logits / t
            if kind == "softmax":
                result[name] = torch.softmax(logits, dim=-1)
            elif kind == "sigmoid":
                result[name] = torch.sigmoid(logits)
            elif kind == "sigmoid_clamp":
                result[name] = monotone_clamp(torch.sigmoid(logits))
            else:
                result[name] = logits
        return result

    def load_legacy_state_dict(self, state_dict):
        """加载旧版(命名 {name}_head.weight) state_dict → 新结构。"""
        mapped = {}
        for k, v in state_dict.items():
            for name in self.head_shapes:
                prefix = f"{name}_head."
                if k.startswith(prefix):
                    mapped[f"head_modules.{name}." + k[len(prefix):]] = v
                    break
            else:
                mapped[k] = v
        self.load_state_dict(mapped)


def monotone_clamp(p):
    """P(>=t_k) 单调不增。"""
    return torch.cummin(p, dim=-1).values


def masked_bce_with_logits(logits, labels, mask):
    """mask (B,3) 广播到 logits 最后一维。"""
    m = mask.reshape(mask.shape[0], mask.shape[1], *([1] * (logits.dim() - 2)))
    m = m.expand_as(logits)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (loss * m).sum() / m.sum().clamp_min(1.0)


def multitask_loss(batch, weights=None, call_pos_weight=8.0, somete_pos_weight=5.0, will_pos_weight=4.0):
    """八头 masked 多任务损失。

    batch = dict(logits 八头, labels, masks...)
    各头:
      shan:   masked CE(one-hot)
      wait:   masked BCE(mask=shan==0)
      dora:   masked CE(one-hot)
      染手:   masked BCE(正例低频 pos_weight)
      山残:   masked CE per tile (mask=tile 有效)
      value:  masked BCE(mask=shan<=1)
      can_call: BCE(全样本)
      will_call: BCE(仅 can_call=true 子集, 且只在实际切的牌位置)
    """
    w = dict(shan=1.0, wait=1.0, dora=1.0, somete=1.0, mountain=1.0, value=1.0, can_call=1.0, will_call=1.0)
    if weights:
        w.update(weights)
    losses = {}

    # shan: CE
    shan = batch["shan"]; shan_lab = batch["shan_label"]
    shan_loss = -torch.log_softmax(shan, -1) * shan_lab
    losses["shan"] = shan_loss.mean()

    # wait: BCE, mask = 该家听牌(shanten==0)
    wait = batch["wait"]; wait_lab = batch["wait_label"]; wait_mask = batch["wait_mask"]
    losses["wait"] = masked_bce_with_logits(wait, wait_lab, wait_mask)

    # dora: CE (全样本有效, 无 mask)
    dora = batch["dora"]; dora_lab = batch["dora_label"]
    dora_loss = -(torch.log_softmax(dora, -1) * dora_lab)
    losses["dora"] = dora_loss.mean()

    # 染手: BCE (全样本有效, 无 mask)
    somete = batch["somete"]; somete_lab = batch["somete_label"]
    pw = torch.full_like(somete, somete_pos_weight)
    losses["somete"] = nn.functional.binary_cross_entropy_with_logits(somete, somete_lab, pos_weight=pw)

    # 山残: CE per tile
    mountain = batch["mountain"]; mountain_lab = batch["mountain_label"]
    mountain_loss = -torch.log_softmax(mountain, -1) * mountain_lab
    losses["mountain"] = mountain_loss.mean()

    # value: BCE, mask = shan<=1
    value = batch["value"]; value_lab = batch["value_label"]; value_mask = batch["value_mask"]
    losses["value"] = masked_bce_with_logits(value, value_lab, value_mask)

    total = sum(w[k] * losses[k] for k in losses if k in w)
    return total, {k: v.item() for k, v in losses.items()}
