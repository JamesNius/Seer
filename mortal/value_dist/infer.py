"""复盘服务器推理助手: 加载八头读牌模型, 对一批 obs 跑一次前向。

与 Mortal 完全独立: 复用同一份 obs 张量, 但不复用 Mortal 的 phi/权重。
每决策点仅一次前向(与 Mortal 前向并行, 不按候选动作重复)。
输出八头概率: shan/wait/dora/somete/mountain/value/can_call/will_call。
"""
import os
import numpy as np
import torch

from value_dist.model import ValueDistNet, HEADS, THRESHOLDS


class ValueDistPredictor:
    def __init__(self, ckpt_path, device):
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        # head 注册表: 新 checkpoint 存 heads; 旧的用默认 HEADS(向后兼容)
        heads = ck.get("heads") or [dict(h) for h in HEADS]
        self.model = ValueDistNet(
            ck.get("conv", 96), ck.get("blocks", 20), heads=heads
        ).to(device).eval()
        try:
            self.model.load_state_dict(ck["model"])
        except RuntimeError:
            # 旧 checkpoint 键名({name}_head.*) → 新结构(head_modules.{name}.*)
            self.model.load_legacy_state_dict(ck["model"])
        # 每头独立温度: {"shan": T1, "value": T2, ...}, 缺省 None=1.0
        self.temps = {
            h["name"]: ck["head_temps"].get(h["name"], 1.0)
            for h in heads
        } if ck.get("head_temps") else {}
        # 兼容旧字段 temperature(旧版单一 value 温度)
        if not self.temps and ck.get("temperature") is not None:
            self.temps = {"value": ck["temperature"]}
        # 校准参数(训练后校准流水线): calibration.temperature 覆盖 head_temps
        self.calibration = ck.get("calibration")
        if self.calibration and self.calibration.get("temperature"):
            self.temps = dict(self.calibration["temperature"])
        self.somete_isotonic = (self.calibration or {}).get("somete_isotonic")
        self.device = device
        self.thresholds = ck.get("thresholds", THRESHOLDS)

    def _apply_calib(self, probs, name):
        """应用温度(已在 predict_probs 做) + somete isotonic 后置校准。"""
        if name != "somete" or not self.somete_isotonic:
            return probs
        out = probs.copy()
        for ch in range(3):
            spec = self.somete_isotonic.get(str(ch))
            if spec is None:
                continue
            if spec[0] == "platt":
                _, a, b = spec
                p = out[..., ch]
                z = np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1 - 1e-9))
                out[..., ch] = 1 / (1 + np.exp(-(a * z + b)))
            else:
                x, y = spec
                out[..., ch] = np.interp(out[..., ch], np.asarray(x), np.asarray(y))
        return out

    @torch.no_grad()
    def predict(self, obs_list, chunk=128):
        """obs_list: list of (1012,34) ndarray。

        返回 dict, 每键 (N, ...) float32, 键 = checkpoint 的 head 注册表名。
        注册表含多少头就返回多少键(加头自动识别)。
        """
        heads = self.model.heads
        names = [h["name"] for h in heads]
        acc = {n: [] for n in names}
        for i in range(0, len(obs_list), chunk):
            b = torch.as_tensor(
                np.stack(obs_list[i:i+chunk], axis=0), dtype=torch.float32
            ).to(self.device)
            probs = self.model.predict_probs(b, temps=self.temps)   # dict[name] -> (B,*shape)
            # 每 chunk 一次 .cpu(): 合并所有头为 (B, Σflat) → 一次 GPU→CPU 同步, 再按各头 flat 长度拆分
            flat_lens = []
            cat_list = []
            for n in names:
                flat = probs[n].reshape(probs[n].shape[0], -1)
                flat_lens.append(flat.shape[1])
                cat_list.append(flat)
            merged = torch.cat(cat_list, dim=-1).cpu().numpy()   # (B, Σflat) 单次 .cpu
            offset = 0
            for idx_n, n in enumerate(names):
                L = flat_lens[idx_n]
                p = merged[:, offset:offset+L].reshape(probs[n].shape).copy()
                offset += L
                if n == "somete":
                    p = self._apply_calib(p, n)   # isotonic 后置校准
                if n == "dora":   # dora 头转期望枚数(与旧版一致)
                    p = p * np.arange(7, dtype=np.float32)
                    p = p.sum(-1, keepdims=False)
                acc[n].append(p)
        out = {}
        for n in names:
            if acc[n]:
                out[n] = np.concatenate(acc[n])
            else:
                shape = (0,) + tuple(self.model.head_shapes[n])
                out[n] = np.zeros(shape, np.float32)
        return out


def load_predictor(config, device):
    """按 config 加载; 关闭或文件缺失时返回 None(优雅降级, 不影响主流程)。"""
    review = config.get("review", {})
    if not review.get("enable_value_dist", True):
        return None
    path = review.get("value_dist_file", "checkpoints/value_dist.pth")
    if not os.path.exists(path):
        return None
    return ValueDistPredictor(path, device)
