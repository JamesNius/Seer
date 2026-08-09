"""一致性对拍: 随机 100 个 val 状态, 训练侧前向 vs infer.py 推理侧(含校准), 容差 1e-5。

用法: python -m value_dist.verify_calib_consistency
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from value_dist.model import ValueDistNet
from value_dist.infer import ValueDistPredictor
from value_dist.calibrate_pipeline import (CKPT_OUT, CAL_DIR, HEADS, MASK_HEADS,
                                           _apply_calib)

SEED = 123
N_SAMPLE = 100


def main():
    rng = np.random.RandomState(SEED)
    ck = torch.load(CKPT_OUT, map_location="cpu", weights_only=False)
    # 随机选 100 个 eval 样本(用 shard index + 片内位置)
    import glob
    shards = sorted(glob.glob(os.path.join(
        os.path.dirname(CKPT_OUT), "value_dist_data", "val_a_*.npz")))
    temps = ck["calibration"]["temperature"]
    somete = ck["calibration"]["somete_isotonic"]

    # infer.py 侧
    pred = ValueDistPredictor(CKPT_OUT, "cuda")

    # 训练侧: 用校准参数应用
    model = ValueDistNet(ck["conv"], ck["blocks"], heads=ck.get("heads")).to("cuda").eval()
    model.load_state_dict(ck["model"])

    n_checked = 0
    max_diff = 0.0
    fails = []
    for trial in range(N_SAMPLE):
        # 随机 shard + 片内 index
        si = rng.randint(len(shards))
        d = np.load(shards[si])
        n = len(d["shan_label"])
        idx = rng.randint(n)
        obs = np.asarray(d["obs"][idx], dtype=np.float32)[None]  # (1,1012,34)
        # 训练侧: 前向 + 温度 + isotonic
        with torch.no_grad():
            b = torch.from_numpy(obs).cuda()
            probs = model.predict_probs(b, temps=temps)   # dict
        train_out = {}
        for h in HEADS:
            p = probs[h].cpu().numpy()[0]
            if h == "somete":
                p = _apply_calib(p[None], "somete", temps["somete"], somete)[0]
            train_out[h] = p
        # infer.py 侧
        infer_out = pred.predict(obs, chunk=1)
        for h in HEADS:
            a = train_out[h]
            b = infer_out[h][0]
            if h == "dora":
                continue  # infer 侧 dora 转期望, 跳过或单独比
            diff = np.abs(a - b).max()
            max_diff = max(max_diff, diff)
            n_checked += 1
            if diff > 1e-5:
                fails.append((trial, h, diff))
        d.close()

    print(f"对拍 {N_SAMPLE} 样本, 检查 {n_checked} 头-样本, 最大差 {max_diff:.2e}")
    if fails:
        print(f"失败 {len(fails)} 处:")
        for t, h, diff in fails[:10]:
            print(f"  trial{t} {h} diff={diff:.2e}")
        sys.exit(1)
    print("✅ 一致性对拍通过 (容差 1e-5)")


if __name__ == "__main__":
    main()
