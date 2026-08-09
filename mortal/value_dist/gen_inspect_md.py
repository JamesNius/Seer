"""100 样本目检包: 随机 100 个 val 状态, 输出人类可读卡片 markdown。

每卡片: 局面概要(手牌/向听标签) + 六头校准后预测 + 真实标签。
用法: python -m value_dist.gen_inspect_md
"""
import glob, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from value_dist.model import ValueDistNet
from value_dist.infer import ValueDistPredictor
from value_dist.calibrate_pipeline import (CKPT_OUT, CAL_DIR, HEADS, MASK_HEADS,
                                           _apply_calib, HEAD_SHAPES)

SEED = 777
N_SAMPLE = 100
OUT_MD = os.path.join(os.path.dirname(CKPT_OUT), "inspect_100.md")

TILE_STR = [f"{i+1}m" for i in range(9)] + [f"{i+1}p" for i in range(9)] + \
           [f"{i+1}s" for i in range(9)] + list("ESWN PFC".replace(" ", ""))


def idx_to_tiles(c):
    out = []
    for i, n in enumerate(c):
        if n > 0:
            out += [TILE_STR[i]] * n
    return " ".join(out)


def main():
    rng = np.random.RandomState(SEED)
    ck = torch.load(CKPT_OUT, map_location="cpu", weights_only=False)
    temps = ck["calibration"]["temperature"]
    somete = ck["calibration"]["somete_isotonic"]
    pred = ValueDistPredictor(CKPT_OUT, "cuda")
    model = ValueDistNet(ck["conv"], ck["blocks"], heads=ck.get("heads")).to("cuda").eval()
    model.load_state_dict(ck["model"])
    shards = sorted(glob.glob(os.path.join(
        os.path.dirname(CKPT_OUT), "value_dist_data", "val_a_*.npz")))

    lines = ["# value_dist 校准后 100 样本目检", ""]
    for trial in range(N_SAMPLE):
        si = rng.randint(len(shards))
        d = np.load(shards[si])
        n = len(d["shan_label"])
        idx = rng.randint(n)
        obs = np.asarray(d["obs"][idx], dtype=np.float32)[None]
        with torch.no_grad():
            b = torch.from_numpy(obs).cuda()
            probs = model.predict_probs(b, temps=temps)
        infer_out = pred.predict(obs, chunk=1)
        lines.append(f"## 样本 {trial}  (shard {os.path.basename(shards[si])} idx {idx})")
        # 局面概要: 用 shan_label 的向听/手牌计数近似
        shan_lab = d["shan_label"][idx]
        shan_true = shan_lab.argmax(-1)  # (3,)
        lines.append(f"- 真实向听(3家): {shan_true.tolist()}")
        # 六头预测 vs 标签
        for h in HEADS:
            lab = d[f"{h}_label"][idx]
            p = infer_out[h][0]
            if h == "shan":
                ps = np.round(p, 3)
                lines.append(f"- **shan** 预测{ps.tolist()} 真值{lab.argmax(-1).tolist()}")
            elif h == "wait":
                top = np.argsort(p.reshape(-1))[::-1][:3]
                lines.append(f"- **wait** top3={top.tolist()} 待牌真值非零数={int((lab.reshape(-1)>0.5).sum())}")
            elif h == "dora":
                continue
            elif h == "somete":
                lines.append(f"- **somete** 预测{np.round(p[:3],3).tolist()} 真值{np.round(lab[:3],3).tolist()}")
            elif h == "mountain":
                exp = (p * np.arange(5)).sum(-1)  # (34,)
                lines.append(f"- **mountain** 期望残数均值={exp.mean():.3f} (34牌)")
            elif h == "value":
                lines.append(f"- **value** 预测{np.round(p.reshape(-1),3).tolist()}")
        lines.append("")
        d.close()

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"目检包已写 {OUT_MD} ({N_SAMPLE} 样本)")


if __name__ == "__main__":
    main()
