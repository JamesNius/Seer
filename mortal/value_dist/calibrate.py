"""温度缩放(Platt) + 可靠性报告。

用法:
  python -m value_dist.calibrate --data checkpoints/value_dist_data \
      --ckpt checkpoints/value_dist.pth --reliability
"""
import argparse, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from value_dist.model import ValueDistNet, monotone_clamp, THRESHOLDS


@torch.no_grad()
def collect_logits(model, obs, device, bs=2048):
    """收集 value 头 logits (B,3,6)。"""
    out = []
    for i in range(0, len(obs), bs):
        b = torch.from_numpy(np.asarray(obs[i:i+bs], dtype=np.float32)).to(device)
        value, _, _, _ = model(b)
        out.append(value.cpu().numpy())
    return np.concatenate(out)


def fit_temperature(logits, labels, mask):
    """每个 (head,thresh) 独立拟合标量温度 T, 最小化 masked BCE。返回 (3,6) 温度。"""
    temps = np.ones((3, 6), np.float32)
    m = mask.astype(bool)                      # (N,3)
    for h in range(3):
        sel = m[:, h]
        if sel.sum() < 50:
            continue
        z = torch.from_numpy(logits[sel, h, :])          # (M,6)
        y = torch.from_numpy(labels[sel, h, :])
        log_t = torch.zeros(6, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], max_iter=50, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                z / log_t.exp(), y)
            loss.backward()
            return loss
        opt.step(closure)
        temps[h] = log_t.detach().exp().numpy()
    return temps


def reliability_table(probs, labels, mask, n_bins=10):
    """每阈值 n_bins 分箱: 预测均值 vs 实测频率。返回 {thresh: (pred, freq, count)}"""
    rep = {}
    m = mask.astype(bool)
    for k, th in enumerate(THRESHOLDS):
        p = probs[:, :, k][m]                  # 所有被选中头的该阈值预测
        y = labels[:, :, k][m]
        bins = np.linspace(0, 1, n_bins + 1)
        idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
        pred = np.zeros(n_bins); freq = np.zeros(n_bins); cnt = np.zeros(n_bins)
        for b in range(n_bins):
            s = idx == b
            if s.sum():
                pred[b] = p[s].mean(); freq[b] = y[s].mean(); cnt[b] = s.sum()
        rep[th] = (pred, freq, cnt)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="checkpoints/value_dist_data")
    ap.add_argument("--ckpt", default="checkpoints/value_dist.pth")
    ap.add_argument("--conv", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=20)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    model = ValueDistNet(ck.get("conv", args.conv), ck.get("blocks", args.blocks)).to(device)
    model.load_state_dict(ck["model"]); model.eval()

    d = os.path.join(args.data, "val")
    obs = np.load(f"{d}_obs.npy", mmap_mode="r")
    labels = np.load(f"{d}_labels.npy")
    mask = np.load(f"{d}_mask.npy")
    turn = np.load(f"{d}_turn.npy")

    logits = collect_logits(model, obs, device)
    temps = fit_temperature(logits, labels, mask)
    print("温度参数 (3头×6阈值):")
    print(np.round(temps, 3))

    # 校准后概率
    cal_probs = 1 / (1 + np.exp(-logits / temps[None]))
    cal_probs = np.minimum.accumulate(cal_probs, axis=-1)   # 单调 clamp
    np.save(os.path.join(args.data, "val_cal_probs.npy"), cal_probs)

    # 保存温度进 checkpoint (存为嵌套 list, 保证 weights_only=True 可加载)
    ck["temperature"] = temps.tolist()
    torch.save(ck, args.ckpt)

    # ---- 可靠性报告 ----
    print("\n===== 校准后可靠性 (预测均值 vs 实测频率, 10分箱) =====")
    rep = reliability_table(cal_probs, labels, mask)
    main_ths = {3900, 7700, 11600}
    for th in THRESHOLDS:
        pred, freq, cnt = rep[th]
        valid = cnt > 0
        mae = np.abs(pred[valid] - freq[valid]).mean() if valid.any() else float('nan')
        tag = " <-- 主阈值" if th in main_ths else ""
        print(f"\n阈值 >={th}{tag}  平均分箱误差 {mae*100:.2f}pp")
        for b in range(10):
            if cnt[b] > 0:
                print(f"   bin{b}: pred {pred[b]:.3f}  freq {freq[b]:.3f}  "
                      f"err {(pred[b]-freq[b])*100:+.1f}pp  n={int(cnt[b])}")

    # ---- 先验合理性: 开局第1巡 ----
    print("\n===== 先验合理性 =====")
    t1 = turn <= 1
    if t1.sum():
        emp = {}
        selm = mask[t1].astype(bool)
        for k, th in enumerate(THRESHOLDS):
            pv = cal_probs[t1][:, :, k][selm].mean()
            ev = labels[t1][:, :, k][selm].mean()
            emp[th] = (pv, ev, abs(pv - ev) * 100)
        for th in THRESHOLDS:
            pv, ev, dev = emp[th]
            print(f"  >={th}: 第1巡预测 {pv:.4f}  实证先验 {ev:.4f}  偏差 {dev:.2f}pp")


if __name__ == "__main__":
    main()
