"""value_dist 训练后校准流水线（温度缩放 + somete 后置校准 + ECE 验收闸）。

纪律:
  - checkpoints/value_dist.pth 只读, 所有校准参数写入 value_dist_calibrated.pth
  - 不重新训练、不改训练代码、不动 obs 分片
  - cal 半拟合参数, eval 半验收 ECE, 禁止在 eval 半拟合任何参数

步骤:
  1. 批量前向: 对 val 分片(cal+eval 半)各跑一次, 保存原始 logits + 标签
  2. 逐头温度缩放: 在 cal 半最小化 NLL
  3. somete 保序校准: isotonic (正例不足回退 Platt)
  4. 参数落盘: value_dist_calibrated.pth
  5. ECE 验收闸: eval 半等样本 20 箱

用法:
  python -m value_dist.calibrate_pipeline --step forward
  python -m value_dist.calibrate_pipeline --step fit
  python -m value_dist.calibrate_pipeline --step report
"""
import argparse, glob, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from value_dist.model import ValueDistNet
from value_dist.train import ShardLoader

CAL_SEED = 42
CAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "checkpoints", "value_dist_cal")
CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "checkpoints", "value_dist.pth")
CKPT_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "checkpoints", "value_dist_calibrated.pth")
HEADS = ["shan", "wait", "dora", "somete", "mountain", "value"]
# 各头形状 (除 batch 维)
HEAD_SHAPES = {"shan": (3, 5), "wait": (3, 34), "dora": (3, 7),
               "somete": (3, 4), "mountain": (34, 5), "value": (3, 6)}
# mask 依赖的头
MASK_HEADS = {"wait": "wait_mask", "value": "value_mask"}


def load_shard_labels(path):
    d = np.load(path, mmap_mode="r")
    out = {}
    for k in ["shan_label", "wait_label", "wait_mask", "dora_label",
              "somete_label", "mountain_label", "value_label", "value_mask"]:
        out[k] = d[k]
    return out


def step_forward():
    """对 val 全量跑一次前向, 存 cal/eval 半的 logits + 标签。"""
    os.makedirs(CAL_DIR, exist_ok=True)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = ValueDistNet(ck["conv"], ck["blocks"], heads=ck.get("heads")).to("cuda").eval()
    model.load_state_dict(ck["model"])

    # 读 val 全量样本(用 ShardLoader 但要拿全部, 不用 batch)
    val_ld = ShardLoader(os.path.join(os.path.dirname(CKPT), "value_dist_data"),
                         "val", 512, require=True, prefetch=0)
    rng = np.random.RandomState(CAL_SEED)
    # 固定种子对样本 index 打乱, 前一半=cal 后一半=eval
    all_idx = rng.permutation(val_ld.total)
    half = val_ld.total // 2
    cal_idx = set(all_idx[:half].tolist())
    print(f"val total={val_ld.total} cal={half} eval={val_ld.total-half} seed={CAL_SEED}", flush=True)

    # 逐分片直接读全量(不用 ShardLoader 的 batch 打乱), 全局 index = 累计偏移 + 片内位置
    shards = sorted(glob.glob(os.path.join(
        os.path.dirname(CKPT), "value_dist_data", "val_a_*.npz")))
    total = sum(len(np.load(p, mmap_mode="r")["shan_label"]) for p in shards)
    rng = np.random.RandomState(CAL_SEED)
    all_idx = rng.permutation(total)
    half = total // 2
    cal_idx = set(all_idx[:half].tolist())
    print(f"val total={total} cal={half} eval={total-half} seed={CAL_SEED}", flush=True)

    accum = {split: {h: [] for h in HEADS} for split in ("cal", "eval")}
    LABEL_KEYS = ["shan_label", "wait_label", "wait_mask", "dora_label",
                  "somete_label", "mountain_label", "value_label", "value_mask"]
    accum_labels = {split: {k: [] for k in LABEL_KEYS} for split in ("cal", "eval")}
    base = 0
    with torch.no_grad():
        for si, path in enumerate(shards):
            d = np.load(path)
            n = len(d["shan_label"])
            global_idx = np.arange(base, base + n)
            # 分片 obs 是 float16, 模型权重 float32 → 转 float32
            obs = torch.from_numpy(np.ascontiguousarray(d["obs"], dtype=np.float32)).cuda()
            logits = model(obs)
            # 分派: 全局 index → 片内局部
            gset = set(global_idx.tolist())
            for split, sel_global in (("cal", cal_idx & gset), ("eval", gset - cal_idx)):
                if not sel_global:
                    continue
                local = np.array(sorted(i - base for i in sel_global))
                for h, lo in zip(HEADS, logits):
                    accum[split][h].append(lo[local].cpu().numpy())
                for k in LABEL_KEYS:
                    accum_labels[split][k].append(d[k][local])
            base += n
            d.close()
            if (si + 1) % 8 == 0 or si + 1 == len(shards):
                print(f"  shard {si+1}/{len(shards)}", flush=True)

    # 落盘
    for split in ("cal", "eval"):
        for h in HEADS:
            arr = np.concatenate(accum[split][h], axis=0)
            np.save(os.path.join(CAL_DIR, f"{split}_{h}_logits.npy"), arr)
        for k, arrs in accum_labels[split].items():
            arr = np.concatenate(arrs, axis=0)
            np.save(os.path.join(CAL_DIR, f"{split}_{k}.npy"), arr)
        n = len(accum[split]["shan"][0])
        print(f"  {split} saved: {n} 样本", flush=True)
    print("step_forward done")


def _fit_temp_nll(logits, labels, mask=None, kind="softmax", init_t=1.0):
    """拟合标量温度 T, 最小化 NLL/BCE。返回 T。"""
    if mask is not None:
        m = mask.astype(bool)
        # 若 mask 全 False, 返回 1.0
        if m.sum() == 0:
            return 1.0
        logits = logits[m]
        labels = labels[m]
    logits_t = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    labels_t = torch.from_numpy(np.asarray(labels, dtype=np.float32))
    log_t = torch.tensor(np.log(init_t), requires_grad=True, dtype=torch.float32)
    opt = torch.optim.LBFGS([log_t], max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = logits_t / log_t.exp()
        if kind == "softmax":
            loss = -(torch.log_softmax(z, dim=-1) * labels_t).sum(dim=-1).mean()
        else:  # sigmoid
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, labels_t)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.detach().exp())


def step_fit():
    """逐头温度缩放: 在 cal 半最小化 NLL 拟合标量 T。"""
    temps = {}
    KIND = {"shan": "softmax", "wait": "sigmoid", "dora": "softmax",
            "somete": "sigmoid", "mountain": "softmax", "value": "sigmoid"}
    for h in HEADS:
        kind = KIND[h]
        logits = np.load(os.path.join(CAL_DIR, f"cal_{h}_logits.npy"))
        labels = np.load(os.path.join(CAL_DIR, f"cal_{h}_label.npy"))
        mask = None
        if h in MASK_HEADS:
            mask = np.load(os.path.join(CAL_DIR, f"cal_{MASK_HEADS[h]}.npy"))
        T = _fit_temp_nll(logits, labels, mask, kind)
        temps[h] = T
        print(f"  {h}: T={T:.4f}", flush=True)
    print("温度:", {k: round(v, 4) for k, v in temps.items()})
    with open(os.path.join(CAL_DIR, "temps.json"), "w") as f:
        json.dump(temps, f, indent=2)
    print("step_fit done")


def _auc(y, p):
    """Mann-Whitney AUC。"""
    pos = p[y == 1]; neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    gt = pos[:, None] > neg[None, :]
    eq = pos[:, None] == neg[None, :]
    return (gt.sum() + 0.5 * eq.sum()) / (len(pos) * len(neg))


def _platt_fit(z, y):
    """Platt: 在 logits 上拟合 logistic 校准。返回 (a, b) 使 p = sigmoid(a*z+b)。"""
    from scipy.optimize import minimize
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    def nll(params):
        a, b = params
        p = 1 / (1 + np.exp(-(a * z + b)))
        eps = 1e-9
        return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    res = minimize(nll, [1.0, 0.0], method="L-BFGS-B")
    return res.x


def step_somete_calib():
    """somete 保序校准: 每通道(ch0/1/2) isotonic, 正例不足回退 Platt。"""
    logits = np.load(os.path.join(CAL_DIR, "cal_somete_logits.npy"))
    labels = np.load(os.path.join(CAL_DIR, "cal_somete_label.npy"))
    T = json.load(open(os.path.join(CAL_DIR, "temps.json")))["somete"]
    cal_probs = 1 / (1 + np.exp(-logits / T))   # (N,3,4) 温度缩放后概率
    from scipy.optimize import isotonic_regression
    results = {}
    for ch in range(3):
        p = cal_probs[:, :, ch].reshape(-1)
        y = labels[:, :, ch].reshape(-1)
        n_pos = (y > 0.5).sum()
        print(f"  ch{ch}: n={len(y)} pos={n_pos} ({n_pos/len(y)*100:.2f}%)", flush=True)
        # sanity: 温度缩放是纯单调变换, 校准前(raw) vs 温度后 AUC 必须严格一致(~1e-10)
        z_raw = (logits[:, :, ch].reshape(-1))
        p_raw = 1 / (1 + np.exp(-z_raw))
        auc_raw = _auc(y, p_raw)
        auc_before = _auc(y, p)
        temp_diff = abs(auc_raw - auc_before)
        print(f"  ch{ch}: n={len(y)} pos={n_pos} ({n_pos/len(y)*100:.2f}%) raw_auc={auc_raw:.4f} temp_auc={auc_before:.4f} temp_diff={temp_diff:.2e} {'OK' if temp_diff<1e-6 else 'BUG!'}", flush=True)
        if n_pos < 200:
            z = (logits[:, :, ch].reshape(-1) / T).astype(np.float64)
            a, b = _platt_fit(z, y)
            results[ch] = {"method": "platt", "a": float(a), "b": float(b),
                           "auc_before": auc_before, "n_pos": int(n_pos)}
            cal_p = 1 / (1 + np.exp(-(a * z + b)))
            print(f"    Platt a={a:.4f} b={b:.4f} AUC_before={auc_before:.4f}", flush=True)
        else:
            # isotonic regression (等样本按概率排序)
            order = np.argsort(p)
            iso_res = isotonic_regression(y[order].astype(np.float64), increasing=True)
            iso = np.asarray(iso_res.x) if hasattr(iso_res, "x") else np.asarray(iso_res)
            x_sorted = p[order]
            step = max(1, len(x_sorted)//200)
            results[ch] = {"method": "isotonic",
                           "x": x_sorted[::step].tolist(),
                           "y": iso[::step].tolist(),
                           "auc_before": auc_before, "n_pos": int(n_pos)}
            # 校准后概率: 完整 isotonic 输出(不降采样), 用于精确 AUC 对拍
            cal_full = np.empty_like(p)
            cal_full[order] = iso
            cal_p = cal_full
            print(f"    isotonic AUC_before={auc_before:.4f}", flush=True)
        auc_after = _auc(y, cal_p)
        results[ch]["auc_after"] = auc_after
        # sanity: 校准前后 AUC 必须逐位一致(单调变换不改排序, 差异应≈0)
        diff = abs(auc_before - auc_after)
        print(f"    AUC_after={auc_after:.4f} diff={diff:.6f} {'OK' if diff < 1e-6 else 'MISMATCH!'}", flush=True)
    with open(os.path.join(CAL_DIR, "somete_calib.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("step_somete_calib done")


def step_save():
    """校准参数落盘: value_dist_calibrated.pth = 原 checkpoint + calibration 字段。"""
    import shutil
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    temps = json.load(open(os.path.join(CAL_DIR, "temps.json")))
    somete = json.load(open(os.path.join(CAL_DIR, "somete_calib.json")))
    # isotonic 存查找表(x,y)
    somete_isotonic = {}
    for ch, v in somete.items():
        if v["method"] == "isotonic":
            somete_isotonic[ch] = (v["x"], v["y"])
        else:
            somete_isotonic[ch] = ("platt", v["a"], v["b"])
    calibration = {
        "temperature": temps,
        "somete_isotonic": somete_isotonic,
        "cal_seed": CAL_SEED,
        "cal_samples": 246604,
        "epoch": ck.get("epoch", None),
    }
    # 只写新文件, 不碰原 checkpoint
    ck["calibration"] = calibration
    torch.save(ck, CKPT_OUT)
    print(f"已写 {CKPT_OUT}, calibration 字段:")
    print(json.dumps(calibration, indent=2)[:800])


def _ece_equal_freq(probs, labels, mask=None, n_bins=20):
    """等样本分箱 ECE (每箱样本数相同)。返回标量 ECE。"""
    if mask is not None:
        m = mask.astype(bool)
        probs = probs[m]
        labels = labels[m]
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    order = np.argsort(probs)
    probs_s, lab_s = probs[order], labels[order]
    n = len(probs_s)
    if n == 0:
        return 0.0, 0
    bin_size = n // n_bins
    ece = 0.0
    for b in range(n_bins):
        s, e = b * bin_size, (b + 1) * bin_size if b < n_bins - 1 else n
        if e <= s:
            continue
        p = probs_s[s:e].mean()
        f = lab_s[s:e].mean()
        ece += (e - s) / n * abs(p - f)
    return ece, n


def _apply_calib(probs, head, T, somete_isotonic=None):
    """对 logits 应用温度 + somete isotonic, 返回校准后概率。"""
    if head == "somete" and somete_isotonic is not None:
        # somete: 每通道 isotonic
        out = np.zeros_like(probs)
        for ch in range(3):
            spec = somete_isotonic.get(str(ch))
            if spec is None:
                out[..., ch] = probs[..., ch]
            elif spec[0] == "platt":
                _, a, b = spec
                z = np.log(probs[..., ch] / np.clip(1 - probs[..., ch], 1e-9, 1))
                out[..., ch] = 1 / (1 + np.exp(-(a * z + b)))
            else:
                x, y = spec
                out[..., ch] = np.interp(probs[..., ch], x, y)
        # ch3 死通道保持原值
        out[..., 3] = probs[..., 3]
        return out
    return probs


def step_report():
    """ECE 验收闸: eval 半, 等样本 20 箱, 每头三列(原始/仅温度/最终)。"""
    temps = json.load(open(os.path.join(CAL_DIR, "temps.json")))
    # 用落盘的 somete_isotonic(通道 key 是 "0"/"1"/"2"), 与 _apply_calib 一致
    ck = torch.load(CKPT_OUT, map_location="cpu", weights_only=False)
    somete = ck["calibration"]["somete_isotonic"]
    THRESH = {"shan": 0.02, "dora": 0.02, "mountain": 0.03,
              "wait": 0.08, "value": 0.10, "somete": 0.05}
    print(f"{'头':10} {'原始ECE':>9} {'仅温度ECE':>9} {'最终ECE':>9} {'闸限':>6} {'判定':>4}")
    print("-" * 60)
    results = {}
    for h in HEADS:
        logits = np.load(os.path.join(CAL_DIR, f"eval_{h}_logits.npy"))
        labels = np.load(os.path.join(CAL_DIR, f"eval_{h}_label.npy"))
        mask = None
        if h in MASK_HEADS:
            mask = np.load(os.path.join(CAL_DIR, f"eval_{MASK_HEADS[h]}.npy"))
        T = temps[h]
        # 原始概率
        if h in ("shan", "dora", "mountain"):
            e = np.exp(logits - logits.max(-1, keepdims=True))
            p_raw = e / e.sum(-1, keepdims=True)
            # softmax 用 argmax 置信度做 ECE
            p_raw_ec = p_raw.max(-1)
            lab_ec = labels.argmax(-1)
            ece_raw, _ = _ece_equal_freq(p_raw_ec, (np.arange(5)[None,None,:] == labels[...,None]).argmax(-1) if False else (p_raw.argmax(-1) == labels.argmax(-1)), mask)
            # 仅温度
            e = np.exp(logits / T - (logits / T).max(-1, keepdims=True))
            p_t = e / e.sum(-1, keepdims=True)
            ece_temp, _ = _ece_equal_freq(p_t.max(-1), p_t.argmax(-1) == labels.argmax(-1), mask)
            ece_final = ece_temp
        else:
            p_raw = 1 / (1 + np.exp(-logits))
            p_t = 1 / (1 + np.exp(-logits / T))
            ece_raw, _ = _ece_equal_freq(p_raw, labels, mask)
            ece_temp, _ = _ece_equal_freq(p_t, labels, mask)
            p_final = _apply_calib(p_t, h, T, somete)
            ece_final, _ = _ece_equal_freq(p_final, labels, mask)
        results[h] = {"raw": ece_raw, "temp": ece_temp, "final": ece_final}
        th = THRESH[h]
        ok = "PASS" if ece_final <= th else "FAIL"
        print(f"{h:10} {ece_raw:.4f}   {ece_temp:.4f}   {ece_final:.4f}   {th:.2f}  {ok}")
    with open(os.path.join(CAL_DIR, "ece_report.json"), "w") as f:
        json.dump(results, f, indent=2)
    # somete 额外校准曲线
    _somete_curve(eval_p=None, labels=None)
    print("step_report done")


def _somete_curve(eval_p, labels):
    """somete 每通道校准曲线: 10 个点(预测均值 vs 实测频率), 供人工目检。"""
    logits = np.load(os.path.join(CAL_DIR, "eval_somete_logits.npy"))
    labels = np.load(os.path.join(CAL_DIR, "eval_somete_label.npy"))
    T = json.load(open(os.path.join(CAL_DIR, "temps.json")))["somete"]
    somete = json.load(open(os.path.join(CAL_DIR, "somete_calib.json")))
    p_t = 1 / (1 + np.exp(-logits / T))
    p_cal = _apply_calib(p_t, "somete", T, somete)
    print("\nsomete 校准曲线 (预测均值 vs 实测频率, 10点):")
    for ch in range(3):
        p = p_cal[:, :, ch].reshape(-1)
        y = labels[:, :, ch].reshape(-1)
        order = np.argsort(p)
        ps, ys = p[order], y[order]
        bin_edges = np.linspace(0, len(ps), 11).astype(int)
        pts = []
        for b in range(10):
            s, e = bin_edges[b], bin_edges[b+1]
            if e > s:
                pts.append((ps[s:e].mean(), ys[s:e].mean()))
        print(f"  ch{ch}: " + "  ".join(f"({a:.2f},{b:.2f})" for a, b in pts))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["forward", "fit", "somete", "save", "report"], default="forward")
    args = ap.parse_args()
    if args.step == "forward":
        step_forward()
    elif args.step == "fit":
        step_fit()
    elif args.step == "somete":
        step_somete_calib()
    elif args.step == "save":
        step_save()
    elif args.step == "report":
        step_report()
