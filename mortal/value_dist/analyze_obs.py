"""分析 obs 通道构成：分类 二值/整数/广播标量/浮点，确定紧凑编码方案。

用法: python -m value_dist.analyze_obs [文件数]
"""
import gzip, json, os, sys, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from libriichi.state import PlayerState
from value_dist.hand_tracker import HandTracker
from value_dist.extract import Extractor, OBS_VERSION, N_THRESH


def load_events(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main():
    n_files = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    pool = [f for f in os.listdir("E:/天凤训练数据/2025/") if f.endswith(".mjson")]
    random.seed(0); random.shuffle(pool)
    # 收集 obs 样本
    obs_list = []
    for fi in range(n_files):
        events = load_events("E:/天凤训练数据/2025/" + pool[fi])
        ps = PlayerState(0)
        tracker = HandTracker()
        for e in events:
            try:
                tracker.update(e)
                cans = ps.update(json.dumps(e))
            except Exception:
                continue
            if e.get("type") == "tsumo" and e.get("actor") == 0 and cans.can_discard:
                obs, _ = ps.encode_obs(OBS_VERSION, False)
                obs_list.append(obs)
                if len(obs_list) >= 200:
                    break
        if len(obs_list) >= 200:
            break
    obs = np.stack(obs_list)  # (N, 1012, 34)
    N = len(obs)
    print(f"样本 {N}, obs shape {obs.shape}, dtype {obs.dtype}")

    # 逐通道分类
    binary = 0    # 值 ∈ {0,1}
    integer = 0   # 值 ∈ 小整数(0-4 计数)
    broadcast = 0 # 沿34轴全相同(广播标量)
    float_chan = 0
    binary_idx, int_idx, bc_idx, float_idx = [], [], [], []
    for c in range(obs.shape[1]):
        ch = obs[:, c, :]
        vals = np.unique(ch)
        if vals.size == 1 or np.all(ch[:, 0:1] == ch):  # 沿34轴广播
            broadcast += 1; bc_idx.append(c)
        elif np.all((ch == 0) | (ch == 1)):
            binary += 1; binary_idx.append(c)
        elif np.all((ch == np.round(ch)) & (ch >= 0) & (ch <= 8)):
            integer += 1; int_idx.append(c)
        else:
            float_chan += 1; float_idx.append(c)
    print(f"二值通道: {binary} ({binary/obs.shape[1]*100:.0f}%)")
    print(f"整数通道(0-8): {integer}")
    print(f"广播标量通道: {broadcast}")
    print(f"浮点通道: {float_chan}")
    print(f"广播标量通道索引: {bc_idx[:20]}... 共{len(bc_idx)}")
    # 二值通道索引分布(是否连续)
    if binary_idx:
        print(f"二值通道索引范围: {min(binary_idx)}-{max(binary_idx)}, 是否连续: {sorted(binary_idx)==list(range(min(binary_idx),max(binary_idx)+1))}")
    # 估算压缩后大小
    orig = obs.nbytes
    # 二值 packbits: /8, 广播: /34, 整数: /2 (float16->uint8)
    est = binary * N * 34 / 8 + integer * N * 34 * 1 + broadcast * N * 1 + float_chan * N * 34 * 2
    print(f"\n原 obs 大小: {orig/1e6:.1f}MB/样本批")
    print(f"单样本原大小: {obs[0].nbytes/1024:.1f}KB")
    print(f"估算压缩后单样本: {est/N/1024:.1f}KB (压缩比 {obs[0].nbytes/est*N:.1f}x)")


if __name__ == "__main__":
    main()
