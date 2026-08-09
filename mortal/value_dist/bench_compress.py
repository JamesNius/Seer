"""闸门1: savez_compressed 压缩比 + 解压耗时实测。

先跑一次小提取生成一个分片(原格式), 再转 savez_compressed,
对比大小 + 实测单分片解压耗时。
用法: python -m value_dist.bench_compress [文件数]
"""
import gzip, json, os, sys, random, time, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from libriichi.state import PlayerState
from value_dist.hand_tracker import HandTracker
from value_dist.extract import extract_file


def main():
    n_files = int(sys.argv[1]) if len(sys.argv) > 1 else 25  # 一个分片规模
    pool = [f for f in os.listdir("E:/天凤训练数据/2025/") if f.endswith(".mjson")]
    random.seed(4); random.shuffle(pool)
    files = pool[:n_files]

    # 跑一次提取, 收集原始样本
    a_rows = []
    for path in files:
        try:
            per_viewer, stats = extract_file("E:/天凤训练数据/2025/" + path)
            for v in range(4):
                a_rows.extend(per_viewer[v])
        except Exception:
            continue
    print(f"样本 {len(a_rows)}")

    # 打包成 dict of arrays (与 run_extract.pack_rows 一致)
    out = {}
    for k in a_rows[0]:
        arrs = [r[k] for r in a_rows]
        if isinstance(arrs[0], np.ndarray):
            out[k] = np.stack(arrs)
        else:
            out[k] = np.array(arrs)

    # 原 savez (不压缩)
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tf:
        raw_path = tf.name
    np.savez(raw_path, **out)
    raw_sz = os.path.getsize(raw_path)
    print(f"savez 原格式: {raw_sz/1e6:.1f}MB")

    # savez_compressed
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tf:
        comp_path = tf.name
    np.savez_compressed(comp_path, **out)
    comp_sz = os.path.getsize(comp_path)
    print(f"savez_compressed: {comp_sz/1e6:.1f}MB (压缩比 {raw_sz/comp_sz:.1f}x)")

    # obs 单独压缩比
    obs_raw = out["obs"].nbytes
    obs_sz = None
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tf:
        obs_path = tf.name
    np.savez_compressed(obs_path, obs=out["obs"])
    obs_sz = os.path.getsize(obs_path)
    print(f"obs 单独: 原 {obs_raw/1e6:.1f}MB → 压缩 {obs_sz/1e6:.2f}MB ({obs_raw/obs_sz:.1f}x)")

    # 解压耗时 (整分片 np.load)
    t0 = time.time()
    for _ in range(5):
        d = np.load(comp_path)
        _ = d["obs"]
    dt = (time.time() - t0) / 5
    print(f"单分片解压耗时: {dt*1000:.0f}ms")

    # 吞吐: 样本数 / 解压耗时
    n = len(out["obs"])
    throughput = n / dt
    print(f"解压吞吐: {throughput:.0f} 样本/秒")

    for p in (raw_path, comp_path, obs_path):
        try: os.unlink(p)
        except Exception: pass


if __name__ == "__main__":
    main()
