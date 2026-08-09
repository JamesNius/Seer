"""闸门2: 分片级读取吞吐基准。

模拟训练读取: 洗分片 → 每分片 np.load 解压 → 洗样本 → 分批。
实测 samples/sec, 必须 > GPU 消耗速率 1.5x。
用法: python -m value_dist.bench_dataloader <分片目录> [num_workers]
"""
import argparse, os, sys, time, glob, random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KEYS = ["obs","shan_label","wait_label","wait_mask","dora_label","somete_label",
        "mountain_label","value_label","value_mask"]


class ShardDataset(Dataset):
    """分片数据集: 每 epoch 洗分片顺序, 每分片解压一次, 分片内洗样本。"""

    def __init__(self, shard_dir, prefix="train"):
        self.shard_paths = sorted(glob.glob(os.path.join(shard_dir, f"{prefix}_a_*.npz")))
        self.perm = None  # 当前分片内样本顺序
        self.data = None  # 当前分片数据
        self.current_shard = None
        self.n = sum(self._shard_len(p) for p in self.shard_paths)
        print(f"分片 {len(self.shard_paths)}, 总样本 {self.n}")

    def _shard_len(self, path):
        # 只读第一个 key 的长度(避免全解压)
        d = np.load(path, mmap_mode="r")
        return len(d["shan_label"])

    def reset_epoch(self):
        random.shuffle(self.shard_paths)

    def load_shard(self, shard_path):
        d = np.load(shard_path)   # 解压进内存
        self.data = {k: d[k] for k in KEYS}
        self.perm = np.random.permutation(len(self.data["shan_label"]))
        self.current_shard = shard_path

    def get_batch(self, idx, batch_size):
        if self.data is None or idx >= len(self.perm):
            return None
        sel = self.perm[idx:idx+batch_size]
        return {k: self.data[k][sel] for k in KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="checkpoints/value_dist_data/_shards")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    ds = ShardDataset(args.shards)
    ds.reset_epoch()

    # 预热: 加载第一个分片
    ds.load_shard(ds.shard_paths[0])
    batch = ds.get_batch(0, args.batch)
    print(f"首分片样本 {len(ds.perm)}, 单批 obs {batch['obs'].shape}")

    # 吞吐基准: 连续加载分片 + 出 batch
    t0 = time.time()
    n_processed = 0
    n_batches = 0
    for shard in ds.shard_paths[:10]:   # 前 10 个分片
        ds.load_shard(shard)
        idx = 0
        while True:
            b = ds.get_batch(idx, args.batch)
            if b is None:
                break
            n_processed += len(b["obs"])
            n_batches += 1
            idx += args.batch
    dt = time.time() - t0
    print(f"10 分片处理: {n_processed} 样本, {dt:.1f}s")
    print(f"吞吐: {n_processed/dt:.0f} 样本/秒 (解压+出批)")
    print(f"目标: >GPU 消耗(2-5k/秒) 1.5x = 3-7.5k/秒")
    ok = n_processed/dt > 3000
    print(f"闸门2: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
