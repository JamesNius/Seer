"""抽取驱动: 多进程遍历数据池, 按局切分 train/val, 写出数据集。

适配 v2 extract(八头 + will_call)。每个样本写出:
  采样点A(tsumo): obs / shan_label / wait_label / dora_label /
    somete_label / mountain_label / value_label / value_mask / can_call_label
  will 数据集: will_label / can_label(切牌事件)

用法:
  python -m value_dist.run_extract --year 2025 --max-files 100 --workers 6 \
      --out checkpoints/value_dist_data
续跑(不删已有分片, 已存在的分片跳过):
  python -m value_dist.run_extract --year 2025 --max-files 100 --workers 6 \
      --out checkpoints/value_dist_data --resume
  注意: 续跑必须与上次使用完全相同的 --year/--max-files/--val-ratio/--seed/
    --batch-files 参数, 否则 shuffle 不一致, 已有分片对不上。
"""
import argparse, os, sys, random, shutil, time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from value_dist.extract import extract_file

OBS_C, OBS_W = 1012, 34


def pack_rows(rows):
    """rows: list of sample dict -> numpy 字典。"""
    n = len(rows)
    if n == 0:
        return None
    out = {}
    for k in rows[0]:
        arrs = [r[k] for r in rows]
        if isinstance(arrs[0], np.ndarray):
            out[k] = np.stack(arrs)
        else:
            out[k] = np.array(arrs)
    return out


def worker(args):
    shard_id, is_val, paths, shard_dir = args
    split = "val" if is_val else "train"
    shard_path = os.path.join(shard_dir, f"{split}_a_{shard_id:05d}.npz")
    # 断点续跑: 分片已存在则跳过(幂等)。已存在分片在启动前应已通过坏分片扫描。
    if os.path.exists(shard_path):
        return shard_path, split, 0, {"samples": 0, "wall_err": 0,
                                      "wall_total": 0, "files_err": 0}
    agg = {"samples": 0, "wall_err": 0, "wall_total": 0,
           "files_err": 0}
    a_rows = []
    for path in paths:
        try:
            per_viewer, stats = extract_file(path)
            for v in range(4):
                a_rows.extend(per_viewer[v])
            for k in ("samples","wall_err","wall_total"):
                agg[k] += stats[k]
        except Exception:
            agg["files_err"] += 1
    data = pack_rows(a_rows) if a_rows else None
    if data is not None:
        np.savez_compressed(shard_path, **data)   # 压缩存储(obs 66x)
    else:
        shard_path = None
    return shard_path, split, len(a_rows), agg


def merge_shards(shard_paths, out_prefix, keys):
    total_n = 0
    shapes = {}
    for sp in shard_paths:
        d = np.load(sp)
        total_n += len(d[keys[0]])
        for k in keys:
            shapes[k] = d[k].shape[1:]
    out = {}
    offsets = {}
    d0 = np.load(shard_paths[0])
    for k in keys:
        p = f"{out_prefix}_{k}.npy"
        mm = np.lib.format.open_memmap(p, mode="w+", dtype=d0[k].dtype, shape=(total_n, *shapes[k]))
        out[k] = mm
        offsets[k] = 0
    for sp in shard_paths:
        d = np.load(sp)
        n = len(d[keys[0]])
        for k in keys:
            o = offsets[k]
            out[k][o:o+n] = d[k]
            offsets[k] += n
    for k in keys:
        out[k].flush()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", default="2025")
    ap.add_argument("--out", default="checkpoints/value_dist_data")
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch-files", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="续跑: 不清空 _shards, 已存在的分片跳过(须与上次参数完全一致)")
    args = ap.parse_args()

    files = sorted(os.listdir(f"E:/天凤训练数据/{args.year}/"))
    files = [f"E:/天凤训练数据/{args.year}/" + f for f in files if f.endswith(".mjson")]
    if args.max_files:
        files = files[:args.max_files]
    random.seed(args.seed); random.shuffle(files)

    n_val = max(1, int(len(files) * args.val_ratio))
    print(f"total {len(files)}  train {len(files)-n_val}  val {n_val}")
    os.makedirs(args.out, exist_ok=True)
    shard_dir = os.path.join(args.out, "_shards")
    if not args.resume:
        shutil.rmtree(shard_dir, ignore_errors=True)
    os.makedirs(shard_dir, exist_ok=True)
    if args.resume:
        existing = sorted(os.listdir(shard_dir))
        print(f"--resume: 保留 _shards, 已有 {len(existing)} 个分片将被跳过")

    val_files = files[:n_val]
    train_files = files[n_val:]
    tasks = []
    for is_val, group in ((False, train_files), (True, val_files)):
        for i in range(0, len(group), args.batch_files):
            batch = group[i:i+args.batch_files]
            tasks.append((len(tasks), is_val, batch, shard_dir))

    t0 = time.time()
    train_shards, val_shards = [], []
    n_tr = n_va = 0
    total = {"samples":0,"wall_err":0,"wall_total":0,"files_err":0}
    done = 0
    with Pool(args.workers) as pool:
        for shard_path, split, nrows, agg in pool.imap_unordered(worker, tasks):
            done += 1
            for k in total: total[k] += agg[k]
            if split == "val": val_shards.append(shard_path); n_va += nrows
            else: train_shards.append(shard_path); n_tr += nrows
            if done % 5 == 0 or done == len(tasks):
                print(f"[{done}/{len(tasks)}] train {n_tr} val {n_va} 山残err {total['wall_err']}/{total['wall_total']}", flush=True)

    # 不做 memmap 合并(会展开成巨量未压缩 npy)。把压缩分片挪到最终目录,
    # 训练直接读压缩分片(两级 shuffle)。
    import shutil as _sh
    for sp in train_shards + val_shards:
        if sp:
            dst = os.path.join(args.out, os.path.basename(sp))
            if not os.path.exists(dst):
                _sh.move(sp, dst)
    shutil.rmtree(shard_dir, ignore_errors=True)

    print("\n===== 抽取报告 =====")
    print(f"文件 {len(files)} 失败 {total['files_err']}")
    print(f"样本 train {n_tr}  val {n_va}")
    print(f"山残校验: {total['wall_err']}/{total['wall_total']} 错误率 {total['wall_err']/max(1,total['wall_total']):.4f}")
    print(f"耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
