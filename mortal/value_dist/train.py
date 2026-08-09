"""训练六头读牌模型（分片级读取）。

分片级训练:
- 两级 shuffle: 先洗分片顺序, 片内再洗样本
- 每分片每 epoch 只解压一次(savez_compressed)
- 完整性校验: 每 epoch 消费样本数 == 总样本数

策略:
- conv=96 blocks=20 从 0 训
- cosine 退火 + val 早停(patience=3)
- 每 epoch 落 checkpoint + 打印 val 各头 ECE
- --resume 从最新 checkpoint 续训

用法:
  python -m value_dist.train --data checkpoints/value_dist_data \
      --out checkpoints/value_dist.pth
  # 从 checkpoint 续训(不重跑):
  python -m value_dist.train --data checkpoints/value_dist_data \
      --out checkpoints/value_dist.pth --resume
  # 冒烟(前 4 分片, 2 epoch, 小模型快速验证逻辑):
  python -m value_dist.train --data checkpoints/value_dist_data \
      --out checkpoints/value_dist_smoke.pth --smoke
"""
import argparse, glob, os, sys, time, math, random, queue, threading

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from value_dist.model import ValueDistNet, multitask_loss, head_specs

KEYS = ["obs","shan_label","wait_label","wait_mask","dora_label","somete_label",
        "mountain_label","value_label","value_mask"]


def shard_len(path):
    d = np.load(path, mmap_mode="r")
    return len(d["shan_label"])


def _load_shard_data(path):
    """完整解压一个分片为 numpy dict(后台预取线程调用)。"""
    d = np.load(path)
    return {k: d[k] for k in KEYS}


class ShardLoader:
    """分片数据集: 两级 shuffle, 每分片每 epoch 解压一次。

    双缓冲预取: 后台线程提前解压下一个分片, 与 GPU 训练并行,
    消除"解压时 GPU 空等"的串行瓶颈(参考 Mortal dataloader 流水线)。
    """

    def __init__(self, shard_dir, prefix="train", batch_size=512, require=True,
                 max_shards=0, prefetch=1):
        self.paths = sorted(glob.glob(os.path.join(shard_dir, f"{prefix}_a_*.npz")))
        if max_shards:
            self.paths = self.paths[:max_shards]
        if require:
            assert self.paths, f"no shards found in {shard_dir}"
        self.shard_n = [shard_len(p) for p in self.paths]
        self.total = sum(self.shard_n)
        self.batch_size = batch_size
        self.prefetch = prefetch
        self.shard_order = []
        self._shard_data = None   # 当前分片数据
        self._cache = {}          # si -> 预取线程解压好的数据 dict
        self.perm = None          # 当前分片内样本顺序
        self.idx = 0
        self.consumed = 0        # 本 epoch 已消费样本(完整性校验)
        self.prefix = prefix
        self._prefetch_q = queue.Queue()
        self._prefetch_th = None
        self._prefetch_stop = threading.Event()
        self._consumed_si = set()  # 已被主线程消费的序号(预取线程跳过)

    def __len__(self):
        return self.total

    def _prefetch_loop(self):
        """后台线程: 从队列拿分片序号, 解压到 _cache[si](已消费的跳过)。

        保持 _cache 最多 prefetch 个未消费分片; 满了就等, 防止预取快于
        消费时堆积(每个分片解压后 ~900MB, 堆积会导致内存耗尽)。
        """
        while not self._prefetch_stop.is_set():
            if len(self._cache) >= self.prefetch:
                time.sleep(0.05)
                continue
            try:
                si = self._prefetch_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if si is None:
                break
            if si in self._consumed_si:
                continue
            try:
                self._cache[si] = _load_shard_data(self.paths[si])
            except Exception:
                pass

    def _start_prefetch(self):
        if self._prefetch_th is None:
            self._prefetch_stop.clear()
            self._prefetch_th = threading.Thread(
                target=self._prefetch_loop, daemon=True, name=f"prefetch-{self.prefix}")
            self._prefetch_th.start()

    def _stop_prefetch(self):
        if self._prefetch_th is not None:
            self._prefetch_stop.set()
            self._prefetch_q.put(None)
            self._prefetch_th.join(timeout=2)
            self._prefetch_th = None

    def reset_epoch(self):
        random.shuffle(self.paths)
        self.shard_order = list(range(len(self.paths)))
        self._shard_data = None
        self._cache.clear()
        self._consumed_si.clear()
        self.consumed = 0

    def _next_shard(self):
        """取下一个分片数据: 优先用预取缓存, 否则同步解压。"""
        if not self.shard_order:
            return None
        si = self.shard_order.pop()
        self._consumed_si.add(si)
        data = self._cache.pop(si, None)
        if data is None:
            data = _load_shard_data(self.paths[si])
        return si, data

    def get_batch(self):
        """返回下一批 dict 或 None(epoch 结束, 尾部不足 batch 也返回不丢弃)。"""
        if self._shard_data is None:
            if not self.shard_order:
                return None
            if self.prefetch > 0:
                self._start_prefetch()
            si, self._shard_data = self._next_shard()
            self.perm = np.random.permutation(len(self._shard_data["shan_label"]))
            self.idx = 0
            # 触发下一个分片的后台预取
            if self.prefetch > 0 and self.shard_order:
                self._prefetch_q.put(self.shard_order[-1])
        n_avail = len(self.perm) - self.idx
        if n_avail <= 0:
            self._shard_data = None
            return self.get_batch()
        sel = self.perm[self.idx:self.idx+self.batch_size]
        self.idx += len(sel)
        self.consumed += len(sel)
        return {k: self._shard_data[k][sel] for k in KEYS}

    def finish_epoch(self):
        """epoch 结束清理后台线程 + 释放已解压分片内存。"""
        self._stop_prefetch()
        self._cache.clear()
        self._shard_data = None
        self.perm = None


def to_tensor(batch, device):
    out = {}
    for k, v in batch.items():
        t = torch.from_numpy(np.ascontiguousarray(v))
        out[k] = t.to(device, non_blocking=True)
    return out


def evaluate(model, val_ld, device):
    """val 全量评估: 返回 dict(val_loss, 各头 loss, 各头 ECE)。

    ECE 用流式 10-bin 统计(每 batch 累加 bin 的 sum_conf/sum_acc/count)，
    不缓存原始 logits，避免 49 万样本 × 6 头累积 OOM。
    """
    model.eval()
    total_loss = 0.0
    n_batch = 0
    losses_acc = {}
    # ECE 缓存: 每头累积 (conf, acc) 展平, 供等样本 20 箱(与校准管线同口径)
    ece_acc = {k: {"conf": [], "acc": []} for k in KEYS_BIN}
    ECE_KIND = {"shan": "softmax", "dora": "softmax", "mountain": "softmax",
                "wait": "sigmoid", "somete": "sigmoid", "value": "sigmoid"}
    with torch.no_grad():
        while True:
            batch = val_ld.get_batch()
            if batch is None:
                break
            dev = to_tensor(batch, device)
            with torch.autocast(device_type=device, enabled=(device=="cuda")):
                shan, wait, dora, somete, mountain, value = model(dev["obs"])
                dev.update({"shan": shan, "wait": wait, "dora": dora,
                            "somete": somete, "mountain": mountain, "value": value})
                loss, comp = multitask_loss(dev)
            total_loss += loss.item()
            n_batch += 1
            for k, v in comp.items():
                losses_acc[k] = losses_acc.get(k, 0.0) + v
            # ECE 累积(缓存 prob/label)
            for k in KEYS_BIN:
                logits = dev[k].float()
                lab = dev[f"{k}_label"].float()
                m = dev[f"{k}_mask"].float() if f"{k}_mask" in dev else None
                _ece_collect(ece_acc[k], logits, lab, m, ECE_KIND[k])
    val_ld.finish_epoch()   # 释放 val 分片内存(避免与主循环 train_ld 叠加 OOM)
    avg_loss = total_loss / max(1, n_batch)
    avg_comp = {k: v / max(1, n_batch) for k, v in losses_acc.items()}
    ece = {k: _ece_from_collected(v) for k, v in ece_acc.items()}
    return avg_loss, avg_comp, ece


KEYS_BIN = ("shan", "wait", "dora", "somete", "mountain", "value")


def _ece_collect(acc, logits, labels, mask=None, kind="softmax"):
    """缓存每头的 (prob, label, mask) 展平, 供等样本 ECE 用(与校准管线同口径)。

    mask 是 (N,head) 布尔, 对 logits/labels 的 (N,head,C) 做首两维布尔索引,
    与校准管线 _ece_equal_freq 的 probs[mask] 语义一致。
    """
    if kind == "softmax":
        probs = torch.softmax(logits.float(), dim=-1)
        conf, pred = probs.max(dim=-1)   # (N,head)
        acc_ok = (pred == labels.argmax(dim=-1)).float()
    else:
        probs = torch.sigmoid(logits.float())
        conf = probs
        acc_ok = labels.float()          # 频率口径
    if mask is not None:
        m = mask.bool()
        conf = conf[m]
        acc_ok = acc_ok[m]
    acc["conf"].append(conf.reshape(-1).cpu().numpy())
    acc["acc"].append(acc_ok.reshape(-1).cpu().numpy())


def _ece_from_collected(acc, n_bins=20):
    """由缓存的 (conf, acc) 算等样本 ECE(排序后均分 n_bins)。与校准管线一致。"""
    conf = np.concatenate(acc["conf"])
    ac = np.concatenate(acc["acc"])
    n = len(conf)
    if n == 0:
        return 0.0
    order = np.argsort(conf)
    cs, as_ = conf[order], ac[order]
    bin_size = n // n_bins
    ece = 0.0
    for b in range(n_bins):
        s, e = b * bin_size, (b + 1) * bin_size if b < n_bins - 1 else n
        if e <= s:
            continue
        ece += (e - s) / n * abs(cs[s:e].mean() - as_[s:e].mean())
    return float(ece)


def _save_ckpt(path, model, args, epoch, best_val, opt, scaler):
    torch.save({"model": model.state_dict(), "conv": args.conv,
                "blocks": args.blocks, "epoch": epoch, "best_val": best_val,
                "heads": head_specs(), "head_temps": {},
                "opt": opt.state_dict(), "scaler": scaler.state_dict()},
               path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="checkpoints/value_dist_data")
    ap.add_argument("--out", default="checkpoints/value_dist.pth")
    ap.add_argument("--conv", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=20)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--final-lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--resume", action="store_true",
                    help="从 --out 已有 checkpoint 续训")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟: 前 4 分片, 2 epoch, 快速验证逻辑")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device)

    # 冒烟模式: 缩小数据与 epoch
    smoke = args.smoke
    max_shards = 4 if smoke else 0
    epochs = 2 if smoke else args.epochs
    if smoke:
        print("SMOKE 模式: 前 4 分片, 2 epoch")

    train_ld = ShardLoader(args.data, "train", args.batch, max_shards=max_shards)
    val_ld = ShardLoader(args.data, "val", args.batch, require=False,
                         max_shards=(1 if smoke else 0), prefetch=0)
    # val 评估不用预取: 逐分片即用即弃, 避免预取缓存积累导致内存耗尽
    print(f"train 分片 {len(train_ld.paths)} 样本 {train_ld.total}")
    print(f"val   分片 {len(val_ld.paths)} 样本 {val_ld.total}")

    # 从 checkpoint 读 head 注册表(有则用它构建, 无则默认), 支持加头后的续训
    resume_heads = None
    if args.resume and os.path.exists(args.out):
        resume_ck = torch.load(args.out, map_location="cpu", weights_only=False)
        resume_heads = resume_ck.get("heads")
    model = ValueDistNet(args.conv, args.blocks, heads=resume_heads).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd,
                            betas=(0.9, 0.999), eps=1e-8)
    scaler = torch.amp.GradScaler(device, enabled=(device=="cuda"))

    start_epoch = 0
    best_val = None
    if args.resume and os.path.exists(args.out):
        ck = resume_ck
        try:
            model.load_state_dict(ck["model"])
        except RuntimeError:
            model.load_legacy_state_dict(ck["model"])   # 旧命名 checkpoint
        if "opt" in ck and "scaler" in ck:
            opt.load_state_dict(ck["opt"])
            scaler.load_state_dict(ck["scaler"])
        start_epoch = ck.get("epoch", 0) + 1
        best_val = ck.get("best_val", None)
        print(f"RESUME: epoch {ck.get('epoch',0)} -> {start_epoch}, best_val={best_val}")
    else:
        print(f"从 0 训练: conv={args.conv} blocks={args.blocks}")

    # steps 估计
    train_steps_est = train_ld.total // args.batch
    total_steps = train_steps_est * epochs
    warmup = min(args.warmup, total_steps // 20)

    def lr_at(step):
        if step < warmup:
            return args.lr * step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return args.final_lr + (args.lr - args.final_lr) * 0.5 * (1 + math.cos(math.pi * prog))

    step = 0
    patience_count = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        train_ld.reset_epoch()
        t0 = time.time()
        run_loss, nb = 0.0, 0
        while True:
            batch = train_ld.get_batch()
            if batch is None:
                break
            dev = to_tensor(batch, device)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast(device_type=device, enabled=(device=="cuda")):
                shan, wait, dora, somete, mountain, value = model(dev["obs"])
                dev.update({"shan": shan, "wait": wait, "dora": dora,
                            "somete": somete, "mountain": mountain, "value": value})
                loss, comp = multitask_loss(dev)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item(); nb += 1; step += 1
        # 完整性校验
        assert train_ld.consumed == train_ld.total, \
            f"epoch {epoch}: consumed {train_ld.consumed} != total {train_ld.total}"
        train_ld.finish_epoch()
        print(f"epoch {epoch}  loss {run_loss/max(1,nb):.4f}  consumed {train_ld.consumed}/{train_ld.total}  {time.time()-t0:.0f}s", flush=True)

        # val 评估 + ECE
        if val_ld.total > 0:
            val_ld.reset_epoch()
            v_loss, v_comp, v_ece = evaluate(model, val_ld, device)
            ece_str = "  ".join(f"{k}:{v:.4f}" for k, v in v_ece.items())
            print(f"  val_loss {v_loss:.4f}  ECE [{ece_str}]", flush=True)
            for k, v in v_comp.items():
                print(f"    head_loss {k}: {v:.4f}", flush=True)
            if best_val is None or v_loss < best_val:
                best_val = v_loss
                patience_count = 0
            else:
                patience_count += 1
                print(f"  val 无改善 {patience_count}/{args.patience}")
                if patience_count >= args.patience:
                    print(f"早停: val 连续 {args.patience} epoch 无改善, best={best_val:.4f}")
                    _save_ckpt(args.out, model, args, epoch, best_val, opt, scaler)
                    break

        # 每 epoch 落 checkpoint(可续) + 单独备份 epoch 版本(供日后挑版)
        _save_ckpt(args.out, model, args, epoch, best_val, opt, scaler)
        epoch_backup = os.path.join(os.path.dirname(args.out), f"value_dist_epoch{epoch}.pth")
        _save_ckpt(epoch_backup, model, args, epoch, best_val, opt, scaler)
        print(f"  saved {args.out} + {epoch_backup} (epoch {epoch})", flush=True)


if __name__ == "__main__":
    main()
