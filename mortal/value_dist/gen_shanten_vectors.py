"""进张计算真值生成器（真值源 = PyPI mahjong 库）。

进张算法(hand14 视角, 全程 14/13 张, 天然合法):
  1. shanten(hand14) = s
  2. 每张去重弃牌 d: shanten(hand14 - d) = s13[d]  (~14 次)
  3. 只对"向听不亏"的弃牌(s13[d] <= s)展开 34 次加牌:
       shanten(hand14 - d + t) < s13[d]  →  t 是进张牌
每条向量 = hand14(34计数) + shanten + {弃牌: [进张牌]} + 并集进张。

用法: python -m value_dist.gen_shanten_vectors [N] [out.jsonl]
"""
import json, random, sys, os
from multiprocessing import Pool
from mahjong.shanten import Shanten

_lib = Shanten()
TILE_POOL = list(range(34))


def gen_hand14(rng):
    """随机生成 14 张手牌(34计数)。"""
    c = [0] * 34
    while sum(c) < 14:
        t = rng.choice(TILE_POOL)
        if c[t] < 4:
            c[t] += 1
    return c


def gen_edge_hand(kind):
    c = [0] * 34
    if kind == "chiitoi_wait":
        for i in [0, 2, 4, 9, 11, 20, 30]: c[i] = 2          # 7对=14
    elif kind == "kokushi_wait":
        for i in [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32]: c[i] = 1
        c[27] = 2; c[33] = 1                                 # 13幺九+中=14
    elif kind == "complex":
        c[2] = c[3] = c[4] = 1; c[3] += 1                    # 3445m 复合
        c[9] = c[10] = c[11] = 1; c[18] = c[19] = c[20] = 1  # 456p 789s
        c[27] = 2; c[33] = 1                                  # 南南中 = 14
    elif kind == "aka5":
        c[4] = c[13] = c[22] = 1                              # 赤5(普通5表示)
        c[0] = c[1] = c[2] = 1; c[9] = c[10] = c[11] = 1
        c[18] = c[19] = c[20] = 1; c[27] = 2                  # = 14
    elif kind == "four_dead":
        c[0] = 4                                              # 4张捏死
        c[9] = c[10] = c[11] = 1; c[18] = c[19] = c[20] = 1
        c[27] = 2; c[33] = 1; c[4] = 1                        # = 14
    # 保证 14 张
    s = sum(c)
    i = 0
    while s < 14:
        if c[i] < 4: c[i] += 1; s += 1
        i = (i + 1) % 34
    while s > 14:
        if c[i] > 0: c[i] -= 1; s -= 1
        i = (i + 1) % 34
    return c


def ukeire_14(c):
    """按用户算法算进张。返回 (shanten, {弃牌: [进张牌]}, 并集进张)。"""
    s = _lib.calculate_shanten(c)
    ukeire_map = {}
    for d in range(34):
        if c[d] == 0:
            continue
        h13 = c.copy(); h13[d] -= 1
        s13 = _lib.calculate_shanten(h13)
        if s13 > s:   # 弃牌更差 → 不展开
            continue
        tiles = []
        for t in range(34):
            if h13[t] >= 4:
                continue
            h14b = h13.copy(); h14b[t] += 1
            if _lib.calculate_shanten(h14b) < s13:
                tiles.append(t)
        if tiles:
            ukeire_map[d] = tiles
    # 并集进张(所有不亏弃牌的进张牌)
    ukeire_union = sorted(set(t for ts in ukeire_map.values() for t in ts))
    return s, ukeire_map, ukeire_union


def _worker_chunk(seed, n):
    """多进程 worker: 生成 n 条随机向量。"""
    rng = random.Random(seed)
    vecs = []
    for _ in range(n):
        c = gen_hand14(rng)
        s, um, un = ukeire_14(c)
        vecs.append({"hand": c, "shanten": s, "ukeire_map": um, "ukeire": un})
    return vecs


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    out = sys.argv[2] if len(sys.argv) > 2 else "shanten_vectors.jsonl"
    workers = min(8, os.cpu_count() or 4)
    chunk = max(1, n // workers)
    print(f"并行生成 {n} 条, {workers} workers, 每 worker {chunk} 条", flush=True)
    with Pool(workers) as pool:
        chunks = pool.starmap(_worker_chunk,
                              [(i * 7919 + 1, chunk if i < workers - 1 else n - chunk * (workers - 1))
                               for i in range(workers)])
    vecs = [v for c in chunks for v in c]
    for kind in ["chiitoi_wait", "kokushi_wait", "complex", "aka5", "four_dead"]:
        c = gen_edge_hand(kind)
        s, um, un = ukeire_14(c)
        vecs.append({"hand": c, "shanten": s, "ukeire_map": um, "ukeire": un, "edge": kind})
    with open(out, "w", encoding="utf-8") as f:
        for v in vecs:
            f.write(json.dumps(v) + "\n")
    from collections import Counter
    dist = Counter(v["shanten"] for v in vecs)
    print(f"生成 {len(vecs)} 条 -> {out}")
    print("向听分布:", dict(sorted(dist.items())))
    n_w_ukeire = sum(1 for v in vecs if v["ukeire"])
    print(f"有进张: {n_w_ukeire}/{len(vecs)}")


if __name__ == "__main__":
    main()
