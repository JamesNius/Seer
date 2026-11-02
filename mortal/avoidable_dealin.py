# -*- coding: utf-8 -*-
"""从 1v3 牌谱里估「无谓铳」有多大：放铳那一手，本来能不能用一张差不多好的安全牌代替？

为什么只做离线统计：这个数直接决定「降铳率而不掉和率」的上限。先量，再决定投不投。

日志事实（已实测确认）：
  - `dahai` 事件的 meta 里有 q_values 和 mask_bits
  - mask_bits 展开后的置位位置 = 动作索引；q_values 按顺序对应
  - 0~33 = 34 种牌，34/35/36 = 赤 5m/5p/5s，37+ = 吃/碰/杠/立直等
  - 实测 argmax==实际打出的牌 99.3%（剩下 0.7% 全因赤5 用了 34~36）

安全定义（**只用决策点当时可见的信息，绝不用"谁赢了"这种后见之明**）：
  Tier-A  `genbutsu_riichi` : 对**每一个已立直的家**都是现物（场上无人立直时退化为 Tier-B）
  Tier-B  `genbutsu_all`    : 对**全部三家**都是现物（最保守）

用法：
    python avoidable_dealin.py                       # 全量
    python avoidable_dealin.py --limit 500           # 先跑 500 局
"""
import argparse
import collections
import glob
import gzip
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

CHALLENGER_PREFIX = 'Hybrid'

TILES = {}
for i in range(9):
    TILES[f'{i+1}m'] = i
    TILES[f'{i+1}p'] = 9 + i
    TILES[f'{i+1}s'] = 18 + i
TILES.update({'E': 27, 'S': 28, 'W': 29, 'N': 30, 'P': 31, 'F': 32, 'C': 33})
for c in 'mps':
    for n in range(1, 10):
        TILES[f'{n}{c}r'] = TILES[f'{n}{c}']

# 动作索引 -> 牌 id（落在一张牌上时）；其余返回 None
def action_to_tile(a):
    if 0 <= a <= 33:
        return a
    if a == 34:
        return 4
    if a == 35:
        return 13
    if a == 36:
        return 22
    return None


def expand_mask(mb):
    mb = int(mb)
    return [i for i in range(64) if (mb >> i) & 1]


EPS_LIST = (0.25, 0.5, 1.0, 2.0, 3.0)


def process_file(path):
    """返回 Counter 或 (None, err)。"""
    c = collections.Counter()
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            seat = None
            kawa = [[] for _ in range(4)]      # 各家舍牌（牌 id）
            riichi = [False] * 4
            last_dahai_by_seat = None          # 最近一次自家 dahai 的 (pai, meta)
            for line in f:
                if not line.strip():
                    continue
                e = json.loads(line)
                t = e.get('type')
                if t == 'start_game':
                    for i, nm in enumerate(e.get('names') or []):
                        if isinstance(nm, str) and nm.startswith(CHALLENGER_PREFIX):
                            seat = i
                    continue
                if t == 'start_kyoku':
                    kawa = [[] for _ in range(4)]
                    riichi = [False] * 4
                    last_dahai_by_seat = None
                    c['kyoku'] += 1
                    continue
                if seat is None:
                    continue
                if t == 'reach':
                    riichi[e.get('actor')] = True
                elif t == 'dahai':
                    a = e.get('actor')
                    ti = TILES.get(e.get('pai'), -1)
                    if a == seat:
                        last_dahai_by_seat = (ti, e.get('meta') or {}, [list(k) for k in kawa],
                                              'dahai')
                    if ti >= 0:
                        kawa[a].append(ti)
                elif t == 'kakan' and e.get('actor') == seat:
                    # 加杠会被搶槓，也是"自家打出/亮出的牌被荣和"的一种
                    last_dahai_by_seat = (TILES.get(e.get('pai'), -1), e.get('meta') or {},
                                          [list(k) for k in kawa], 'kakan')
                elif t == 'hora':
                    win, tgt = e.get('actor'), e.get('target')
                    if tgt != seat or win == seat:
                        continue
                    c['dealin'] += 1
                    # 铳时是否有立直家
                    others_riichi = [k for k in range(4) if k != seat and riichi[k]]
                    if others_riichi:
                        c['dealin_vs_riichi'] += 1
                    if last_dahai_by_seat is None:
                        c['no_prev_event'] += 1
                        continue
                    ti, meta, kawa_snap, kind = last_dahai_by_seat
                    q, mb = meta.get('q_values'), meta.get('mask_bits')
                    if not q:
                        c['no_q'] += 1
                        continue
                    if mb is None:
                        c['no_mask'] += 1
                        continue
                    legal = expand_mask(mb)
                    if len(legal) != len(q):
                        c['bad_mask'] += 1
                        continue
                    # 每个牌 id 的最好 Q（赤5 与非赤5 合并取 max）
                    best_q = {}
                    for k, a in enumerate(legal):
                        tid = action_to_tile(a)
                        if tid is None:
                            continue
                        if tid not in best_q or q[k] > best_q[tid]:
                            best_q[tid] = q[k]
                    if ti not in best_q:
                        c['bad_tile'] += 1
                        continue
                    q_best = max(best_q.values())
                    # 铳的那张牌本身是不是现物（若是 -> 筋引挂/无解，不算可避免）
                    if all(ti in kawa_snap[k] for k in range(4) if k != seat):
                        c['dealin_on_genbutsu_all'] += 1
                    safe_all = {t: all(t in kawa_snap[k] for k in range(4) if k != seat)
                                for t in best_q}
                    # 只有"场上有立直家"时，现物才是可判定的安全牌。
                    # 无人立直时不知道谁听牌，现物没有安全含义 -> 不计入可避免统计。
                    if others_riichi:
                        c['scored_ri'] += 1
                        safe_ri = {t: all(t in kawa_snap[k] for k in others_riichi)
                                   for t in best_q}
                        for eps in EPS_LIST:
                            n_alt = sum(1 for t, v in best_q.items()
                                        if t != ti and v >= q_best - eps)
                            c[f'hasalt_ri_{eps}'] += 1 if n_alt > 0 else 0
                            ok_ri = any(safe_ri[t] and t != ti and v >= q_best - eps
                                        for t, v in best_q.items())
                            if ok_ri:
                                c[f'avoid_ri_{eps}'] += 1
                    else:
                        c['scored_nori'] += 1
                        for eps in EPS_LIST:
                            ok_all = any(safe_all[t] and t != ti and v >= q_best - eps
                                         for t, v in best_q.items())
                            if ok_all:
                                c[f'avoid_all_{eps}'] += 1
                    c['scored'] += 1
        return c, None
    except Exception as ex:
        return None, f'{type(ex).__name__}: {ex}'


def _worker(p):
    return process_file(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log-dir', default='1v3_logs_hybrid_band13')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.log_dir, '*.json.gz')))
    if not files:
        sys.exit(f'[FATAL] 没有日志: {args.log_dir}')
    if args.limit:
        files = files[:args.limit]
    print(f'目录 : {args.log_dir}')
    print(f'文件数: {len(files)}')
    print('解析中 ...', flush=True)

    wk = args.workers or min(8, os.cpu_count() or 2)
    agg = collections.Counter()
    errs = []
    done = 0
    with ProcessPoolExecutor(max_workers=wk) as ex:
        for c, err in ex.map(_worker, files, chunksize=32):
            done += 1
            if c is None:
                if len(errs) < 5:
                    errs.append(err)
                continue
            agg.update(c)
            if done % 2000 == 0:
                print(f'  {done}/{len(files)}', flush=True)

    dealin = agg['dealin']
    scored = agg['scored']
    if not dealin:
        sys.exit('[FATAL] 没有放铳事件')

    print()
    print('=' * 64)
    print('  放铳分析（挑战者视角，只用决策时可见信息）')
    print('=' * 64)
    ky = max(agg['kyoku'], 1)
    print(f'  总局数              : {ky:,}')
    print(f'  放铳总数            : {dealin:,}   (铳率 {dealin/ky*100:.2f}%)')
    print(f'  其中对手有立直      : {agg["dealin_vs_riichi"]:,} '
          f'({agg["dealin_vs_riichi"]/dealin*100:.1f}%)')
    print(f'  铳的牌对三家都是现物: {agg["dealin_on_genbutsu_all"]:,}  <- 不可能躲，不算')
    print(f'  可分析(有 q_values) : {scored:,} ({scored/dealin*100:.1f}%)')
    if agg['bad_mask'] or agg['bad_tile'] or agg['no_q'] or agg['no_mask'] \
            or agg['no_prev_event']:
        print(f'  跳过: no_prev_event={agg["no_prev_event"]} no_q={agg["no_q"]} '
              f'no_mask={agg["no_mask"]} bad_mask={agg["bad_mask"]} '
              f'bad_tile={agg["bad_tile"]}')
    sri = max(agg['scored_ri'], 1)
    print()
    print(f'  【有立直家、可判定安全】的放铳 {agg["scored_ri"]:,} 次'
          f'（占可分析 {agg["scored_ri"]/max(scored,1)*100:.1f}%）')
    print(f'    {"ε":>6} {"有替代牌":>10} {"其中更安全":>12} {"折算铳率可降":>14}')
    for eps in EPS_LIST:
        ha = agg[f'hasalt_ri_{eps}']
        av = agg[f'avoid_ri_{eps}']
        pp = av / ky * 100
        print(f'    {eps:>6.2f} {ha:>6,}({ha/sri*100:>4.1f}%) '
              f'{av:>6,}({av/sri*100:>4.1f}%) {pp:>10.3f} pp')
    print()
    print(f'  【无人立直】的放铳 {agg["scored_nori"]:,} 次 —— 不知道谁听牌，'
          f'现物无安全含义，无法判定，故不计入上限')
    print('=' * 64)
    if errs:
        print(f'  [!] {len(errs)} 个文件出错，样例: {errs[:2]}')


if __name__ == '__main__':
    main()
