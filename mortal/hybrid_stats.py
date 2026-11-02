# -*- coding: utf-8 -*-
"""从 1v3 的 mjai 日志统计「拼接模型」的四项率：和了率 / 铳率 / 立直率 / 副露率。

日志格式（libriichi/src/arena/one_vs_three.rs:207-224 写出）：
    {log_dir}/{seed}_{key}_{a|b|c|d}.json.gz，**JSON Lines**，每行一个 mjai 事件。

怎么只统计拼接模型：
    文件里第一行 `start_game` 带 `names`，数组按座位顺序给出 4 个 agent 名。
    挑战者的名字一定以 `Hybrid` 开头 —— 用它定位座位，比靠 split 字母映射更稳。

口径（严格对齐 libriichi/src/stat.rs，即 run_test.py 用的那份）：
    round  = start_kyoku 的个数（含连荘，每个 StartKyoku 都 +1）
    和了率 = hora 且 actor==seat / round
    铳率   = hora 且 target==seat 且 actor!=seat / round
    立直率 = reach 且 actor==seat / round
    副露率 = **该局至少副露过一次** 的局数 / round        ← stat.rs:407-410
             （不是"副露次数/局"！chi/pon/daiminkan 都算，暗杠不算）
    流局率 = ryukyoku 事件数 / round
    平均和了巡目 = 和了时自家已舍牌数 / 和了次数           ← stat.rs:291-293, 340

用法：
    python hybrid_stats.py                        # 统计 1v3_logs_hybrid
    python hybrid_stats.py --log-dir 1v3_logs_hybrid_wide
    python hybrid_stats.py --seeds 10000:12500    # 只统计某个 seed 区间
"""
import argparse
import collections
import glob
import gzip
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

MELD_TYPES = ('chi', 'pon', 'daiminkan')
FILE_RE = re.compile(r'^(\d+)_(\d+)_([abcd])\.json\.gz$')


def _parse_one(path):
    """返回 (seat, counters, n_kyoku) 或 (None, err, None)。"""
    try:
        seat = None
        c = collections.Counter()
        melded_this_kyoku = False
        jun = 0
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                t = e.get('type')
                if t == 'start_game':
                    names = e.get('names') or []
                    for i, nm in enumerate(names):
                        if isinstance(nm, str) and nm.startswith('Hybrid'):
                            seat = i
                            break
                if t == 'start_kyoku':
                    c['kyoku'] += 1
                    melded_this_kyoku = False
                    jun = 0
                elif seat is None:
                    continue
                elif t == 'dahai' and e.get('actor') == seat:
                    jun += 1                        # 自家已舍牌数 = 巡目
                elif t == 'hora':
                    a, tg = e.get('actor'), e.get('target')
                    if a == seat:
                        c['hora'] += 1
                        c['agari_jun'] += jun
                        if tg == seat:
                            c['tsumo'] += 1        # 自摸: actor == target
                        else:
                            c['ron'] += 1          # 荣和
                    elif tg == seat:
                        c['deal_in'] += 1
                elif t == 'reach' and e.get('actor') == seat:
                    c['reach'] += 1
                elif t in MELD_TYPES and e.get('actor') == seat:
                    c['meld'] += 1
                    if not melded_this_kyoku:
                        c['meld_kyoku'] += 1       # 该局至少副露过一次
                        melded_this_kyoku = True
                elif t == 'kakan' and e.get('actor') == seat:
                    c['kakan'] += 1
                elif t == 'ryukyoku':
                    c['ryukyoku'] += 1
        if seat is None:
            return None, 'no Hybrid in names', None
        return seat, c, c['kyoku']
    except Exception as ex:                      # 文件可能正在被写/损坏
        return None, f'{type(ex).__name__}: {ex}', None


def _worker(args):
    path, = args
    return _parse_one(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log-dir', default='1v3_logs_hybrid')
    ap.add_argument('--seeds', default=None, help='形如 10000:12500，只统计该区间（含头不含尾）')
    ap.add_argument('--key', default='auto',
                    help='只统计某个 seed-key（默认 auto = 取文件数最多的那个；all = 不过滤）')
    ap.add_argument('--workers', type=int, default=0, help='0 = 自动')
    ap.add_argument('--per-seat', action='store_true', help='额外打印按座位的分布（校验用）')
    args = ap.parse_args()

    if not os.path.isdir(args.log_dir):
        sys.exit(f'[FATAL] 目录不存在: {os.path.abspath(args.log_dir)}')

    lo, hi = None, None
    if args.seeds:
        a, b = args.seeds.split(':')
        lo, hi = int(a), int(b)

    files_all, key_cnt = [], collections.Counter()
    for p in glob.glob(os.path.join(args.log_dir, '*.json.gz')):
        m = FILE_RE.match(os.path.basename(p))
        if not m:
            continue
        sd = int(m.group(1))
        if lo is not None and not (lo <= sd < hi):
            continue
        files_all.append((p, sd, m.group(2)))
        key_cnt[m.group(2)] += 1
    if not files_all:
        sys.exit('[FATAL] 没有匹配的日志文件')

    # 目录里常混着早先不同 seed-key 的残留，必须只取一个 key
    if args.key == 'auto':
        use_key = key_cnt.most_common(1)[0][0]
    elif args.key == 'all':
        use_key = None
    else:
        use_key = args.key
    if use_key is not None and key_cnt[use_key] == 0:
        sys.exit(f'[FATAL] key {use_key} 在目录里没有文件；现有: {dict(key_cnt)}')

    files, seeds = [], set()
    for p, sd, k in files_all:
        if use_key is not None and k != use_key:
            continue
        files.append(p)
        seeds.add(sd)

    print(f'目录   : {args.log_dir}')
    if len(key_cnt) > 1:
        others = {k: v for k, v in key_cnt.items() if k != use_key}
        print(f'[!] 目录里有 {len(key_cnt)} 个 seed-key，只统计 {use_key}；'
              f'已排除: {others}')
    print(f'key    : {use_key if use_key else "ALL"}')
    print(f'文件数 : {len(files)}   去重 seed 数: {len(seeds)}   '
          f'seed 范围: {min(seeds)}~{max(seeds)}')
    print('解析中 ...', flush=True)

    wk = args.workers or min(8, os.cpu_count() or 2)
    agg = collections.Counter()
    seat_cnt = collections.Counter()
    errs = []
    done = 0
    with ProcessPoolExecutor(max_workers=wk) as ex:
        for seat, c, nk in ex.map(_worker, ((p,) for p in files), chunksize=32):
            done += 1
            if seat is None:
                if len(errs) < 8:
                    errs.append(c)
                continue
            seat_cnt[seat] += 1
            agg.update(c)
            if done % 2000 == 0:
                print(f'  {done}/{len(files)}', flush=True)

    kyoku = agg['kyoku']
    if not kyoku:
        sys.exit('[FATAL] 没有解析到任何小局')

    def pct(n):
        return f'{n:>7,}  {n / kyoku * 100:>6.2f}%'

    print()
    print('=' * 60)
    print('  拼接模型（挑战者座位）统计')
    print('=' * 60)
    print(f'  总局数      : {kyoku:,}')
    print(f'  和了率      : {pct(agg["hora"])}   (自摸 {agg["tsumo"]:,} / 荣和 {agg["ron"]:,})')
    print(f'  铳率        : {pct(agg["deal_in"])}')
    print(f'  立直率      : {pct(agg["reach"])}')
    print(f'  副露率      : {pct(agg["meld_kyoku"])}   (至少副露过一次的局，对齐 stat.rs)')
    print(f'    其中副露次数/局: {agg["meld"] / kyoku:.3f}  共 {agg["meld"]:,} 次'
          f'（加杠另计 {agg["kakan"]:,}）')
    print(f'  流局率      : {pct(agg["ryukyoku"])}')
    print(f'  平均和了巡目: {agg["agari_jun"] / agg["hora"]:.2f}' if agg['hora'] else '  平均和了巡目: -')
    print('-' * 60)
    print(f'  和了-放铳   : {agg["hora"] - agg["deal_in"]:+,} 次   '
          f'({(agg["hora"] - agg["deal_in"]) / kyoku * 100:+.2f} pp)')
    if args.per_seat:
        print()
        print('  挑战者座位分布（应大致均匀，用于校验座位轮换）:')
        for s in sorted(seat_cnt):
            print(f'    座位 {s}: {seat_cnt[s]:,} 局')
    if errs:
        print()
        print(f'  [!] 跳过 {len(errs)}+ 个无法解析的文件，样例: {errs[:3]}')
    print('=' * 60)


if __name__ == '__main__':
    main()
