# -*- coding: utf-8 -*-
"""拼接模型 1v3 测试：Hybrid(前 N 巡 SeerX + 其余 Seer) 打 Seer1.0。

与 one_vs_three.py 的区别只有一处：challenger 不是单个权重，而是 HybridEngine。
不改 one_vs_three.py，也不依赖 config.toml（全部走命令行参数）。

用法（在 mortal/ 目录下）：
    python one_vs_three_hybrid.py --seeds 100
    python one_vs_three_hybrid.py --seeds 100 --gate-turn 6 --dry-run

输出：每 chunk 打印累计成绩；最后打印四位率/一位率与门控命中分布。
"""
import argparse
import secrets
import sys
import time

import prelude  # noqa: F401  (与 one_vs_three.py 保持一致的环境初始化)
import numpy as np
import torch

from libriichi.arena import OneVsThree
from model_loader import build_engine_from_state
from hybrid_engine import HybridEngine

DEFAULT_SEERX = 'checkpoints/SeerX1.0bug修复版.pth'
DEFAULT_SEER = 'checkpoints/seer.1.0.收敛版.pth'
PT_TABLE = np.array([90, 45, 0, -135])


def build(path, name, device, amp, guard):
    if not (path.endswith('.pth')):
        raise SystemExit(f'[FATAL] 不是 .pth: {path}')
    import os
    if not os.path.exists(path):
        raise SystemExit(f'[FATAL] 文件不存在: {os.path.abspath(path)}')
    eng, is_policy = build_engine_from_state(
        path,
        device=torch.device(device),
        name=name,
        enable_compile=False,
        enable_amp=amp,
        enable_rule_based_agari_guard=guard,
    )
    print(f'  [ok] {name:<22} {path}   policy={is_policy}')
    return eng


def summarize(rankings):
    total = rankings.sum()
    rates = rankings / total
    avg_rank = float(rankings @ np.arange(1, 5) / total)
    avg_pt = float(rankings @ PT_TABLE / total)
    return total, rates, avg_rank, avg_pt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seerx', default=DEFAULT_SEERX, help='前 N 巡使用的权重')
    ap.add_argument('--seer', default=DEFAULT_SEER, help='立直后/南4/其余 使用的权重')
    ap.add_argument('--gate-turn', type=int, default=6, help='SeerX 只在前 N 巡生效')
    ap.add_argument('--gate-mode', choices=['early', 'wide'], default='early',
                    help='early = SeerX 只盖前 N 巡(v3); '
                         'wide = 南4之前全程 SeerX, 南4后仍是前 N 巡(v4)')
    ap.add_argument('--late-from', type=int, default=0,
                    help='>0 时开启"深巡窗口"：巡目 >= 该值且平静也用 SeerX（仅南4之前）')
    ap.add_argument('--base', choices=['v3', 'v21'], default='v3',
                    help='v3 = 南4+需无副露才 SeerX, 自家立直含 ch878; '
                         'v21 = 南4+ 一律 Seer, 自家立直只认 ch869（与 +1.472 那版一致）')
    ap.add_argument('--seeds', type=int, default=100, help='总 seed 数（1 seed = 4 局）')
    ap.add_argument('--chunk', type=int, default=25, help='每多少 seed 打一次累计')
    ap.add_argument('--seed-key', type=int, default=-1, help='-1 = 随机并打印出来')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--log-dir', default='1v3_logs_hybrid')
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--no-agari-guard', action='store_true')
    ap.add_argument('--dry-run', action='store_true', help='只验证权重能加载 + 跑 1 seed')
    ap.add_argument('--seed-start', type=int, default=10000)
    args = ap.parse_args()

    key = args.seed_key if args.seed_key != -1 else secrets.randbits(64)
    guard = not args.no_agari_guard

    print('=' * 68)
    print(f'gate-mode  = {args.gate_mode}   gate-turn = {args.gate_turn}'
          f'   late-from = {args.late_from}   base = {args.base}')
    if args.gate_mode == 'early':
        if args.base == 'v3':
            print('  规则: 前 N 巡 且 无人立直(含自家/ch878) 且 (南4需无副露) -> SeerX')
        else:
            print('  规则: 前 N 巡 且 无人立直(自家只认 ch869) 且 南4一律 Seer -> SeerX')
        if args.late_from > 0:
            print(f'        + 深巡窗口: 巡目 >= {args.late_from} 且同样平静 -> SeerX'
                  f'  (仅南4之前)')
            print(f'        => 南4之前 Seer 只剩 巡目 {args.gate_turn + 1}~'
                  f'{args.late_from - 1} 这一段')
    else:
        print('  规则: 南4之前 无他家立直 -> SeerX(全程); '
              '南4及之后 前 N 巡且无他家立直且无副露/杠 -> SeerX')
    print(f'seed-key  = {key}   (固定它才能复现)')
    print('=' * 68)
    print('building engines ...')
    early = build(args.seerx, 'SeerX', args.device, args.amp, guard)
    late = build(args.seer, 'Seer', args.device, args.amp, guard)
    # 对手 = Seer1.0。和 late 同一个权重，但必须是独立实例（各自持有 BN 统计量）。
    champ = build(args.seer, 'Seer1.0(champ)', args.device, args.amp, guard)

    chal = HybridEngine(early, late, gate_turn=args.gate_turn, mode=args.gate_mode,
                        late_from=args.late_from, base=args.base)
    print(f'challenger = {chal.name}')
    print()

    if args.dry_run:
        args.seeds = 1

    total_seeds = args.seeds
    chunk = max(1, min(args.chunk, total_seeds))
    acc = np.zeros(4, dtype=np.int64)
    t0 = time.time()
    done = 0

    # 每 chunk 追加一行到磁盘：中途关窗口也留有累计结果
    import os
    os.makedirs(args.log_dir, exist_ok=True)
    summary_path = os.path.join(args.log_dir, 'hybrid_summary.txt')

    def append_summary(tag, n, rates, avg_rank, avg_pt, g):
        line = (f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {tag} '
                f'mode={args.gate_mode} base={args.base} late_from={args.late_from} '
                f'seerx={os.path.basename(args.seerx)} seer={os.path.basename(args.seer)} '
                f'gate={args.gate_turn} key={key} N={n} '
                f'r={acc.tolist()} 1st={rates[0]*100:.2f}% 4th={rates[3]*100:.2f}% '
                f'avg_rank={avg_rank:.4f} avg_pt={avg_pt:+.3f} '
                f'seerx_share={g["early_pct"]:.1f}%\n')
        with open(summary_path, 'a', encoding='utf-8') as f:
            f.write(line)

    while done < total_seeds:
        take = min(chunk, total_seeds - done)
        env = OneVsThree(disable_progress_bar=False, log_dir=args.log_dir)
        r = env.py_vs_py(
            challenger=chal,
            champion=champ,
            seed_start=(args.seed_start + done, key),
            seed_count=take,
        )
        acc += np.asarray(r, dtype=np.int64)
        done += take
        n, rates, avg_rank, avg_pt = summarize(acc)
        g = chal.stats()
        print(f'[seed {done:>4}/{total_seeds}] N={n:>5} 局  '
              f'1位 {rates[0]*100:5.2f}%  2位 {rates[1]*100:5.2f}%  '
              f'3位 {rates[2]*100:5.2f}%  4位 {rates[3]*100:5.2f}%  |  '
              f'avg_rank {avg_rank:.4f}  avg_pt {avg_pt:+.3f}  |  '
              f'SeerX 占比 {g["early_pct"]:.1f}%', flush=True)
        append_summary(f'chunk{done}', n, rates, avg_rank, avg_pt, g)

    n, rates, avg_rank, avg_pt = summarize(acc)
    g = chal.stats()
    print()
    print('=' * 68)
    print(f'总计           : {n} 局  用时 {(time.time()-t0)/60:.1f} 分钟')
    print(f'gate-mode      : {args.gate_mode}   gate-turn {args.gate_turn}   '
          f'late-from {args.late_from}   base {args.base}')
    print(f'rankings       : {acc.tolist()}')
    print(f'1/2/3/4 位率   : {rates[0]*100:.2f}% / {rates[1]*100:.2f}% / '
          f'{rates[2]*100:.2f}% / {rates[3]*100:.2f}%')
    print(f'avg_rank       : {avg_rank:.4f}')
    print(f'avg_pt         : {avg_pt:+.3f}   (口径 [90,45,0,-135])')
    print(f'四位率-一位率  : {(rates[3]-rates[0])*100:+.2f} pp')
    print('-' * 68)
    print(f'门控命中       : SeerX {g["early"]} 次 ({g["early_pct"]:.1f}%)  '
          f'Seer {g["late"]} 次 ({100-g["early_pct"]:.1f}%)')
    print(f'  走 SeerX (开局窗)         : {g["why_turn"]}')
    print(f'  走 SeerX (深巡窗)         : {g["why_late"]}'
          + ('' if args.late_from else '   (未开启)'))
    print(f'  因立直切 Seer             : {g["why_riichi"]}')
    print(f'  南4/之后+副露或杠切 Seer  : {g["why_final_meld"]}')
    print(f'  南4/之后一律切 Seer       : {g["why_final"]}'
          + ('' if args.base == 'v21' else '   (v3 不用这条)'))
    print(f'  巡目已过切 Seer           : {g["why_turn_over"]}')
    print(f'  南4/之后决策数 {g["n_final"]}，其中走了 SeerX 的 {g["n_final_seerx"]}'
          f' ({(100.0*g["n_final_seerx"]/g["n_final"] if g["n_final"] else 0):.1f}%)')
    print(f'seed_key       : {key}')
    print('=' * 68)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n[interrupted]')
        sys.exit(0)
