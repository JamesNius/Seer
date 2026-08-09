"""
并行对战评估：起 N 个进程同时跑，最后合并统计
用法: python review/run_test_parallel.py [总局数=100] [进程数=4]
例:   python review/run_test_parallel.py 400 4
"""
import sys
import os
import shutil
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 100
n_proc = int(sys.argv[2]) if len(sys.argv) > 2 else 4

base_log = os.path.abspath(config['test_play']['log_dir'])
per_dirs = [f"{base_log}_p{i}" for i in range(n_proc)]

# 局数均分，余数分给前几个进程
per_games = [n_games // n_proc + (1 if i < n_games % n_proc else 0) for i in range(n_proc)]

print(f"spawning {n_proc} processes, games each: {per_games}", flush=True)
procs = []
for i in range(n_proc):
    env = os.environ.copy()
    env['MORTAL_TEST_LOG_DIR'] = per_dirs[i]
    procs.append(subprocess.Popen(
        [sys.executable, 'review/run_test.py', str(per_games[i])],
        env=env,
    ))

for p in procs:
    p.wait()

# 合并所有进程的日志到一个目录，统一算统计
merged = f"{base_log}_merged"
if os.path.isdir(merged):
    shutil.rmtree(merged)
os.makedirs(merged)
for d in per_dirs:
    if not os.path.isdir(d):
        continue
    for f in os.listdir(d):
        os.rename(os.path.join(d, f), os.path.join(merged, f"{os.path.basename(d)}_{f}"))
    os.rmdir(d)

from libriichi.stat import Stat
stat = Stat.from_dir(merged, 'mortal')

print(f'\n===== 对战结果 (合并 {n_games} 局, {n_proc} 进程) =====')
print(f'平均顺位: {stat.avg_rank:.4f}')
print(f'平均 PT:  {stat.avg_pt([6, 4, 2, 0]):.4f}')
print(f'1位率:    {stat.rank_1_rate:.2%}')
print(f'2位率:    {stat.rank_2_rate:.2%}')
print(f'3位率:    {stat.rank_3_rate:.2%}')
print(f'4位率:    {stat.rank_4_rate:.2%}')
print(f'和率:     {stat.agari_rate:.2%}')
print(f'铳率:     {stat.houjuu_rate:.2%}')
print(f'立直率:   {stat.riichi_rate:.2%}')
print(f'副露率:   {stat.fuuro_rate:.2%}')
print(f'平均和了巡目: {stat.avg_agari_jun:.1f}')