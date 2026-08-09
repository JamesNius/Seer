"""
快速对战评估：当前模型 vs baseline
用法: python review/run_test.py [游戏局数，默认 40]
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model import Brain, DQN
from player import TestPlayer
from config import config

# 用最新的训练模型
state_file = config['control']['state_file']
state = torch.load(state_file, weights_only=True, map_location='cpu')
cfg = state['config']
version = cfg['control'].get('version', 1)

mortal = Brain(version=version, conv_channels=cfg['resnet']['conv_channels'],
               num_blocks=cfg['resnet']['num_blocks']).eval()
dqn = DQN(version=version).eval()
mortal.load_state_dict(state['mortal'])
dqn.load_state_dict(state['current_dqn'])

device = torch.device(config['control']['device'])

n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 40
test_player = TestPlayer()
stat = test_player.test_play(n_games, mortal, dqn, device)
mortal.train(); dqn.train()

print(f'\n===== 对战结果 ({n_games} 局) =====')
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
