"""Generate proper self-play training data using the OneVsThree arena."""
import sys
sys.path.insert(0, '.')
import os
import shutil
import torch
from model import Brain, DQN
from engine import MortalEngine
from libriichi.arena import OneVsThree
from config import config

version = config['control']['version']
print(f'Version: {version}')

mortal = Brain(
    version=version,
    conv_channels=config['resnet']['conv_channels'],
    num_blocks=config['resnet']['num_blocks'],
).eval()
dqn = DQN(version=version).eval()

# Load checkpoint if available
state_file = config['control']['state_file']
if os.path.exists(state_file):
    state = torch.load(state_file, weights_only=True, map_location='cpu')
    mortal.load_state_dict(state['mortal'])
    dqn.load_state_dict(state['current_dqn'])
    print(f'Loaded checkpoint from {state_file}')
else:
    print('No checkpoint found, using random weights')

device = torch.device('cpu')

# Two identical engines for self-play
engine1 = MortalEngine(
    mortal, dqn,
    is_oracle=False, version=version,
    device=device, name='p1',
    enable_amp=False,
)
engine2 = MortalEngine(
    mortal, dqn,
    is_oracle=False, version=version,
    device=device, name='p2',
    enable_amp=False,
)

log_dir = 'selfplay_data'
if os.path.isdir(log_dir):
    shutil.rmtree(log_dir)

num_games = 5  # Generate 5 games (each game = 1 kyoku in 1v3 format)
env = OneVsThree(
    disable_progress_bar=False,
    log_dir=log_dir,
)
env.py_vs_py(
    challenger=engine1,
    champion=engine2,
    seed_start=(10000, 0x2000),
    seed_count=num_games,
)

# List generated files
files = list(map(lambda p: os.path.join(log_dir, p), os.listdir(log_dir)))
print(f'Generated {len(files)} log files in {log_dir}')
for f in files[:5]:
    print(f'  {f}')
if len(files) > 5:
    print(f'  ... and {len(files) - 5} more')
