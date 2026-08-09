import gzip, json
import numpy as np
from libriichi.dataset import GameplayLoader
from dataloader import decode_invisible_obs

LOG_PATH = r'E:\天凤训练数据\2021\01\2021010100gm-00a9-0000-0af99d68.mjson'
TARGET = ('S', 2)   # 想看哪局改这里: 东1=('E',1) 南1=('S',1) 南3=('S',3)...

TILES = [f'{i}m' for i in range(1, 10)] + [f'{i}p' for i in range(1, 10)] \
        + [f'{i}s' for i in range(1, 10)] + ['东', '南', '西', '北', '白', '发', '中']

def hand_str(counts):
    return ' '.join(f'{TILES[t]}x{c}' for t, c in enumerate(counts) if c > 0)

raw = gzip.open(LOG_PATH, 'rt', encoding='utf-8').read()
print('verify 本局玩家:', json.loads(raw.splitlines()[0]).get('names'))

# 建 at_kyoku 序号 -> 真实局名 的映射 (连庄会占序号, 不能硬编码)
kyoku_map = []
for line in raw.splitlines():
    ev = json.loads(line)
    if ev.get('type') == 'start_kyoku':
        kyoku_map.append((ev['bakaze'], ev['kyoku'], ev.get('honba', 0)))
matches = [i for i, (b, k, _) in enumerate(kyoku_map) if (b, k) == TARGET]
if not matches:
    raise SystemExit(f'这局牌没有 {TARGET[0]}{TARGET[1]} 局 (可能连庄/东风战), kyoku_map={kyoku_map}')
target_idx = matches[0]
b, k, h = kyoku_map[target_idx]
print(f'目标局 {b}{k} 本场{h} = at_kyoku {target_idx}')

loader = GameplayLoader(version=4, oracle=True)
game = loader.load_gz_log_files([LOG_PATH])[0][0]
obs = game.take_obs()
inv = game.take_invisible_obs()
pid = game.take_player_id()
print(f'player_id={pid}, steps={len(obs)}, inv shape={inv[0].shape}')

# 全局自洽性检查: one-hot 向听 与 数值通道向听 必须处处一致 (验证行偏移理解)
bad = 0
for arr in inv:
    for k_ in range(3):
        b_ = 17 * k_
        if int(arr[b_+7:b_+14, 0].argmax()) != int(round(float(arr[b_+14, 0]) * 6)):
            bad += 1
print(f'shanten cross-check mismatches: {bad} (应为 0)')

# 打印目标局每一步的三家解码
at_kyoku = game.take_at_kyoku()
at_turns = game.take_at_turns()
last_t = None
for t in range(len(obs)):
    if at_kyoku[t] != target_idx:
        continue
    last_t = t
    cnt, bin_ = decode_invisible_obs(inv[t])
    row = ' '.join(
        f"{name}[{hand_str(cnt[k, :34])}]"
        for k, name in enumerate(('下家', '对家', '上家'))
    )
    print(f'第{t}步 第{at_turns[t]}巡: {row}')

# 目标局最后一步的详细打印 (含向听/待牌/赤/振听)
if last_t is not None:
    print(f'--- 目标局最后一个决策点 (第{last_t}步) 详情 ---')
    cnt, bin_ = decode_invisible_obs(inv[last_t])
    for k, name in enumerate(('下家', '对家', '上家')):
        waits = [TILES[i] for i in np.where(bin_[k, :34] > 0)[0]]
        akas = [a for a, f in zip(('5mr', '5pr', '5sr'), bin_[k, 34:37]) if f > 0]
        print(f'{name}: 手牌[{hand_str(cnt[k, :34])}] 共{cnt[k, :34].sum()}张 '
              f'向听={cnt[k, 34]} 待牌={waits} 赤={akas} 振听={bool(bin_[k, 37])}')

# 全步合法性断言
for t, arr in enumerate(inv):
    cnt, bin_ = decode_invisible_obs(arr)
    assert (cnt[:, :34] <= 4).all(), f'step {t}: 某牌超过4张'
    sums = cnt[:, :34].sum(-1)
    # 手牌数 = 13 - 3×副露组数，合法值为 13/10/7/4/1
    assert (sums >= 1).all() and ((sums - 1) % 3 == 0).all(), f'step {t}: {sums}'
    tenpai = cnt[:, 34] == 0
    if (~tenpai).any():
        assert (bin_[~tenpai][:, :34] == 0).all(), f'step {t}: 未听牌却有待牌'
print('全部断言通过')